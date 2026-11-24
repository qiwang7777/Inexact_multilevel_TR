"""One shared uncorrected multilevel IRMTR + Moreau continuation algorithm.

The recursive trust-region logic is independent of the control discretization.
P0 semilinear and P1 elliptic models supply only PDE/metric/Hessian operations.
The coarse Hessian is always Galerkin: B_c = R B_f P.
"""

import time
import numpy as np
from .geometry import p0_inner, p0_norm, element_geometry
from .transfer_operators import build_transfer_operators
from .regularization import moreau_value, moreau_grad, moreau_hessian_diagonal, prox_phi
from .types import Config, Evaluation, Level, InnerResult, OuterResult
from .hessian_models import LBFGSHessian, GalerkinHessian, HessianPlusDiagonal
from .gradient_errors import semilinear_gradient_certificate
from .pde import semilinear


class IRMTRMoreau:
    def __init__(self, meshes, config=None, model=None, backend=None):
        self.cfg = config or Config()
        self.backend = backend
        self.problem = semilinear if model is None else model
        self.control_space = 'p0' if backend is None else backend.control_space
        self.levels = [Level(mesh) for mesh in meshes]
        self.meshes = meshes  # convenient read-only view for experiment runners
        self.pde = getattr(backend, 'pde', None)
        for ell in range(1, len(self.levels)):
            R, P = build_transfer_operators(
                meshes[ell-1], meshes[ell], control_space=self.control_space)
            self.levels[ell].R, self.levels[ell].P = R, P
        self.history = []  # finest/coarse iteration trace
        self.trace = self.history  # same object, convenient for both runners
        self.hessian_stats = {'builds':0, 'products':0}
        self.cache_hits = 0
        self.gradient_evals = 0
        self.max_grad_error = 0.0
        self.max_grad_ratio = 0.0
        self.gradient_refinements = 0
        self.max_gradient_error = 0.0
        self.max_gradient_ratio = 0.0
        self.lbfgs_pairs = {ell: [] for ell in range(len(self.levels))}
        self.lbfgs_updates = {ell: 0 for ell in range(len(self.levels))}
        self.counters = dict(eligible=0, attempted=0, used=0, fallback=0,
                             coarse_accepted_finest=0)
        self.cost = {
            ell:{'eval_calls':0,'eval_time':0.0,
                 'obj_calls':0,'obj_time':0.0,
                 'reused_trial_states':0,
                 'exact_hessian_builds':0,
                 'exact_hessian_products':0}
            for ell in range(len(self.levels))
            }


    def objective(self, ell, z, mu, return_state=False, u_initial=None):
        """Trial objective. Optionally return its converged PDE state for reuse.

        State reuse is safe because objective() and evaluate() use the same
        control, mesh, and state solver tolerance.
        """
        if self.backend is not None:
            return self.backend.objective(self, ell, z, mu,
                                          return_state=return_state,
                                          u_initial=u_initial)
        t0 = time.perf_counter()
        c = self.cfg
        u = None
        if not np.all(np.isfinite(z)):
            result = np.inf
        else:
            mesh = self.levels[ell].mesh
            if return_state:
                u, J = self.problem.solve_state(mesh, z, tol=c.state_tol,
                                   u0=u_initial, return_jacobian=True)
            else:
                u = self.problem.solve_state(mesh, z, tol=c.state_tol, u0=u_initial)
                J = None
            result = self.problem.value_f(mesh, z, u, c.alpha, c.desired_state)
            if not c.smooth_only:
                result += moreau_value(mesh, z, mu, c.beta)
        self.cost[ell]['obj_calls'] += 1
        self.cost[ell]['obj_time'] += time.perf_counter()-t0
        return (result, (u, J)) if return_state else result

    def evaluate(self, ell, z, mu, tau, state=None, u_initial=None,
                 p_initial=None, delta=None):
        """Inexact state/adjoint, CERTIFIED gradient error, optional forcing.

        If delta is supplied, require e <= kappa_g min(delta, ||G_mu||)
        for nonterminating iterations; refine PDE solves if needed.
        The certified bound refers to the exactly discretized FE gradient.
        """
        if self.backend is not None:
            return self.backend.evaluate(self, ell, z, mu, tau,
                                         state=state, u_initial=u_initial,
                                         p_initial=p_initial, delta=delta)
        t0 = time.perf_counter()
        c = self.cfg
        mesh = self.levels[ell].mesh
        state_tol = c.state_tol
        adjoint_tol = c.adjoint_tol
        if state is not None:
            u, J = state if isinstance(state, tuple) else (state, None)
            self.cost[ell]['reused_trial_states'] += 1
            if J is None:
                _, J = self.problem.assemble_nonlinear_terms(mesh, u)
            p = self.problem.solve_adjoint(mesh, u, w_value=c.desired_state,
                                           rtol=adjoint_tol, J=J, p0=p_initial)
            g = self.problem.reduced_gradient(mesh, z, p, c.alpha)
        else:
            u, p, g, J = self.problem.compute_inexact_gradient(
                mesh, z, alpha=c.alpha, w_value=c.desired_state,
                state_tol=state_tol, adjoint_tol=adjoint_tol,
                u0=u_initial, p0=p_initial, return_jacobian=True)
        for refinement in range(12):
            err, _, _ = semilinear_gradient_certificate(
                mesh, z, u, p, c.desired_state, J=J)
            G = g if c.smooth_only else g + moreau_grad(z, mu, c.beta)
            ng = self.norm(ell, G)
            # No dynamic condition at original-stationarity diagnostic calls.
            # At an inner terminating iterate, ng+err<=tau suffices.
            limit = c.kappa_g*min(float(delta), ng) if delta is not None else np.inf
            if ng+err <= tau or err <= limit*(1+1e-12):
                break
            if limit <= 0 or refinement == 11:
                raise RuntimeError('Unable to certify semilinear dynamic gradient forcing')
            # Re-solve both PDE equations more accurately, warm-starting at
            # the current approximations. Recompute state/J before adjoint.
            state_tol *= 0.1
            adjoint_tol *= 0.1
            u, p, g, J = self.problem.compute_inexact_gradient(
                mesh, z, alpha=c.alpha, w_value=c.desired_state,
                state_tol=state_tol, adjoint_tol=adjoint_tol,
                u0=u, p0=p, return_jacobian=True)
            self.gradient_refinements += 1
        F = self.problem.value_f(mesh, z, u, c.alpha, c.desired_state)
        if not c.smooth_only:
            F += moreau_value(mesh, z, mu, c.beta)
        result = Evaluation(F, g, G, err, u, p, J)
        self.cost[ell]['eval_calls'] += 1
        self.cost[ell]['eval_time'] += time.perf_counter()-t0
        self.max_gradient_error = max(self.max_gradient_error, float(err))
        if delta is not None and ng+err > tau and limit > 0:
            self.max_gradient_ratio = max(self.max_gradient_ratio, err/limit)
        return result

    def inner(self, ell, x, y):
        mesh = self.levels[ell].mesh
        return (p0_inner(mesh, x, y) if self.backend is None
                else self.backend.inner(mesh, x, y))

    def norm(self,ell,z):
        return (p0_norm(self.levels[ell].mesh,z) if self.backend is None
                else self.backend.norm(self.levels[ell].mesh,z))

    def model_diagonal(self, z, mu):
        """Positive diagonal B approximating curvature of f + Moreau(phi).

        Away from soft-threshold / box transitions, the Moreau curvature is
        either 1/mu or 0. At transition points we take 1/mu conservatively.
        """
        c = self.cfg
        if c.smooth_only:
            return np.full_like(z,c.curvature,dtype=float)
        else:
            return c.curvature + moreau_hessian_diagonal(z, mu, c.beta)

    def reset_lbfgs(self):
        """Reset across distinct Moreau smoothing stages, not on recursion."""
        self.lbfgs_pairs = {ell: [] for ell in range(len(self.levels))}
        self.lbfgs_updates = {ell: 0 for ell in range(len(self.levels))}

    def add_lbfgs_pair(self, ell, s, y):
        """Use the smooth reduced gradient difference (weighted L2 geometry)."""
        if self.cfg.hessian_mode != 'lbfgs':
            return False
        mesh = self.levels[ell].mesh
        sy = self.inner(ell, s, y)
        ns, ny = self.norm(ell, s), self.norm(ell, y)
        # Skip noisy, negative or very weak secant curvature.
        if not np.isfinite(sy) or ns <= 1e-14 or ny <= 1e-14:
            return False
        if sy <= 1e-8 * ns * ny:
            return False
        mem = self.lbfgs_pairs[ell]
        mem.append((s.copy(), y.copy()))
        if len(mem) > self.cfg.lbfgs_memory:
            del mem[:-self.cfg.lbfgs_memory]
        self.lbfgs_updates[ell] += 1
        return True

    def make_model(self, ell, z, mu, ev=None):
        """Build diagonal, L-BFGS, or exact reduced Hessian model.

        In exact mode, ev provides the already computed state and adjoint,
        avoiding repeated nonlinear state/adjoint solves at this control.
        """
        c = self.cfg
        if c.hessian_mode == 'diagonal':
            return self.model_diagonal(z, mu)
        if self.backend is not None:
            return self.backend.make_model(self, ell, z, mu, ev=ev)
        if c.hessian_mode == 'lbfgs':
            areas, _ = element_geometry(self.levels[ell].mesh)
            H = LBFGSHessian(areas, c.curvature, self.lbfgs_pairs[ell])
        elif c.hessian_mode == 'exact':
            mesh = self.levels[ell].mesh
            if ev is not None and ev.u is not None and ev.p is not None:
                u, p = ev.u, ev.p
            else:
                u, p, _ = self.problem.compute_inexact_gradient(
                    mesh, z, alpha=c.alpha, w_value=c.desired_state,
                    state_tol=c.state_tol, adjoint_tol=c.adjoint_tol
                )
            H = self.problem.ExactReducedHessian(mesh, u, p, c.alpha,
                                    stats=self.cost[ell],
                                    J=(ev.J if ev is not None else None))
        else:
            raise ValueError('unknown hessian_mode')
        if c.smooth_only:
            return H
        D = moreau_hessian_diagonal(z, mu, c.beta)
        return HessianPlusDiagonal(H, D)

    @staticmethod
    def hessian_matvec(B, v):
        return B * v if isinstance(B, np.ndarray) else B.matvec(v)

    @staticmethod
    def hessian_bound(B):
        return float(np.max(B)) if isinstance(B, np.ndarray) else B.bound

    def tr_cg_step(self, ell, G, B, delta):
        """Preconditioned Steihaug CG in the weighted P0 L2 geometry.

        Preserves the original trust-region constraint and supports indefinite
        Hessians through negative-curvature boundary termination.  For an
        L-BFGS model without a preconditioner this reduces to ordinary CG.
        """
        inner = lambda u, v: self.inner(ell, u, v)
        s = np.zeros_like(G)
        r = G.copy()
        rnorm_initial = self.norm(ell, r)
        if rnorm_initial == 0 or delta <= 0:
            return s
        diag = getattr(B, 'preconditioner_diagonal', None)
        if diag is None:
            apply_Minv = lambda x: x.copy()
        else:
            diag = np.maximum(np.asarray(diag), 1e-12)
            apply_Minv = lambda x: x / diag
        zr = apply_Minv(r)
        rz = inner(r, zr)
        d = -zr

        def boundary_step(s, d):
            aa = inner(d, d)
            bb = 2.0 * inner(s, d)
            cc = inner(s, s) - delta**2
            disc = max(0.0, bb**2 - 4.0 * aa * cc)
            t = (-bb + np.sqrt(disc)) / (2.0 * aa)
            return s + t*d

        for _ in range(self.cfg.lbfgs_cg_maxiter):
            Bd = B.matvec(d)
            dBd = inner(d, Bd)
            if not np.isfinite(dBd) or dBd <= 0:
                return boundary_step(s, d)
            alpha = rz / dBd
            s_next = s + alpha*d
            if self.norm(ell, s_next) >= delta:
                return boundary_step(s, d)
            r_next = r + alpha*Bd
            if self.norm(ell, r_next) <= self.cfg.lbfgs_cg_rtol * rnorm_initial:
                return s_next
            zr_next = apply_Minv(r_next)
            rz_next = inner(r_next, zr_next)
            d = -zr_next + (rz_next/rz)*d
            s, r, rz = s_next, r_next, rz_next
        return s

    def cauchy_step(self, ell, G, B, delta, return_pred=False):
        """Cauchy safeguard. Reuse <G,BG> in its predicted reduction."""
        gnorm = self.norm(ell, G)
        if delta <= 0 or gnorm == 0:
            s = np.zeros_like(G)
            return (s, 0.0) if return_pred else s
        gBg = self.inner(ell, G, self.hessian_matvec(B, G))
        t = delta/gnorm
        if gBg > 0:
            t = min(t, gnorm*gnorm/gBg)
        s = -t*G
        predicted = t * gnorm**2 - .5*t*t*gBg
        return (s, predicted) if return_pred else s

    def model_step(self, ell, G, Bdiag, delta, return_pred=False):
        """Steihaug-CG/Cauchy step; avoid repeat exact Hessian products."""
        if delta <= 0:
            s = np.zeros_like(G)
            return (s, 0.0) if return_pred else s
        if not isinstance(Bdiag, np.ndarray):
            scg = self.tr_cg_step(ell, G, Bdiag, delta)
            # Preserve L-BFGS behavior; safeguard exact Newton-CG.
            if not hasattr(Bdiag, 'preconditioner_diagonal'):
                pred_cg = self.pred(ell, G, scg, Bdiag) if return_pred else None
                return (scg, pred_cg) if return_pred else scg
            scauchy, pred_cauchy = self.cauchy_step(
                ell, G, Bdiag, delta, return_pred=True)
            pred_cg = self.pred(ell, G, scg, Bdiag)
            if pred_cauchy > pred_cg:
                return (scauchy, pred_cauchy) if return_pred else scauchy
            return (scg, pred_cg) if return_pred else scg
        s = -G/Bdiag
        if self.norm(ell,s) > delta:
            lo, hi = 0., 1.
            while self.norm(ell,G/(Bdiag+hi)) > delta:
                hi *= 2.
            for _ in range(55):
                mid = .5*(lo+hi)
                if self.norm(ell,G/(Bdiag+mid)) > delta:
                    lo = mid
                else:
                    hi = mid
            s = -G/(Bdiag+hi)
        return (s, self.pred(ell,G,s,Bdiag)) if return_pred else s

    def pred(self, ell, G, s, Bdiag):
        mesh = self.levels[ell].mesh
        Bs = self.hessian_matvec(Bdiag, s)
        return -self.inner(ell,G,s)-.5*self.inner(ell,s,Bs)

    def fcd_ok(self, ell, G, s, delta, Bdiag, predicted=None):
        c = self.cfg
        gnorm = self.norm(ell, G)
        rhs = c.kappa_fcd*gnorm*min(gnorm/(1+self.hessian_bound(Bdiag)),delta)
        if predicted is None:
            predicted = self.pred(ell, G, s, Bdiag)
        return predicted+1e-12>=rhs

    def irmtr(self, ell, z_initial, mu, tau, inherited_delta=np.inf, B_initial=None, ev_initial=None):
        """Recursive uncorrected IRMTR: returns a CONTROL, not a step.

        The parent computes s_coarse = result.z - z_initial_coarse.
        The predicted decrease of a parent coarse step uses the INITIAL
        lower-level quadratic model (not the fine model).
        """
        c = self.cfg
        z0 = np.asarray(z_initial, dtype=float).copy()
        z = z0.copy()
        delta = min(c.delta_local, inherited_delta)
        limit = c.max_inner if ell == len(self.levels)-1 else c.max_coarse_inner
        ev_cached = ev_initial
        accepted_trial_state = None
        B_cached = None
        coarse_ev_cached = None
        last_coarse_state = None
        last_coarse_adjoint = None
        last_fine_adjoint = None
        pending_pair = None  # (accepted step, smooth gradient at old control)
        single_step_model_prediction = None
        for k in range(limit):
            if ev_cached is None:
                ev_cached = self.evaluate(ell,z,mu,tau,
                                          state=accepted_trial_state,
                                          p_initial=last_fine_adjoint,
                                          delta=delta)
                accepted_trial_state = None
            ev = ev_cached
            # The inherited/coarse cached evaluation or a rejected-step cached
            # evaluation may have been computed at a larger trust radius.
            if (self.norm(ell, ev.G)+ev.e > tau and
                ev.e > c.kappa_g*min(delta, self.norm(ell, ev.G))*(1+1e-12)):
                ev = self.evaluate(ell, z, mu, tau, state=(ev.u, ev.J),
                                   p_initial=ev.p, delta=delta)
                ev_cached = ev
                B_cached = None
                coarse_ev_cached = None
            # The new fine gradient is needed anyway for stopping. Use it now
            # to record the previous accepted step's secant pair at zero extra solves.
            if pending_pair is not None:
                old_step, old_g = pending_pair
                self.add_lbfgs_pair(ell, old_step, ev.g - old_g)
                pending_pair = None
            #ev = self.evaluate(ell, z, mu, tau)
            #if k==0 and ev_initial is not None:
            #    ev = ev_initial
            #else:
            #    ev = self.evaluate(ell, z, mu, tau)
            stationarity = self.norm(ell, ev.G) + ev.e
            if (c.verbose and c.progress_every > 0
                and ell == len(self.levels)-1
                and (k == 0 or (k + 1) % c.progress_every == 0)):
                print(f'    mu={mu:.3e} inner k={k:4d} '
                      f'F_mu={ev.F:.9e} ||G_mu||+e={stationarity:.3e} '
                      f'delta={delta:.3e}')
            if stationarity <= tau:
                if c.verbose and ell == len(self.levels)-1:
                    print(f'  Inner converged: ||G||+e={stationarity:.6e} '
                          f'<= tau={tau:.6e}, iterations={k}')
                return InnerResult(z, k, stationarity, True, 'stationarity')
            if np.isfinite(inherited_delta):
                remaining = inherited_delta - self.norm(ell, z-z0)
                if remaining <= c.tau_delta*inherited_delta:
                    return InnerResult(z, k, stationarity, False, 'parent radius limit')
                delta = min(delta, remaining)
            if delta <= 1e-14:
                return InnerResult(z, k, stationarity, False, 'radius underflow')
            # After radius changes (especially inherited coarse radii),
            # enforce the dynamic error certificate at the ACTUAL radius.
            if ev.e > c.kappa_g*min(delta,self.norm(ell,ev.G))*(1+1e-12):
                ev = self.evaluate(ell,z,mu,tau,state=(ev.u,ev.J),
                                   p_initial=ev.p,delta=delta)
                ev_cached = ev
                B_cached = None
                coarse_ev_cached = None
                stationarity = self.norm(ell,ev.G)+ev.e
                if stationarity <= tau:
                    return InnerResult(z,k,stationarity,True,'stationarity')

            if k==0 and B_initial is not None:
                Bdiag = (B_initial.copy() if isinstance(B_initial, np.ndarray)
                         else B_initial)
            elif B_cached is not None:
                Bdiag = B_cached
            else:
                Bdiag = self.make_model(ell, z, mu, ev=ev)

            

            #step = self.model_step(ell, ev.G, Bdiag, delta)
            #predicted = self.pred(ell, ev.G, step, Bdiag)
            step = None
            predicted = None
            mode = 'current'
            zeta = np.nan
            eligible = False
            reject_reason = ''
            force_fine = ((c.fine_every > 0 and ell == len(self.levels)-1
                           and (k+1) % c.fine_every == 0) or
                          (c.smooth_only and ell == len(self.levels)-1
                           and (k+1) % 5 == 0))

            if ell > 0 and (not c.smooth_only or ell==len(self.levels)-1) and not force_fine: #and mu>=0.05:
                
                R, P = self.levels[ell].R, self.levels[ell].P
                zc = R @ z
                if coarse_ev_cached is None:
                    coarse_ev_cached = self.evaluate(
                        ell-1, zc, mu, tau,
                        u_initial=last_coarse_state,
                        p_initial=last_coarse_adjoint, delta=delta)
                cev = coarse_ev_cached
                zeta = self.norm(ell-1, cev.G - R @ ev.G)
                if c.smooth_only and ell == len(self.levels)-1:
                    Gf = ev.G
                    Gc = cev.G
                    RGf = R @ Gf
                    consistency_error = self.norm(ell-1,Gc-RGf)
                    eta_cons = consistency_error/max(self.norm(ell-1,RGf), 1e-14)
                    G_projected = P@RGf
                    G_fine_only = Gf -G_projected
                    eta_fine = self.norm(ell,G_fine_only)/max(self.norm(ell,Gf),1e-14)
                    if k<3 or k%100 ==0:
                        print(
                            f"k={k:4d},"
                            f"eta_cons={eta_cons:.6e},"
                            f"eta_fine={eta_fine:.6e},"
                            f"zeta={zeta:.6e},"
                            f"||Gf||={self.norm(ell,Gf):.6e}")
                #coarsening_error = self.norm(ell,z-P@zc)
                
                #if ell == len(self.levels)-1 and mu <= 0.025:
                #    if k < 3 or k % 50 == 0:
                #        print(f"mu={mu:.5f},k={k}," 
                #              f"||z-PRz|| = {coarsening_error:.6e},"
                #              f"bound = {coarsening_error/mu:.6e}")

                #        d_smooth = cev.g - R @ ev.g

                #        d_moreau = (moreau_grad(zc, mu, c.beta)- R @ moreau_grad(z, mu, c.beta))

                #        print(f"mu={mu:.5f}, k={k}, "f"zeta={zeta:.3e}, "f"smooth_gap={self.norm(ell-1, d_smooth):.3e}, "f"moreau_gap={self.norm(ell-1, d_moreau):.3e}")
                gf = self.norm(ell,ev.G)
                gc = self.norm(ell-1,cev.G)
                ml_threshold = c.kappa_ml*min(delta,gf)
                et_threshold = c.kappa_et*gf
                ml_ok = zeta <= ml_threshold
                et_ok = gc >= et_threshold
                eligible = ml_ok and et_ok

                #eligible = (zeta <= c.kappa_ml * min(delta, self.norm(ell, ev.G))
                #            and self.norm(ell-1, cev.G) >= c.kappa_et*self.norm(ell, ev.G))
                #if ell== len(self.levels)-1 and mu<= 0.025:
                #    if k < 3 or k % 50 == 0:
                #        print(
                #            f"mu={mu:.5f}, k={k}, "
                #            f"zeta={zeta:.3e}, "
                #            f"ML threshold={ml_threshold:.3e}, "
                #            f"ML condition={ml_ok}, "
                #            f"coarse gradient={gc:.3e}, "
                #            f"ET threshold={et_threshold:.3e}, "
                #            f"ET condition={et_ok}"
                #            )
                if eligible:
                    self.counters['eligible'] += 1
                    # For nested uniform P0 elements, ||P||_{L2->L2}=1.
                    inherited = delta
                    self.counters['attempted'] += 1
                    Bc = (R @ Bdiag if isinstance(Bdiag, np.ndarray)
                          else GalerkinHessian(Bdiag, R, P))
                    low = self.irmtr(ell-1, zc, mu, tau, inherited_delta=inherited, B_initial=Bc, ev_initial = cev)
                    sc = low.z - zc
                    sf = P @ sc
                    
                    #Bc = self.model_diagonal(zc, mu)
                    #Bc = R @ Bdiag
                    # For a single-step child, its own model prediction is
                    # already known. That model is exactly Bc = R Bf P at zc.
                    # Reusing it avoids another fine-grid Hessian action.
                    if low.local_predicted is not None:
                        coarse_pred = low.local_predicted
                    elif not np.any(sc):
                        coarse_pred = 0.0
                    else:
                        coarse_pred = self.pred(ell-1, cev.G, sc, Bc)
                    # Exact Galerkin identity: Bc = R Bf P, R = P^* in L2.
                    # Same fine quadratic term, so no second fine H*sf action:
                    fine_pred = coarse_pred + self.inner(
                        ell-1, cev.G - R @ ev.G, sc)
                    #current_step = self.model_step(ell,ev.G,Bdiag,delta)
                    #current_pred = self.pred(ell,ev.G,current_step,Bdiag)
                    #step_efficiency = (fine_pred/current_pred if current_pred>1e-14 else np.nan)
                    
                    #if coarse_pred > 1e-14:
                    #    model_ratio = fine_pred/coarse_pred
                    #    discrepancy_ratio = (zeta*self.norm(ell-1,sc)/coarse_pred)
                    #else:
                    #    model_ratio = np.nan
                    #    discrepancy_ratio=np.nan
                        
                        
                    
                        
                        
                    # Print model-quality diagnostics at the finest level
                    #if ell == len(self.levels) - 1:
                    #    if k < 3 or k % 20 == 0:
                            
                    #        print(f"mu={mu:.5f}, k={k},"
                    #                  f"coarse_model_ratio = {model_ratio:.4f},"
                    #                  f"step_efficiency={step_efficiency:.4f}")
                    #        print(
                    #            f"mu={mu:.5f}, k={k}, "
                    #            f"D_coarse={coarse_pred:.6e}, "
                    #            f"D_fine={fine_pred:.6e}, "
                    #            f"model_ratio={model_ratio:.4f}, "
                    #            f"discrepancy_ratio={discrepancy_ratio:.4f}"
                    #        )
                    #discrepancy = abs(fine_pred - coarse_pred)
                    #relative_discrepancy = discrepancy/max(abs(coarse_pred),1e-14)
                    
                   # print(f"level={ell}," f"zeta={zeta:.3e}, "f"step_norm={self.norm(ell-1, sc):.3e}," 
                   #       f"coarse_pred={coarse_pred:.3e},"
                   #       f"fine_pred={fine_pred:.3e},"
                   #       f"relative model discrepancy = {relative_discrepancy:.3e}")
                    feasible = self.norm(ell, sf) <= delta*(1+1e-10)
                    fcd = self.fcd_ok(ell-1, cev.G, sc, inherited, Bc,
                                      predicted=coarse_pred)
                    model_quality_ok=(coarse_pred>1e-14 and fine_pred >= 0.5*coarse_pred)
                    #efficiency_ok = (current_pred > 1e-14 and fine_pred >= 0.9 * current_pred)
                    if feasible and model_quality_ok and fcd :#and efficiency_ok:
                        step, predicted, mode = sf, coarse_pred, 'coarse'
                        self.counters['used'] += 1
                    else:
                        self.counters['fallback'] += 1
                        reject_reason = ('infeasible' if not feasible else
                                         'nonpositive coarse prediction' if coarse_pred <= 1e-14 else
                                         'coarse FCD failed' if not fcd else
                                         'fine-model quality failed')
            if mode == 'current':
                step, predicted = self.model_step(
                    ell, ev.G, Bdiag, delta, return_pred=True)
            if mode == 'current' and not self.fcd_ok(
                    ell, ev.G, step, delta, Bdiag, predicted=predicted):
                return InnerResult(z, k, stationarity, False, 'FCD failed')
            if predicted <= 0 or not np.isfinite(predicted):
                return InnerResult(z, k, stationarity, False, 'nonpositive predicted reduction')
            step_ratio = self.norm(ell,step)/max(delta,1e-14)
            if (c.smooth_only and c.hessian_mode == 'diagonal'
                and ell==len(self.levels)-1 and mode == "coarse"
                and (k<3 or k%200==0)):
                fine_step=self.model_step(ell,ev.G,Bdiag,delta)
                D_fine_opt = self.pred(ell,ev.G,fine_step, Bdiag)
                D_coarse_actual = self.pred(ell,ev.G, step, Bdiag)
                eta_step = (D_coarse_actual/D_fine_opt if D_fine_opt>1e-14 else np.nan)
                print(f"\nk={k}, actual eta_step={eta_step:.4f}")
                for j in range(ell-1,-1,-1):
                    gproj = ev.G.copy()
                    for lev in range(ell,j,-1):
                        gproj = self.levels[lev].R @ gproj
                    for lev in range(j+1,ell+1):
                        gproj = self.levels[lev].P @ gproj
                    s_best = -gproj/c.curvature
                    step_norm = self.norm(ell, s_best)
                    if step_norm > delta:
                        s_best *= delta/step_norm
                    D_best = self.pred(ell,ev.G,s_best,Bdiag)
                    efficiency_bound = D_best / D_fine_opt
                    fine_only = self.norm(ell,ev.G-gproj)/max(self.norm(ell,ev.G),1e-14)
                    print(
                        f" level {j}:"
                        f"best_efficiency={efficiency_bound:.4f},"
                        f"fine_only={fine_only:.4f}"
                        )
                print(
                    f"k={k:4d},"
                    f"eta_step={eta_step:.6f},"
                    f"D_fine_opt={D_fine_opt:.6e},"
                    f"D_coarse_actual={D_coarse_actual:.6e}"
                    )
            trial = z + step
            F_trial, trial_state = self.objective(
                ell, trial, mu, return_state=True, u_initial=ev.u)
            cred = ev.F - F_trial
            rho = cred / predicted
            accepted = bool(rho >= c.eta1)
            if limit == 1 and ell < len(self.levels)-1:
                # Parent's coarse_pred is the quadratic reduction in THIS
                # level's original model (not the deeper model's prediction).
                single_step_model_prediction = (
                    (fine_pred if mode == 'coarse' else predicted)
                    if accepted else 0.0)
            #delta_before = delta
            
            if accepted:
                # Defer the gradient difference to the next loop iteration.
                # Rejected steps do NOT contribute L-BFGS secant pairs.
                if c.hessian_mode == 'lbfgs':
                    pending_pair = (step.copy(), ev.g.copy())
                last_fine_adjoint = ev.p
                z = trial
                if ell == len(self.levels)-1 and mode == 'coarse':
                    self.counters['coarse_accepted_finest'] += 1
                ev_cached = None
                accepted_trial_state = trial_state
                B_cached = None
                if coarse_ev_cached is not None:
                    last_coarse_state = coarse_ev_cached.u
                    last_coarse_adjoint = coarse_ev_cached.p
                coarse_ev_cached = None
            else:
                if not (k==0 and B_initial is not None):
                    B_cached = Bdiag
                
            #if accepted and ell==len(self.levels)-1:
            #    ev_after = self.evaluate(ell,z,mu,tau)
            #    stationarity_after = self.norm(ell,ev_after.G)+ev_after.e
            #    self.history_stationarity = getattr(self, 'history_stationarity',[])
            #    self.history_stationarity.append({'mu':mu,'mode':mode,'before':stationarity,'after':stationarity_after,'change':stationarity_after-stationarity})
            if rho >= c.eta2:
                delta = min(c.gamma_expand*delta, c.delta_max)
            elif rho < c.eta1:
                delta = c.gamma_shrink*delta
            #if np.isfinite(inherited_delta):
            #    delta = min(delta, max(0., inherited_delta-self.norm(ell, z-z0)))
            #print(f"level={ell}, k={k}, "f"rho={rho:.6e}, "f"delta_before={delta_before:.6e}, "f"delta_after={delta:.6e}")
            self.history.append(dict(level=ell,k=k,mode=mode,mu=mu,F=ev.F,
                                     residual=stationarity,delta=delta,rho=rho,
                                     accepted=accepted,cred=cred,predicted=predicted,
                                     gradient_error=ev.e, certified_residual=stationarity,
                                     zeta=zeta,eligible=eligible,
                                     rejection=reject_reason,step_ratio = step_ratio))
            #if c.verbose  and (k < 3 or k % 50 == 0): #and ell == len(self.levels)-1
            #    indent = " "*(2-ell)
            #    print(
            #        f"{indent}Level {ell}, "
            #        f"k={k:04d}, "
            #        f"mode={mode}, "
            #        f"F={ev.F:.6e}, "
            #        f"rho={rho:.3e}, "
            #        f"delta={delta:.3e}, "
            #        f"zeta={zeta:.3e}"
            #    )
                #print(f'  inner k={k:04d} level={ell} {mode:7s} '
                #      f'F={ev.F:.7e} stat<={stationarity:.3e} '
                #      f'rho={rho:.3e} delta={delta:.2e} '
                #      f'zeta={zeta:.2e}')
        #ev = ev_cached if ev_cached is not None else self.evaluate(ell, z, mu, tau)
        
        if ell<len(self.levels)-1:
            return InnerResult(z, limit, np.nan, False,
                               'coarse iteration limit',
                               local_predicted=single_step_model_prediction)
        ev = (ev_cached if ev_cached is not None else self.evaluate(ell,z,mu,tau))
        final_stat = self.norm(ell, ev.G) + ev.e
        if c.verbose:# and ell == len(self.levels)-1:
            print(f'  Inner reached max_inner: stationarity={final_stat:.6e}, tau={tau:.6e}')
        return InnerResult(z, limit, final_stat, False, 'max_inner reached')

    def original_stationarity(self,z,r,tau=None,eps_hint=None):
        """Return (certified upper bound, approximate prox residual, error)."""
        if self.backend is not None:
            hint = eps_hint if eps_hint is not None else (
                getattr(self, '_outer_eps', 1e-3) if tau is None else tau)
            return self.backend.original_stationarity(self, z, r, hint)
        if tau is None:
            tau = eps_hint if eps_hint is not None else 1e-3
        ell=len(self.levels)-1
        # For outer test, just evaluate gradient f; mu choice does not change g.
        ev=self.evaluate(ell,z,1.,tau)
        step=prox_phi(z-r*ev.g,r,self.cfg.beta)
        psi=self.norm(ell,z-step)/r
        return psi+ev.e,psi,ev.e

    def solve_outer(self, z0, mu0, tau_fixed, eps,
                    theta_mu=.5, r=1., max_outer=None,
                    report_z=True, print_full_z=False, theta_tau=None):
        """Shared continuation loop; feasible stopping for both P0 and P1.

        The `tau_fixed` parameter is historically named; tau decreases each
        stage.  `theta_tau` can optionally differ from `theta_mu`.
        """
        if not (mu0 > 0 and eps > 0 and 0 < tau_fixed and r > 0
                and 0 < theta_mu < 1):
            raise ValueError('Require positive mu/tau/eps/r and 0<theta_mu<1')
        if theta_tau is None:
            theta_tau = theta_mu
        if not 0 < theta_tau < 1:
            raise ValueError('Require 0 < theta_tau < 1')
        self._outer_eps = eps
        c = self.cfg
        h = len(self.levels)-1
        z = np.asarray(z0, dtype=float).copy()
        mu, tau = mu0, tau_fixed
        history=[]
        nouter = c.max_outer if max_outer is None else max_outer
        for j in range(nouter):
            if c.verbose:
                print(f'outer j={j}, mu={mu:.3e}, tau={tau:.3e}')
            z_start = z.copy()
            if c.hessian_mode == 'lbfgs':
                self.reset_lbfgs()
            start = time.perf_counter()
            low = self.irmtr(h, z, mu, tau, inherited_delta=np.inf)
            elapsed = time.perf_counter()-start
            z = low.z
            z_feas = np.clip(z,-25.,25.)
            feas_upper, feas_psi, feas_e = self.original_stationarity(z_feas,r,tau)
            upper, psi, e = self.original_stationarity(z,r,tau)
            history.append(dict(
                j=j, mu=float(mu), tau=float(tau),
                F_mu=self.objective(h,z,mu),
                inner_converged=low.converged,
                inner_reason=low.reason,
                inner_iterations=low.iterations,
                inner_final_residual=low.residual_bound,
                psi=psi, e=e, stationarity_upper=upper,
                stationarity_hat=psi, stationarity_error=e,
                feasible_psi=feas_psi, feasible_e=feas_e,
                feasible_error=feas_e,
                feasible_stationarity_upper=feas_upper,
                z_start=z_start.copy(), z_end=z.copy(),
                elapsed=elapsed))
            if report_z:
                changed = np.count_nonzero(z_feas!=z)
                print(f'  z at outer j={j}, mu={mu:.6e}: '
                      f'min={z_feas.min():.8e}, max={z_feas.max():.8e}, '
                      f'L2={self.norm(h,z_feas):.8e}, '
                      f'clipped elements={changed}/{z.size}')
            if c.verbose:
                print(f'  stage {j}: iterations={low.iterations} '
                      f'raw psi+e={upper:.6e} feasible psi+e={feas_upper:.6e} '
                      f'elapsed={elapsed:.2f}s')
            if not low.converged:
                return OuterResult(z_feas,history,False)
            if feas_upper <= eps:
                return OuterResult(z_feas,history,True)
            mu_new = mu*theta_mu
            if not c.smooth_only:
                v = prox_phi(z,mu,c.beta)
                z=v+(mu_new/mu)*(z-v)
            mu=mu_new
            tau *= theta_tau
        return OuterResult(np.clip(z,-25.,25.),history,False)
