"""Tiny tests; run from the project folder using: python -m unittest discover -s tests"""
import unittest
import numpy as np

from optimal_control.geometry import (create_unit_square_mesh, p0_transfers,
                                      p0_inner, p0_norm)
from optimal_control.irmtr import IRMTRMoreau
from optimal_control.types import Config


class SolverSmokeTests(unittest.TestCase):
    def test_transfer_is_l2_adjoint(self):
        mc, mf = create_unit_square_mesh(2), create_unit_square_mesh(4)
        R, P = p0_transfers(mc, mf)
        a = np.linspace(-1., 1., len(mc.elements))
        b = np.linspace(0.5, -0.7, len(mf.elements))
        self.assertAlmostEqual(p0_inner(mc, a, R @ b), p0_inner(mf, P @ a, b), places=12)

    def test_exact_hessian_action(self):
        mesh = create_unit_square_mesh(3)
        solver = IRMTRMoreau([mesh], Config(hessian_mode='exact', verbose=False))
        z = 0.5 * np.ones(len(mesh.elements))
        rng = np.random.default_rng(11)
        s = rng.normal(size=len(z))
        s /= p0_norm(mesh, s)
        mu = 0.4
        ev = solver.evaluate(0, z, mu, tau=8e-5)
        H = solver.make_model(0, z, mu, ev)
        step = 1e-4
        Gplus = solver.evaluate(0, z + step*s, mu, tau=8e-5).G
        Gminus = solver.evaluate(0, z - step*s, mu, tau=8e-5).G
        fd = (Gplus - Gminus)/(2*step)
        self.assertLess(p0_norm(mesh, fd-H.matvec(s)), 1e-4)

    def test_short_outer_run_records_feasible_residual(self):
        meshes = [create_unit_square_mesh(n) for n in (2, 4)]
        solver = IRMTRMoreau(meshes, Config(hessian_mode='exact', max_inner=3,
                                          max_outer=1, verbose=False))
        z0 = 2.0 * np.ones(len(meshes[-1].elements))
        out = solver.solve_outer(z0, 0.4, 8e-5, eps=1e-4,
                                 max_outer=1, report_z=False)
        self.assertEqual(out.z.shape, z0.shape)
        self.assertTrue(np.all(np.abs(out.z) <= 25))
        self.assertIn('feasible_stationarity_upper', out.history[-1])


if __name__ == '__main__':
    unittest.main()
