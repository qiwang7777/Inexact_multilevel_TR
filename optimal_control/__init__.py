"""Optimal control via Moreau continuation and multilevel IRMTR."""

from .geometry import create_unit_square_mesh, element_geometry, p0_norm, p0_inner
from .types import Config, OuterResult
from .transfer_operators import build_transfer_operators, p0_transfers, p1_transfers
from .irmtr import IRMTRMoreau
from .pde import semilinear
from .pde.elliptic_backend import EllipticBackend

__all__ = ["create_unit_square_mesh", "element_geometry", "p0_norm",
           "p0_inner", "Config", "OuterResult", "IRMTRMoreau", "semilinear",
           "build_transfer_operators", "p0_transfers", "p1_transfers", "EllipticBackend"]
