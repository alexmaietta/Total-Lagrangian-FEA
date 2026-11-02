#!/usr/bin/env python3
"""FEniCSx reference for the T10 beam benchmark (revision accuracy check).

Runs test-scripts/validation/fenics/beam_dynamic.py unchanged, then
  * copies its tracked-node position history into --out, and
  * writes, for every cell, the von Mises value of the Cauchy stress averaged
    over the five Keast points used by the GPU code (same component formula),
    together with the reference-configuration cell centroid.

The Keast point set is invariant under vertex permutations, so the per-cell
average is independent of the cell vertex ordering used by DOLFINx.

Usage (repository root):
  mpirun -np 8 python test-scripts/revision/fenics_beam_vm.py --res 8 --out DIR [--mr]
"""
import argparse
import os
import runpy
import shutil
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, required=True)
ap.add_argument("--mr", action="store_true")
ap.add_argument("--out", required=True)
ap.add_argument("--no-vm", action="store_true")
ap.add_argument("--keast-mass", action="store_true")
ap.add_argument("--snes-tol", type=float, default=None,
                help="override SNES atol/rtol of beam_dynamic.py")
args = ap.parse_args()
if args.snes_tol is not None:
    os.environ["FENICS_SNES_TOL"] = repr(args.snes_tol)

here = os.path.dirname(os.path.abspath(__file__))
script = os.path.normpath(os.path.join(here, "..", "validation", "fenics", "beam_dynamic.py"))
sys.argv = [script, "--res", str(args.res)] + (["--mr"] if args.mr else []) \
    + (["--keast-mass"] if args.keast_mass else [])
sys.path.insert(0, os.path.dirname(script))
g = runpy.run_path(script, run_name="__main__")

import ufl  # noqa: E402
from dolfinx import fem  # noqa: E402

domain = g["domain"]
comm = domain.comm
rank = comm.rank
os.makedirs(args.out, exist_ok=True)

mat = "mr" if args.mr else "svk"
src = os.path.join(g["root_output_dir"], f"node_xyz_history_fenics_res{args.res}_{mat}.csv")
if rank == 0 and os.path.exists(src):
    shutil.copy(src, os.path.join(args.out, "x_fenics.csv"))

if not args.no_vm:
    F, P = g["F"], g["P"]
    J = ufl.det(F)
    sigma = (1.0 / J) * P * F.T
    s00, s11, s22 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s01, s02, s12 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    vm2 = 0.5 * ((s00 - s11) ** 2 + (s11 - s22) ** 2 + (s22 - s00) ** 2) \
        + 3.0 * (s01 ** 2 + s02 ** 2 + s12 ** 2)
    vm = ufl.sqrt(abs(vm2))
    keast = np.array([[0.25, 0.25, 0.25],
                      [1 / 6, 1 / 6, 1 / 6],
                      [0.5, 1 / 6, 1 / 6],
                      [1 / 6, 0.5, 1 / 6],
                      [1 / 6, 1 / 6, 0.5]])
    expr = fem.Expression(vm, keast)
    tdim = domain.topology.dim
    n_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(n_local, dtype=np.int32)
    vals = np.asarray(expr.eval(domain, cells)).reshape(n_local, -1)
    vm_cell = vals.mean(axis=1)
    dofmap = domain.geometry.dofmap
    xg = domain.geometry.x
    centroids = np.array([xg[dofmap[c, :4], :3].mean(axis=0) for c in range(n_local)])
    all_c = comm.gather(centroids, root=0)
    all_v = comm.gather(vm_cell, root=0)
    if rank == 0:
        c = np.vstack(all_c)
        v = np.concatenate(all_v)
        with open(os.path.join(args.out, "vm_fenics.csv"), "w") as f:
            f.write("cx,cy,cz,von_mises\n")
            for (x, y, z), val in zip(c, v):
                f.write(f"{x:.17g},{y:.17g},{z:.17g},{val:.17g}\n")
        print(f"Wrote {len(v)} cell von Mises values to {args.out}/vm_fenics.csv")
