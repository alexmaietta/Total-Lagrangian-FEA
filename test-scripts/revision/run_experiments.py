#!/usr/bin/env python3
"""Experiments for the Part II revision (Engineering with Computers).

Each batch writes one folder per run under --out:
  run.log, *.csv outputs, and (with --guard) guard.jsonl / summary.json.

Batches
  scaling    GPU RTF tables: T10 (SVK, MR), ANCF3243, ANCF3443; Newton and AdamW
  breakdown  per-phase Newton cost (TLFEA_PHASE_TIMING=1) and fixed-sparsity ablation
  vase       collision vs. solve time in the vase-on-foam test
  reference  tight-tolerance Newton reference for T10, plus a von Mises dump
  tolerance  Newton at 1e-4 where the paper relaxed to 1e-3
  impact     time step x contact modulus sweep for the oblique impact test
  fenics     FEniCS T10 references (CPU, MPI; needs a DOLFINx 0.10 env, no GPU build)

Run from the repository root after building with --config=sm120, e.g.
  python3 test-scripts/revision/run_experiments.py scaling --out /path/to/runs \
      --guard /path/to/guard.py
  python3 test-scripts/revision/run_experiments.py fenics --out /path/to/runs \
      --fenics-python "$(which python)" --np 8
"""
import argparse
import math
import os
import shlex
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN = os.path.join(REPO, "bazel-bin", "lib_bin")
T10 = os.path.join(BIN, "beam_sag", "test_feat10_resolution")
A3243 = os.path.join(BIN, "beam_sag", "test_ancf3243")
A3443 = os.path.join(BIN, "beam_sag", "test_ancf3443")
IMPACT = os.path.join(BIN, "ptest", "oblique_impact")
VASE = os.path.join(BIN, "large_scale", "test_vase")
RES = [0, 2, 4, 8, 16, 32]


def run(args, name, cmd, env_extra=None, timing=False, collect=None):
    run_dir = os.path.join(args.out, args.batch, name)
    if os.path.exists(os.path.join(run_dir, "DONE")) and not args.force:
        print(f"skip {name} (done)", flush=True)
        return
    os.makedirs(run_dir, exist_ok=True)
    cmd = [c.replace("{RUN}", run_dir) for c in cmd]
    env = dict(os.environ)
    env.update(env_extra or {})
    if args.guard:
        # The data cap applies to everything under --out, not just this batch.
        full = [sys.executable, args.guard, "--name", f"{args.batch}/{name}",
                "--data-dir", args.out]
        full += shlex.split(args.guard_args)
        if timing:
            full.append("--timing")
        full += ["--"] + cmd
        rc = subprocess.call(full, cwd=REPO, env=env, stdout=subprocess.DEVNULL)
    else:
        with open(os.path.join(run_dir, "run.log"), "w") as log:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    with open(os.path.join(run_dir, "cmd.txt"), "w") as f:
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
        f.write(" ".join(f"{k}={v}" for k, v in (env_extra or {}).items()) + "\n")
    for src in collect or []:
        src_path = os.path.join(REPO, src)
        if os.path.isdir(src_path):
            for fn in os.listdir(src_path):
                os.replace(os.path.join(src_path, fn), os.path.join(run_dir, fn))
            os.rmdir(src_path)
    if rc == 0:
        open(os.path.join(run_dir, "DONE"), "w").close()
    print(f"{name}: rc={rc}", flush=True)


def batch_scaling(args):
    for res in RES:
        for mat in ["svk", "mr"]:
            matflag = "--mr_mat" if mat == "mr" else "--svk_mat"
            for solver in ["newton", "adamw"]:
                run(args, f"t10_{mat}_{solver}_res{res}",
                    [T10, f"--solver={solver}", f"--res={res}", "--steps=50",
                     "--dt=1e-3", matflag, "--csv={RUN}/x.csv"], timing=True)
    for res in RES:
        for solver in ["newton", "adamw"]:
            run(args, f"a3243_{solver}_res{res}",
                [A3243, f"--solver={solver}", f"--res={res}", "--steps=200",
                 "--dt=1e-3", "--csv={RUN}/tip.csv"], timing=True)
            run(args, f"a3443_{solver}_res{res}",
                [A3443, f"--solver={solver}", f"--res={res}",
                 "--dt=5e-4", "--csv={RUN}/tip.csv"], timing=True)


def batch_breakdown(args):
    t = {"TLFEA_PHASE_TIMING": "1"}
    for res in [16, 32]:
        base = [T10, "--solver=newton", f"--res={res}", "--steps=50", "--dt=1e-3",
                "--svk_mat"]
        run(args, f"t10_res{res}_refactor", base, {**t, "TLFEA_FIXED_SPARSITY": "1"}, True)
        run(args, f"t10_res{res}_factor", base, {**t, "TLFEA_FIXED_SPARSITY": "0"}, True)
        run(args, f"t10_res{res}_reanalysis", base,
            {**t, "TLFEA_FIXED_SPARSITY": "0", "TLFEA_FORCE_REANALYSIS": "1"}, True)
    for res in [8, 16]:
        base = [A3443, "--solver=newton", f"--res={res}", "--dt=5e-4"]
        run(args, f"a3443_res{res}_factor", base, {**t, "TLFEA_FIXED_SPARSITY": "0"}, True)
        run(args, f"a3443_res{res}_refactor", base, {**t, "TLFEA_FIXED_SPARSITY": "1"}, True)
        base = [A3243, "--solver=newton", f"--res={res}", "--steps=200", "--dt=1e-3"]
        run(args, f"a3243_res{res}_factor", base, {**t, "TLFEA_FIXED_SPARSITY": "0"}, True)
        run(args, f"a3243_res{res}_refactor", base, {**t, "TLFEA_FIXED_SPARSITY": "1"}, True)


def batch_vase(args):
    # positional: mu_s mu_k self_collision steps export_interval
    run(args, "vase_eva80", [VASE, "--mat=eva80", "0.6", "0.5", "0", str(args.vase_steps), "0"],
        {"TLFEA_PHASE_TIMING": "1"}, True)


def batch_reference(args):
    # The tolerances can sit below the round-off floor of ||g||; the caps bound the
    # cost (the inner loop is at round-off after a few quadratic steps).
    tight = ["--newton_atol=1e-11", "--newton_rtol=1e-12", "--outer_tol=1e-11",
             "--max_inner=8", "--max_outer=3"]
    for res in RES:
        for mat in ["svk", "mr"]:
            matflag = "--mr_mat" if mat == "mr" else "--svk_mat"
            extra = ["--vm_out={RUN}/vm.csv"] if (res == 8 and mat == "svk") else []
            run(args, f"t10_{mat}_ref_res{res}",
                [T10, "--solver=newton", f"--res={res}", "--steps=50", "--dt=1e-3",
                 matflag, "--csv={RUN}/x.csv"] + tight + extra)


def batch_tolerance(args):
    # Paper relaxed 1e-4 -> 1e-3 at: T10 RES32, ANCF3243 RES32, ANCF3443 RES8-32.
    run(args, "t10_svk_tol1e-4_res32",
        [T10, "--solver=newton", "--res=32", "--steps=50", "--dt=1e-3", "--svk_mat",
         "--csv={RUN}/x.csv", "--newton_atol=1e-4", "--newton_rtol=1e-4",
         "--outer_tol=1e-4"])
    run(args, "a3243_tol1e-4_res32",
        [A3243, "--solver=newton", "--res=32", "--steps=200", "--dt=1e-3",
         "--csv={RUN}/tip.csv", "--newton_tol=1e-4"])
    for res in [8, 16, 32]:
        run(args, f"a3443_tol1e-4_res{res}",
            [A3443, "--solver=newton", f"--res={res}", "--dt=5e-4",
             "--csv={RUN}/tip.csv", "--newton_tol=1e-4"])


def batch_impact(args):
    theta = math.radians(args.theta_deg)
    sim_time = 0.2
    for dt in args.dts:
        for e_c in args.contact_moduli:
            steps = int(round(sim_time / dt))
            name = f"impact_th{args.theta_deg:g}_dt{dt:g}_E{e_c:g}"
            run(args, name,
                [IMPACT, f"--theta={theta}", "--v_i=1.0", f"--dt={dt}",
                 f"--contact_E={e_c}", f"--steps={steps}", "--export_interval=0",
                 f"--out_suffix={name}", "--rigid_equiv_csv=rigid.csv",
                 "--rigid_equiv_interval=1"],
                {"TLFEA_PHASE_TIMING": "1"},
                collect=[os.path.join("output", f"oblique_impact_{name}")])


def batch_fenics(args):
    py = args.fenics_python
    mpirun = os.path.join(os.path.dirname(py), "mpirun")
    script = os.path.join(REPO, "test-scripts", "revision", "fenics_beam_vm.py")
    # "keast": inertia term integrated with the GPU's 5-point rule (same discrete
    # problem as the GPU code); default: degree-5 rule for all terms.
    for variant in ["keast", "default"]:
        for mat in ["svk", "mr"]:
            for res in RES:
                if variant == "keast" and res > 16:
                    continue
                extra = [] if (res == 8 and mat == "svk") else ["--no-vm"]
                if variant == "keast":
                    extra.append("--keast-mass")
                tag = "_keast" if variant == "keast" else ""
                run(args, f"t10_{mat}{tag}_res{res}",
                    [mpirun, "-np", str(args.np), py, script, "--res", str(res),
                     "--out", "{RUN}", "--snes-tol", "1e-8"]
                    + (["--mr"] if mat == "mr" else []) + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", choices=["scaling", "breakdown", "vase", "reference",
                                      "tolerance", "impact", "fenics"])
    ap.add_argument("--fenics-python", default=None, help="python of a DOLFINx 0.10 env")
    ap.add_argument("--np", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guard", default=None, help="optional path to a guard wrapper")
    ap.add_argument("--guard-args", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--vase-steps", type=int, default=1400)
    ap.add_argument("--theta-deg", type=float, default=75.0)
    ap.add_argument("--dts", type=float, nargs="+", default=[5e-5, 1e-4, 2e-4])
    ap.add_argument("--contact-moduli", type=float, nargs="+", default=[1e7, 5e7, 2.5e8])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    globals()[f"batch_{args.batch}"](args)


if __name__ == "__main__":
    main()
