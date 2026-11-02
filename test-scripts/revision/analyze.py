#!/usr/bin/env python3
"""Summarize the revision runs (see run_experiments.py).

Writes <out>/summary.json and LaTeX table bodies / figures under <out>/tex.
Missing runs are skipped, so the script can be re-run as batches finish.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = [0, 2, 4, 8, 16, 32]
T10_TRACKED = {0: 23, 2: 89, 4: 353, 8: 1409, 16: 5633, 32: 22529}
STEP_RE = re.compile(r"OneStep(?:NewtonCuDSS|AdamWNocoop) kernel time: ([0-9.eE+-]+) ms")
PHASE_RE = re.compile(r"PHASE_TIMING (.*)")
VASE_RE = re.compile(r"VASE_TIMING (.*)")
ADAMW_RE = re.compile(r"Outer iter (\d+): inner loop (?:converged in (\d+) iterations|hit max_inner=(\d+))")
STEP_SPLIT_RE = re.compile(r"OneStepAdamWNocoop kernel time: [0-9.eE+-]+ ms")


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def kv(s):
    out = {}
    for tok in s.split():
        k, v = tok.split("=")
        out[k] = float(v)
    return out


def rtf_from_log(log, steps, dt):
    times = [float(x) for x in STEP_RE.findall(read(log))]
    if len(times) != steps:
        return None, times
    return sum(times) / 1000.0 / (steps * dt), times


def adamw_stats(log):
    chunks = STEP_SPLIT_RE.split(read(log))[:-1]
    outer, inner, capped_loops, loops, capped_steps = [], [], 0, 0, 0
    for ch in chunks:
        recs = ADAMW_RE.findall(ch)
        outer.append(len(recs))
        tot = 0
        step_capped = False
        for _o, conv, cap in recs:
            loops += 1
            if cap:
                capped_loops += 1
                step_capped = True
                tot += int(cap)
            else:
                tot += int(conv)
        inner.append(tot)
        capped_steps += step_capped
    if not chunks:
        return None
    return {"steps": len(chunks), "mean_outer": float(np.mean(outer)),
            "mean_inner_per_step": float(np.mean(inner)),
            "max_inner_per_step": int(np.max(inner)),
            "capped_inner_loops": capped_loops, "inner_loops": loops,
            "steps_with_cap_hit": int(capped_steps)}


def node_x0(res, idx):
    path = os.path.join(REPO, "data", "meshes", "T10", "resolution", f"beam_3x2x1_res{res}.1.node")
    with open(path) as f:
        f.readline()
        for line in f:
            s = line.split()
            if s and not s[0].startswith("#") and int(s[0]) == idx + 1:
                return float(s[1])
    raise KeyError(idx)


def load_series(path, col):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [l for l in f if l.strip() and not l.startswith("#")]
    head = rows[0].strip().split(",")
    j = head.index(col)
    return np.array([float(r.split(",")[j]) for r in rows[1:]])


def rel_err(a, ref):
    n = min(len(a), len(ref))
    if n == 0:
        return None
    a, ref = a[:n], ref[:n]
    scale = np.max(np.abs(ref))
    return float(np.max(np.abs(a - ref)) / scale) if scale > 0 else None


def scaling(runs):
    out = {}
    specs = []
    for mat in ["svk", "mr"]:
        for s in ["newton", "adamw"]:
            for r in RES:
                specs.append((f"t10_{mat}_{s}_res{r}", 50, 1e-3, ("t10", mat, s, r)))
    for s in ["newton", "adamw"]:
        for r in RES:
            specs.append((f"a3243_{s}_res{r}", 200, 1e-3, ("ancf3243", "svk", s, r)))
            specs.append((f"a3443_{s}_res{r}", 200, 5e-4, ("ancf3443", "svk", s, r)))
    for name, steps, dt, key in specs:
        d = os.path.join(runs, "scaling", name)
        log = os.path.join(d, "run.log")
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        rtf, _ = rtf_from_log(log, steps, dt)
        rec = {"rtf": rtf}
        summ = os.path.join(d, "summary.json")
        if os.path.exists(summ):
            c = json.load(open(summ))["contention"]
            rec["max_other_cores"] = c["max_other_cores"]
        if key[2] == "adamw":
            rec["adamw"] = adamw_stats(log)
        out["/".join(map(str, key))] = rec
    return out


def final_err(a, ref):
    n = min(len(a), len(ref))
    if n == 0 or ref[n - 1] == 0:
        return None
    return float(abs(a[n - 1] - ref[n - 1]) / abs(ref[n - 1]))


FENICS_VARIANTS = [("fenics", ""), ("fenics_keast", "_keast")]


def accuracy(runs, scal):
    """Tracked-node x displacement: GPU (production / tight tolerance) vs FEniCS.

    "hist": max over the history of |a - b| / max |b|; "final": relative error at t_end.
    """
    out = {}
    for mat in ["svk", "mr"]:
        for r in RES:
            x0 = node_x0(r, T10_TRACKED[r])
            u = {}
            ref = load_series(os.path.join(runs, "reference", f"t10_{mat}_ref_res{r}", "x.csv"), "x_position")
            if ref is not None:
                u["gpu_ref"] = ref - x0
            for s in ["newton", "adamw"]:
                x = load_series(os.path.join(runs, "scaling", f"t10_{mat}_{s}_res{r}", "x.csv"), "x_position")
                if x is not None:
                    u[f"gpu_{s}"] = x - x0
            for tag, sub in FENICS_VARIANTS:
                fe = load_series(os.path.join(runs, "fenics", f"t10_{mat}{sub}_res{r}", "x_fenics.csv"),
                                 "x_position")
                if fe is not None:
                    u[tag] = fe - x0
            if not u:
                continue
            rec = {"u_max": {k: float(np.max(np.abs(v))) for k, v in u.items()},
                   "u_final": {k: float(v[-1]) for k, v in u.items()}}
            for a in ["gpu_newton", "gpu_adamw", "gpu_ref"]:
                for b in ["gpu_ref", "fenics", "fenics_keast"]:
                    if a != b and a in u and b in u:
                        rec[f"{a}_vs_{b}"] = {"hist": rel_err(u[a], u[b]), "final": final_err(u[a], u[b])}
            out[f"t10/{mat}/res{r}"] = rec
    return out


def vm_compare(runs):
    g = os.path.join(runs, "reference", "t10_svk_ref_res8", "vm.csv")
    if not os.path.exists(g):
        return None
    gd = np.loadtxt(g, delimiter=",", skiprows=1)
    key = lambda c: tuple(np.round(c, 7))
    out = {}
    for tag, sub in FENICS_VARIANTS:
        f = os.path.join(runs, "fenics", f"t10_svk{sub}_res8", "vm_fenics.csv")
        if not os.path.exists(f):
            continue
        fd = np.loadtxt(f, delimiter=",", skiprows=1)
        fmap = {key(row[:3]): row[3] for row in fd}
        a = np.array([(row[4], fmap[key(row[1:4])]) for row in gd if key(row[1:4]) in fmap])
        if len(a) == 0:
            out[tag] = {"matched": 0}
            continue
        diff = a[:, 0] - a[:, 1]
        out[tag] = {"matched": int(len(a)), "n_gpu": int(len(gd)), "n_fenics": int(len(fd)),
                    "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(a[:, 1])),
                    "max_vm_gpu": float(a[:, 0].max()), "max_vm_fenics": float(a[:, 1].max()),
                    "rel_err_max_vm": float(abs(a[:, 0].max() - a[:, 1].max()) / a[:, 1].max())}
    return out


def tolerance(runs):
    out = {}
    pairs = [
        ("t10_svk_res32", os.path.join(runs, "scaling", "t10_svk_newton_res32", "x.csv"),
         os.path.join(runs, "tolerance", "t10_svk_tol1e-4_res32", "x.csv"), "x_position", 32),
        ("ancf3243_res32", os.path.join(runs, "scaling", "a3243_newton_res32", "tip.csv"),
         os.path.join(runs, "tolerance", "a3243_tol1e-4_res32", "tip.csv"), "tip_z", None),
    ]
    for r in [8, 16, 32]:
        pairs.append((f"ancf3443_res{r}", os.path.join(runs, "scaling", f"a3443_newton_res{r}", "tip.csv"),
                      os.path.join(runs, "tolerance", f"a3443_tol1e-4_res{r}", "tip.csv"), "tip_z", None))
    for name, loose, tight, col, t10res in pairs:
        a, b = load_series(loose, col), load_series(tight, col)
        if a is None or b is None:
            continue
        if t10res is not None:
            x0 = node_x0(t10res, T10_TRACKED[t10res])
            a, b = a - x0, b - x0
        rec = {"rel_diff_1e-3_vs_1e-4": rel_err(a, b)}
        if t10res is not None:
            ref = load_series(os.path.join(runs, "reference", "t10_svk_ref_res32", "x.csv"), col)
            if ref is not None:
                ref = ref - node_x0(32, T10_TRACKED[32])
                rec["err_1e-3_vs_ref"] = rel_err(a, ref)
                rec["err_1e-4_vs_ref"] = rel_err(b, ref)
        out[name] = rec
    return out


def phase_summary(log, skip_first=True):
    text = read(log)
    recs = [kv(m) for m in PHASE_RE.findall(text)]
    if skip_first and recs:
        n_dofs = recs[0]["n_dofs"]
        first = [i for i, r in enumerate(recs) if r["n_dofs"] == n_dofs][0]
        recs = recs[first + 1:]
    if not recs:
        return None
    keys = ["p_ms", "fint_ms", "con_ms", "grad_ms", "hess_ms", "analysis_ms", "factor_ms",
            "refactor_ms", "solve_ms", "ls_ms", "step_ms", "newton", "n_factor",
            "n_refactor", "n_analysis"]
    return {k: float(np.mean([r[k] for r in recs])) for k in keys} | {"steps": len(recs)}


def breakdown(runs):
    out = {}
    for d in sorted(glob.glob(os.path.join(runs, "breakdown", "*"))):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        s = phase_summary(os.path.join(d, "run.log"))
        text = read(os.path.join(d, "run.log"))
        m = re.search(r"SPARSITY_BUILD_MS=([0-9.]+)", text)
        a = re.search(r"CuDSS analysis done \(([0-9.]+) s\)", text)
        if s:
            s["sparsity_build_ms"] = float(m.group(1)) if m else None
            s["analysis_once_ms"] = 1000 * float(a.group(1)) if a else None
            out[os.path.basename(d)] = s
    return out


def vase(runs):
    d = os.path.join(runs, "vase", "vase_eva80")
    if not os.path.exists(os.path.join(d, "DONE")):
        return None
    text = read(os.path.join(d, "run.log"))
    v = [kv(m) for m in VASE_RE.findall(text)]
    contact = [r for r in v if r["contacts"] > 0 and r["step"] > 0]
    free = [r for r in v if r["contacts"] == 0 and r["step"] > 0]
    out = {}
    for label, rows in [("contact", contact), ("no_contact", free)]:
        if rows:
            coll = np.mean([r["coll_ms"] for r in rows])
            solve = np.mean([r["solve_ms"] for r in rows])
            out[label] = {"steps": len(rows), "coll_ms": float(coll), "solve_ms": float(solve),
                          "coll_frac": float(coll / (coll + solve))}
    phases = [kv(m) for m in PHASE_RE.findall(text)]
    if phases:
        keys = ["p_ms", "fint_ms", "con_ms", "grad_ms", "hess_ms", "factor_ms", "refactor_ms",
                "solve_ms", "ls_ms", "step_ms", "newton"]
        out["newton_phases_sum_per_block_mean"] = {k: float(np.mean([p[k] for p in phases])) for k in keys}
    return out


def impact(runs, mu=0.3, e=1.0, radius=0.15, v_i=1.0):
    out = {}
    for d in sorted(glob.glob(os.path.join(runs, "impact", "impact_*"))):
        name = os.path.basename(d)
        m = re.match(r"impact_th([0-9.]+)_dt([0-9.e+-]+)_E([0-9.e+-]+)", name)
        if not m:
            continue
        th, dt, e_c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        rec = {"theta_deg": th, "dt": dt, "E_c": e_c, "done": os.path.exists(os.path.join(d, "DONE"))}
        csv = os.path.join(d, "rigid.csv")
        if os.path.exists(csv):
            with open(csv) as f:
                head = f.readline().strip().split(",")
                rows = [l.strip().split(",") for l in f if l.strip()]
            col = {h: i for i, h in enumerate(head)}
            last = rows[-1]
            vx = float(last[col["v_com_x"]])
            om = math.sqrt(sum(float(last[col[k]]) ** 2 for k in ["omega_x", "omega_y", "omega_z"]))
            finite = all(math.isfinite(float(x)) for x in last[3:])
            t = math.radians(th)
            vt0, vn0 = v_i * math.sin(t), v_i * math.cos(t)
            et_th = 1 - mu * (1 + e) / math.tan(t)
            om_th = 2.5 * mu * (1 + e) * vn0 / radius
            rec.update({"finite": finite, "e_t": abs(vx) / vt0, "e_t_theory": et_th,
                        "omega": om, "omega_theory": om_th,
                        "e_t_rel_err": abs(abs(vx) / vt0 - et_th) / et_th,
                        "omega_rel_err": abs(om - om_th) / om_th})
        log = os.path.join(d, "run.log")
        if os.path.exists(log):
            text = read(log)
            ph = [kv(x) for x in PHASE_RE.findall(text)]
            if ph:
                rec["max_newton_per_step"] = max(p["newton"] for p in ph)
                rec["mean_newton_per_step"] = float(np.mean([p["newton"] for p in ph]))
            rec["inner_not_converged"] = text.count("did not satisfy tolerance")
            rec["nan"] = re.search(r"\bnan\b", text, re.IGNORECASE) is not None
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scal = scaling(args.runs)
    res = {"scaling": scal, "accuracy": accuracy(args.runs, scal), "vm": vm_compare(args.runs),
           "tolerance": tolerance(args.runs), "breakdown": breakdown(args.runs),
           "vase": vase(args.runs), "impact": impact(args.runs)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
