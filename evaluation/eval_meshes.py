#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import torch, numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def pair_files(refined_dir: Path, gt_dir: Path):
    pairs = []
    gts = {p.stem: p for p in gt_dir.glob("*.obj")}
    for rp in sorted(refined_dir.glob("*.obj")):
        base = rp.stem.removesuffix("_refined")
        gp = gts.get(base)
        if gp is not None: pairs.append((rp, gp))
    print(f"[pairing] pairs={len(pairs)}"); 
    return pairs

def load_pts(path: Path, n: int, device):
    mesh = load_objs_as_meshes([str(path)], device=device)
    return sample_points_from_meshes(mesh, n).squeeze(0)

def norm_unit_sphere(pts: torch.Tensor, eps=1e-12):
    c = pts.mean(0, keepdim=True)
    p0 = pts - c
    s = torch.linalg.norm(p0, dim=1).max().clamp_min(eps)
    return p0 / s

def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    a1, b1 = a.unsqueeze(0), b.unsqueeze(0)
    d_ab = knn_points(a1, b1, K=1).dists.squeeze().squeeze(-1)   # squared
    d_ba = knn_points(b1, a1, K=1).dists.squeeze().squeeze(-1)
    return float(((d_ab.mean() + d_ba.mean()) * 0.5).item())

def f1_at_tau(a: torch.Tensor, b: torch.Tensor, tau: float) -> float:
    a1, b1 = a.unsqueeze(0), b.unsqueeze(0)
    d_ab = knn_points(a1, b1, K=1).dists.squeeze().squeeze(-1).sqrt()
    d_ba = knn_points(b1, a1, K=1).dists.squeeze().squeeze(-1).sqrt()
    prec = (d_ab <= tau).float().mean().item()
    rec  = (d_ba <= tau).float().mean().item()
    return 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec)

def bbox_diag(pts: torch.Tensor) -> float:
    mn, mx = pts.min(0).values, pts.max(0).values
    return float(torch.norm(mx - mn).item())

def emd_hungarian(a: torch.Tensor, b: torch.Tensor) -> float:
    if not SCIPY_OK: return float("nan")
    N = min(len(a), len(b))
    if len(a)!=N: a = a[torch.randperm(len(a), device=a.device)[:N]]
    if len(b)!=N: b = b[torch.randperm(len(b), device=b.device)[:N]]
    C = torch.cdist(a, b).cpu().numpy()
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())

def main():
    ap = argparse.ArgumentParser("Tiny mesh eval (CD, F1, optional EMD) with unit-sphere normalization")
    ap.add_argument("--refined_dir", required=True)
    ap.add_argument("--gt_dir",      required=True)
    ap.add_argument("--n_points",    type=int, default=5000)
    ap.add_argument("--emd_points",  type=int, default=0, help=">0 enables EMD (slow, needs SciPy)")
    ap.add_argument("--tau",         type=float, default=0.01, help="F1 threshold; default=1% of GT bbox diag")
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--out",         type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    pairs = pair_files(Path(args.refined_dir), Path(args.gt_dir))
    if not pairs: 
        print("No pairs. Expect names like ID_refined.obj ↔ ID.obj"); return
    if args.emd_points>0 and not SCIPY_OK:
        print("SciPy not found; EMD will be NaN")

    rows, cds, f1s, emds = [], [], [], []
    for rp, gp in pairs:
        base = rp.stem.removesuffix("_refined")
        try:
            r = norm_unit_sphere(load_pts(rp, args.n_points, device))
            g = norm_unit_sphere(load_pts(gp, args.n_points, device))
            cd = chamfer(r, g)
            tau = args.tau if args.tau is not None else 0.01 * bbox_diag(g)
            f1 = f1_at_tau(r, g, tau)
            if args.emd_points>0:
                r_e = norm_unit_sphere(load_pts(rp, args.emd_points, device))
                g_e = norm_unit_sphere(load_pts(gp, args.emd_points, device))
                emd = emd_hungarian(r_e, g_e)
            else:
                emd = float("nan")
            rows.append((base, cd, emd, f1, tau))
            cds.append(cd); f1s.append(f1); 
            if not math.isnan(emd): emds.append(emd)
            print(f"{base} | CD={cd:.6f}  EMD={emd if not math.isnan(emd) else float('nan'):.6f}  F1@{tau:.4g}={f1:.4f}")
        except Exception as e:
            print(f"Error {base}: {e}")

    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    mcd, memd, mf1 = mean(cds), mean(emds), mean(f1s)
    print("\n==== Averages ====\n"
          f"CD  : {mcd:.6f}\n"
          f"EMD : {memd if not math.isnan(memd) else float('nan'):.6f}\n"
          f"F1  : {mf1:.6f}")

    if args.out:
        import csv
        outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["id","chamfer","emd","f1","tau"])
            for r in rows: w.writerow([r[0], f"{r[1]:.8f}", f"{(r[2] if not math.isnan(r[2]) else float('nan')):.8f}", f"{r[3]:.8f}", f"{r[4]:.8f}"])
        print(f"Saved CSV to: {outp}")

if __name__ == "__main__":
    main()
