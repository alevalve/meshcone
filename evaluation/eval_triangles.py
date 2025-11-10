#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

def pair_files(refined_dir: Path, gt_dir: Path):
    pairs, gts = [], {p.stem: p for p in gt_dir.glob("*.obj")}
    for rp in sorted(refined_dir.glob("*.obj")):
        base = rp.stem.replace("_refined", "")
        if base in gts: pairs.append((rp, gts[base]))
    return pairs

def norm_unit_sphere_mesh(mesh: Meshes, eps=1e-12) -> Meshes:
    V = mesh.verts_packed()
    c = V.mean(0, keepdim=True); V0 = V - c
    s = torch.linalg.norm(V0, dim=1).max().clamp_min(eps)
    Vn = V0 / s
    F = mesh.faces_packed()
    return Meshes(verts=[Vn], faces=[F])


def tri_metrics(mesh: Meshes, min_area=1e-8):
    V = mesh.verts_packed(); F = mesh.faces_packed()
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    e0, e1, e2 = (v1 - v0), (v2 - v1), (v0 - v2)
    a = torch.linalg.norm(e0, dim=1)
    b = torch.linalg.norm(e1, dim=1)
    c = torch.linalg.norm(e2, dim=1)
    
    # Calculate area first
    A = 0.5 * torch.linalg.norm(torch.cross(e0, v2 - v0, dim=1), dim=1)
    
    # Filter out degenerate triangles
    valid_mask = A > min_area
    
    if valid_mask.sum() == 0:
        print("WARNING: All triangles are degenerate!")
        return torch.tensor([float('nan')]), torch.tensor([float('nan')])
    
    # Only compute metrics for valid triangles
    a, b, c = a[valid_mask], b[valid_mask], c[valid_mask]
    A = A[valid_mask]
    
    # Aspect ratio
    edges = torch.stack([a, b, c], dim=1)
    ar = edges.max(dim=1).values / edges.min(dim=1).values.clamp_min(1e-12)
    
    # Mean-ratio quality
    q = (4.0*math.sqrt(3.0)*A) / (a*a + b*b + c*c + 1e-12)
    
    return ar, q


def compute_hausdorff_distance(mesh1: Meshes, mesh2: Meshes, num_samples: int = 10000):
    """
    Compute bidirectional Hausdorff distance between two meshes.
    Returns the maximum of the two directed Hausdorff distances.
    """
    points1 = sample_points_from_meshes(mesh1, num_samples).squeeze()
    points2 = sample_points_from_meshes(mesh2, num_samples).squeeze()
    
    # Directed Hausdorff: mesh1 -> mesh2
    dists_1to2 = torch.cdist(points1, points2)
    min_dists_1to2 = dists_1to2.min(dim=1)[0]
    hausdorff_1to2 = min_dists_1to2.max()
    
    # Directed Hausdorff: mesh2 -> mesh1
    dists_2to1 = torch.cdist(points2, points1)
    min_dists_2to1 = dists_2to1.min(dim=1)[0]
    hausdorff_2to1 = min_dists_2to1.max()
    
    return max(hausdorff_1to2.item(), hausdorff_2to1.item())


def compute_normal_consistency(mesh1: Meshes, mesh2: Meshes, num_samples: int = 10000):
    """
    Compute normal consistency score between two meshes.
    Returns the average cosine similarity of normals at corresponding points.
    """
    # Sample points with normals
    points1, normals1 = sample_points_from_meshes(mesh1, num_samples, return_normals=True)
    points2, normals2 = sample_points_from_meshes(mesh2, num_samples, return_normals=True)
    
    points1, normals1 = points1.squeeze(), normals1.squeeze()
    points2, normals2 = points2.squeeze(), normals2.squeeze()
    
    # Find nearest neighbors
    dists = torch.cdist(points1, points2)
    nearest_idx = dists.argmin(dim=1)
    
    # Get corresponding normals
    normals2_matched = normals2[nearest_idx]
    
    # Compute cosine similarity (dot product of unit normals)
    cos_sim = (normals1 * normals2_matched).sum(dim=1)
    
    # Return average absolute cosine similarity
    return cos_sim.abs().mean().item()


def compute_curvature_error(mesh1: Meshes, mesh2: Meshes, num_samples: int = 5000):
    """
    Compute curvature error between two meshes.
    Approximates mean curvature using local point neighborhoods.
    """
    points1 = sample_points_from_meshes(mesh1, num_samples).squeeze()
    points2 = sample_points_from_meshes(mesh2, num_samples).squeeze()
    
    def estimate_curvature(points, k=10):
        # For each point, find k nearest neighbors
        dists = torch.cdist(points, points)
        _, knn_idx = dists.topk(k + 1, largest=False, dim=1)
        knn_idx = knn_idx[:, 1:]  # Exclude self
        
        curvatures = []
        for i in range(points.shape[0]):
            neighbors = points[knn_idx[i]]
            centered = neighbors - points[i].unsqueeze(0)
            
            # Estimate local curvature via covariance eigenvalues
            cov = torch.matmul(centered.T, centered) / k
            eigenvalues = torch.linalg.eigvalsh(cov)
            curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
            curvatures.append(curvature)
        
        return torch.stack(curvatures)
    
    curv1 = estimate_curvature(points1)
    curv2 = estimate_curvature(points2)
    
    # Find nearest neighbors between meshes
    dists = torch.cdist(points1, points2)
    nearest_idx = dists.argmin(dim=1)
    curv2_matched = curv2[nearest_idx]
    
    # Compute mean absolute difference
    curvature_error = (curv1 - curv2_matched).abs().mean()
    
    return curvature_error.item()


def main():
    ap = argparse.ArgumentParser("Triangle quality only (aspect ratio & mean-ratio) on refined meshes")
    ap.add_argument("--refined_dir", required=True)
    ap.add_argument("--gt_dir",      required=True)
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--ar_thresh",   type=float, default=3.0, help="Flag skinny tris: AR > this")
    ap.add_argument("--q_thresh",    type=float, default=0.30, help="Flag poor tris: Q < this")
    ap.add_argument("--out",         type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    pairs = pair_files(Path(args.refined_dir), Path(args.gt_dir))
    if not pairs:
        print("No pairs. Expect ID_refined.obj ↔ ID.obj"); return

    rows = []
    ar_means, q_means, bad_ar_pct, bad_q_pct = [], [], [], []
    hd_vals, nc_vals, ce_vals = [], [], []

    for rp, gtp in pairs:
        base = rp.stem.replace("_refined","")
        try:
            m = load_objs_as_meshes([str(rp)], device=device)
            m = norm_unit_sphere_mesh(m)
            
            gt = load_objs_as_meshes([str(gtp)], device=device)
            gt = norm_unit_sphere_mesh(gt)
            
            ar, q = tri_metrics(m)
            ar_mean = float(ar.mean().item())
            q_mean  = float(q.mean().item())
            ar_bad  = float((ar > args.ar_thresh).float().mean().item())
            q_bad  = float((q < args.q_thresh).float().mean().item())
            
            hd = compute_hausdorff_distance(m, gt)
            nc = compute_normal_consistency(m, gt)
            ce = compute_curvature_error(m, gt)

            rows.append((base, ar_mean, q_mean, ar_bad, q_bad, hd, nc, ce))
            ar_means.append(ar_mean); q_means.append(q_mean)
            bad_ar_pct.append(ar_bad); bad_q_pct.append(q_bad)
            hd_vals.append(hd); nc_vals.append(nc); ce_vals.append(ce)

            print(f"{base} | AR_mean={ar_mean:.3f}  Q_mean={q_mean:.3f}  "
                  f"AR> {args.ar_thresh:g}: {ar_bad*100:.1f}%  Q< {args.q_thresh:g}: {q_bad*100:.1f}%  "
                  f"HD={hd:.6f}  NC={nc:.4f}  CE={ce:.6f}")
        except Exception as e:
            print(f"Error {base}: {e}")

    mean = lambda x: float(sum(x)/len(x)) if x else float("nan")
    print("\n==== Averages (refined) ====")
    print(f"AR_mean : {mean(ar_means):.3f}  (↓ better, 1 = equilateral)")
    print(f"Q_mean  : {mean(q_means):.3f}  (↑ better, 1 = equilateral)")
    print(f"AR>thr  : {mean(bad_ar_pct)*100:.1f}%")
    print(f"Q<thr   : {mean(bad_q_pct)*100:.1f}%")
    print(f"HD      : {mean(hd_vals):.6f}  (↓ better)")
    print(f"NC      : {mean(nc_vals):.4f}  (↑ better)")
    print(f"CE      : {mean(ce_vals):.6f}  (↓ better)")

    if args.out:
        import csv
        outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "aspect_ratio_mean", "mean_ratio_quality_mean", 
                       f"pct_ar_gt_{args.ar_thresh}", f"pct_q_lt_{args.q_thresh}",
                       "hausdorff_distance", "normal_consistency", "curvature_error"])
            for r in rows:
                w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}", 
                           f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}"])
        print(f"Saved CSV to: {outp}")

if __name__ == "__main__":
    main()