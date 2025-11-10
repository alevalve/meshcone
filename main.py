import torch
import numpy as np
import cvxpy as cp
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import argparse
import json
import time
import os
from pathlib import Path


class FastConvexOptim:
    def __init__(self, raw_mesh: Meshes, target_mesh: Meshes, num_samples: int, device="cpu"):
        self.device = torch.device(device)
        self.raw_mesh = raw_mesh
        self.target_mesh = target_mesh
        self.num_samples = num_samples
        
        self.target_points = (
            sample_points_from_meshes(self.target_mesh, self.num_samples)
            .squeeze()
            .cpu()
            .numpy()
        )
        
        self.raw_verts_np = self.raw_mesh.verts_packed().cpu().numpy()
        self.raw_faces_np = self.raw_mesh.faces_packed().cpu().numpy()
        self.tgt_verts_np = self.target_mesh.verts_packed().cpu().numpy()
        
        self.use_target_reg = (self.tgt_verts_np.shape[0] == self.raw_verts_np.shape[0])
        self.edges = self._extract_unique_edges()
        self.warm_start = None
        
    def _extract_unique_edges(self):
        edges_set = set()
        for face in self.raw_faces_np:
            for i in range(3):
                a = int(face[i])
                b = int(face[(i + 1) % 3])
                edge = tuple(sorted([a, b]))
                edges_set.add(edge)
        return np.array(list(edges_set))
    
    def build_vectorized_problem(self, delta: float, lambd: float):
        V = self.raw_verts_np.shape[0]
        vertices_cvx = cp.Variable((V, 3))
        
        if self.warm_start is not None:
            vertices_cvx.value = self.warm_start
        else:
            vertices_cvx.value = self.raw_verts_np.copy()
        
        edge_diffs = vertices_cvx[self.edges[:, 0], :] - vertices_cvx[self.edges[:, 1], :]
        edge_norms = cp.norm(edge_diffs, axis=1)
        constraints = [edge_norms <= delta]

        # Regularization: to target if same topology, else to raw
        if self.use_target_reg:
            regularization = cp.sum_squares(vertices_cvx - self.tgt_verts_np)
        else:
            regularization = cp.sum_squares(vertices_cvx - self.raw_verts_np)

        # Align to target centroid
        mu = self.target_points.mean(axis=0, keepdims=True)
        approx_alignment = cp.sum_squares(vertices_cvx - mu)

        objective = cp.Minimize(approx_alignment + lambd * regularization)
        problem = cp.Problem(objective, constraints)
        return problem, vertices_cvx
    
    def solve_fast(self, delta: float, lambd: float, max_iters: int = 1000):
        problem, vertices_cvx = self.build_vectorized_problem(delta=delta, lambd=lambd)
        
        problem.solve(
            solver=cp.SCS,
            max_iters=max_iters,
            eps=1e-5,
            warm_start=True,
            verbose=False
        )
        
        if vertices_cvx.value is not None:
            self.warm_start = vertices_cvx.value.copy()
        
        return {
            "status": problem.status,
            "optimal_value": problem.value,
            "optimized_vertices": vertices_cvx.value,
        }


class MeshManager:
    def __init__(self, mesh_path: str, device="cpu"):
        self.device = device
        self.mesh_path = mesh_path
        self.mesh = self.load_mesh()
    
    def load_mesh(self):
        mesh = load_objs_as_meshes([self.mesh_path], device=self.device)
        return mesh
    
    def get_mesh(self):
        return self.mesh


def process_single_mesh(raw_mesh_path, target_mesh_path, output_mesh_path, args):
    device = torch.device(args.device)
    start_time = time.time()
    
    
    raw_manager = MeshManager(raw_mesh_path, device=device)
    target_manager = MeshManager(target_mesh_path, device=device)
    raw_mesh = raw_manager.get_mesh()
    target_mesh = target_manager.get_mesh()
    
    opt = FastConvexOptim(
        raw_mesh=raw_mesh,
        target_mesh=target_mesh,
        num_samples=args.num_samples,
        device=args.device,
    )
    
    result = opt.solve_fast(delta=args.delta, lambd=args.lambd)

    elapsed = time.time() - start_time

    if result["optimized_vertices"] is not None:
        optimized_verts = torch.tensor(result["optimized_vertices"], dtype=torch.float32, device=device)
        faces = raw_mesh.faces_packed()
        save_obj(output_mesh_path, optimized_verts, faces)
        print(f"Saved to {output_mesh_path}")
        return True, elapsed
    else:
        print(f"Failed to optimize {raw_mesh_path}")
        return False, elapsed


def main(args):
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    raw_meshes_dir = Path(args.raw_meshes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_count = 0
    total_time = 0.0 

    global_start = time.time()

    for category_id, items in data.items():
        
        for item in items:
            if isinstance(item, dict) and 'model_id' in item:
                model_id = category_id  
                target_path = item.get('path', '')
                
                raw_mesh_name = f"{model_id}_raw.obj"
                raw_mesh_path = raw_meshes_dir / raw_mesh_name
                
                output_mesh_name = f"{model_id}_refined.obj"
                output_mesh_path = output_dir / output_mesh_name
                
                total_count += 1
                success, mesh_time = process_single_mesh(str(raw_mesh_path), target_path, str(output_mesh_path), args)
                
                total_time += mesh_time
                if success:
                    success_count += 1
    
    global_elapsed = time.time() - global_start
    avg_time = total_time / max(1, total_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convex optimization for mesh refinement")
    parser.add_argument("--json_path", type=str, 
                       default="/data1/alex/convmesh/datahandler/models.json",
                       help="Path to JSON file containing mesh paths")
    parser.add_argument("--raw_meshes_dir", type=str, 
                       default="/data1/alex/convmesh/meshes/raw_meshes",
                       help="Directory containing raw meshes")
    parser.add_argument("--output_dir", type=str, 
                       default="/data1/alex/convmesh/meshes/refined_meshes",
                       help="Directory to save refined meshes")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)