import torch
import numpy as np
import cvxpy as cp
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes


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