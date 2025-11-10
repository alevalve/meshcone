import torch
from pytorch3d.io import load_objs_as_meshes


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