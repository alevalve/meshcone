import torch
import json
import os
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
JSON_PATH = "shapenet_paths.json"
OUTPUT_DIR = "/data1/alex/convmesh/meshes/raw_meshes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load mesh paths
with open(JSON_PATH, 'r') as f:
    mesh_paths = json.load(f)

print(f"Processing {len(mesh_paths)} meshes...")

for category, obj_path in mesh_paths.items():
    print(f"\nProcessing {category}...")
    
    if not os.path.exists(obj_path):
        print(f"  File not found: {obj_path}")
        continue
    
    try:
        # Load target mesh
        verts, faces, _ = load_obj(obj_path)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)
        
        # Normalize
        center = verts.mean(0)
        verts = verts - center
        scale = verts.abs().max()
        verts = verts / scale
        
        target_mesh = Meshes(verts=[verts], faces=[faces_idx])
        
        # Initialize random mesh with same topology
        target_verts = target_mesh.verts_packed()
        target_faces = target_mesh.faces_packed()
        init_verts = torch.randn(target_verts.shape, device=device) * 0.5
        
        src_mesh = Meshes(verts=[init_verts], faces=[target_faces])
        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
        
        # Optimizer
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
        
        # Optimization loop
        for i in range(100):
            optimizer.zero_grad()
            new_src_mesh = src_mesh.offset_verts(deform_verts)
            
            # Losses
            sample_trg = sample_points_from_meshes(target_mesh, 1000)
            sample_src = sample_points_from_meshes(new_src_mesh, 1000)
            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
            loss_edge = mesh_edge_loss(new_src_mesh)
            loss_normal = mesh_normal_consistency(new_src_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh)
            
            total_loss = loss_chamfer + loss_edge + 0.01 * loss_normal + 0.1 * loss_laplacian
            total_loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"  Iter {i}, Loss: {total_loss.item():.4f}")
        
        # Save result
        output_path = os.path.join(OUTPUT_DIR, f"{category}_raw.obj")
        verts, faces = new_src_mesh.get_mesh_verts_faces(0)
        save_obj(output_path, verts, faces)
        print(f"  Saved: {output_path}")
        
    except Exception as e:
        print(f"  Error: {e}")

print(f"\nDone! All meshes saved to {OUTPUT_DIR}")