import torch
import time
from pytorch3d.io import save_obj
from mesh_manager import MeshManager
from convex_optimizer import FastConvexOptim


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
        optimized_verts = torch.tensor(
            result["optimized_vertices"], 
            dtype=torch.float32, 
            device=device
        )
        faces = raw_mesh.faces_packed()
        save_obj(output_mesh_path, optimized_verts, faces)
        print(f"Saved to {output_mesh_path}")
        return True, elapsed
    else:
        print(f"Failed to optimize {raw_mesh_path}")
        return False, elapsed