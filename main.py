import argparse
import json
import time
from pathlib import Path
from scripts.mesh_processor import process_single_mesh


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
                success, mesh_time = process_single_mesh(
                    str(raw_mesh_path), 
                    target_path, 
                    str(output_mesh_path), 
                    args
                )
                
                total_time += mesh_time
                if success:
                    success_count += 1
    
    global_elapsed = time.time() - global_start
    avg_time = total_time / max(1, total_count)
    
    print(f"\n{'='*50}")
    print(f"Batch Processing Complete")
    print(f"{'='*50}")
    print(f"Total meshes: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Total time: {global_elapsed:.2f}s")
    print(f"Average time per mesh: {avg_time:.2f}s")


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
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
