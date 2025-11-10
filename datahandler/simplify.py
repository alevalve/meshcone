#!/usr/bin/env python3
"""
Convert OBJ files with normals/textures to simple format (vertices + faces only)
Usage: python simplify_obj.py input.obj output.obj
       python simplify_obj.py --batch input_dir output_dir
"""

import argparse
from pathlib import Path


def simplify_obj(input_path: Path, output_path: Path):
    """Convert OBJ to simple format with only vertices and faces"""
    
    vertices = []
    faces = []
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # Extract vertices
            if parts[0] == 'v':
                # v x y z [w]
                vertices.append(f"v {parts[1]} {parts[2]} {parts[3]}\n")
            
            # Extract faces and strip texture/normal references
            elif parts[0] == 'f':
                # Convert formats like:
                # f v1//vn1 v2//vn2 v3//vn3  ->  f v1 v2 v3
                # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3  ->  f v1 v2 v3
                # f v1/vt1 v2/vt2 v3/vt3  ->  f v1 v2 v3
                simple_verts = []
                for vert_ref in parts[1:]:
                    # Take only the first number (vertex index)
                    v_idx = vert_ref.split('/')[0]
                    simple_verts.append(v_idx)
                
                faces.append(f"f {' '.join(simple_verts)}\n")
    
    # Write simplified OBJ
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        # Write vertices
        f.writelines(vertices)
        # Write faces
        f.writelines(faces)
    
    print(f"✓ Converted: {input_path.name}")
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    return len(vertices), len(faces)


def batch_simplify(input_dir: Path, output_dir: Path):
    """Batch process all OBJ files in a directory"""
    obj_files = sorted(list(input_dir.rglob("*.obj")))
    
    if not obj_files:
        print(f"No OBJ files found in {input_dir}")
        return
    
    print(f"Found {len(obj_files)} OBJ files to process\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_verts = 0
    total_faces = 0
    
    for i, obj_path in enumerate(obj_files, 1):
        print(f"[{i}/{len(obj_files)}] Processing {obj_path.name}...")
        
        # Preserve relative path structure
        rel_path = obj_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        try:
            v_count, f_count = simplify_obj(obj_path, output_path)
            total_verts += v_count
            total_faces += f_count
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Batch conversion complete!")
    print(f"Total vertices: {total_verts:,}")
    print(f"Total faces: {total_faces:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Simplify OBJ files to vertices + faces only"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input OBJ file or directory"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output OBJ file or directory"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all OBJ files in input directory"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return
    
    if args.batch:
        if not input_path.is_dir():
            print(f"Error: {input_path} must be a directory for batch processing")
            return
        batch_simplify(input_path, output_path)
    else:
        if input_path.is_dir():
            print("Error: Use --batch flag for directory processing")
            return
        simplify_obj(input_path, output_path)


if __name__ == "__main__":
    main()