# MeshCone

Mesh generation using convex optimization through a set of linear constraints and conic solvers on ShapeNetV2.

## Installation

```bash
pip install torch pytorch3d cvxpy numpy
```

## Usage

```bash
python main.py \
    --json_path datahandler/models.json \
    --raw_meshes_dir meshes/raw_meshes \
    --output_dir meshes/refined_meshes \
    --delta 0.05 \
    --lambd 0.1 \
    --num_samples 1000 \
    --device cuda
```

## Parameters

- `delta`: Edge length constraint (default: 0.05)
- `lambd`: Regularization weight (default: 0.1)
- `num_samples`: Sample points from target (default: 1000)
- `device`: cpu or cuda (default: cuda)

## License 

MIT 

## Author

- Alexander Valverde Guillen
- Contact: alexandervalverdeguillen@gmail.com

