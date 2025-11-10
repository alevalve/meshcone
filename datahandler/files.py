import os
import json

def generate_shapenet_json(shapenet_root, output_file):
    """
    Simple function to generate ShapeNet JSON paths with subfolder structure
    """
    paths = {}
    
    # Iterate through category folders
    for category in sorted(os.listdir(shapenet_root)):
        category_path = os.path.join(shapenet_root, category)
        
        if os.path.isdir(category_path):
        
            subcategories = sorted([d for d in os.listdir(category_path) 
                                  if os.path.isdir(os.path.join(category_path, d))])
            
            if subcategories:
                first_subcategory = subcategories[0]
                subcategory_path = os.path.join(category_path, first_subcategory)
                
                
                subfolders = sorted([d for d in os.listdir(subcategory_path) 
                                   if os.path.isdir(os.path.join(subcategory_path, d))])
                
                if subfolders:
                    first_subfolder = subfolders[0]
                    obj_path = os.path.join(subcategory_path, first_subfolder, "models", "model_normalized.obj")
                    paths[category] = obj_path
                    print(f"Category {category}: {obj_path}")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(paths, f, indent=2)
    
    return paths

# Usage
shapenet_root = "/data1/alex/datasets/shapenet_data"
output_json = "shapenet_paths.json"
generate_shapenet_json(shapenet_root, output_json)