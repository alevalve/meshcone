import json
import shutil
from pathlib import Path

# --- Config ---
JSON_PATH = Path("/data1/alex/convmesh/datahandler/models.json")  # your JSON file
OUTPUT_DIR = Path("/data1/alex/convmesh/meshes/original_meshes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load JSON ---
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# --- Copy and rename ---
count = 0
for category, entries in data.items():
    if not entries:
        continue
    src = Path(entries[0]["path"])  # first model per category
    if src.exists():
        dst = OUTPUT_DIR / f"{category}.obj"
        shutil.copy2(src, dst)
        print(f"Copied {src.name} → {dst.name}")
        count += 1
    else:
        print(f"⚠️ Missing: {src}")

print(f"Done! Copied {count} meshes to {OUTPUT_DIR}")
