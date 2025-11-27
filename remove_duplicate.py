#!/usr/bin/env python3
"""Remove duplicate use_pt_extra from colab.ipynb"""

import json
from pathlib import Path

# Read the notebook
notebook_path = Path("colab.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find and fix the cell with duplicate use_pt_extra
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "source" in cell:
        source = "".join(cell["source"])
        
        # Remove duplicate use_pt_extra
        if "preprocess_all(" in source:
            new_source = []
            use_pt_extra_count = 0
            for line in cell["source"]:
                if "use_pt_extra=use_pt_extra," in line:
                    use_pt_extra_count += 1
                    if use_pt_extra_count == 1:
                        new_source.append(line)
                    else:
                        print(f"Removing duplicate line: {line.strip()}")
                else:
                    new_source.append(line)
            
            if use_pt_extra_count > 1:
                cell["source"] = new_source
                print("Removed duplicate use_pt_extra!")

# Save the corrected notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Notebook {notebook_path} has been cleaned")
