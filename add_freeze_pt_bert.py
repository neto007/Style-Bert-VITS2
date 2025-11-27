#!/usr/bin/env python3
"""Add missing freeze_PT_bert parameter to colab.ipynb"""

import json
from pathlib import Path

# Read the notebook
notebook_path = Path("colab.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find and fix the cell with preprocess_all
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "source" in cell:
        source = "".join(cell["source"])
        
        # Add freeze_PT_bert parameter after freeze_ZH_bert
        if "preprocess_all(" in source and "freeze_ZH_bert=False" in source and "freeze_PT_bert" not in source:
            print("Found preprocess_all without freeze_PT_bert, adding...")
            new_source = []
            for line in cell["source"]:
                new_source.append(line)
                if "freeze_ZH_bert=False," in line:
                    new_source.append("    freeze_PT_bert=False,\n")
            cell["source"] = new_source
            print("Added freeze_PT_bert parameter!")

# Save the corrected notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Notebook {notebook_path} has been updated")
