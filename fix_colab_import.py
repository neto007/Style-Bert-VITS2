#!/usr/bin/env python3
"""Fix colab.ipynb to use correct pyopenjtalk_worker import"""

import json
from pathlib import Path

# Read the notebook
notebook_path = Path("colab.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find and fix the cell with incorrect import
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "source" in cell:
        source = "".join(cell["source"])
        
        # Fix the incorrect import
        if "from style_bert_vits2.nlp.portuguese import pyopenjtalk_worker" in source:
            print("Found incorrect import, fixing...")
            cell["source"] = [line.replace(
                "from style_bert_vits2.nlp.portuguese import pyopenjtalk_worker",
                "from style_bert_vits2.nlp.japanese import pyopenjtalk_worker"
            ) for line in cell["source"]]
            print("Fixed!")

# Save the corrected notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Notebook {notebook_path} has been corrected")
