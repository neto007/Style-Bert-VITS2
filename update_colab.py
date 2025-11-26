import json

notebook_path = "colab.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# 1. Disable use_jp_extra in Cell 13 (approximate index, searching by content)
found_config = False
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "use_jp_extra =" in source and "batch_size =" in source:
            print("Found config cell. Updating use_jp_extra to False.")
            new_source = []
            for line in cell["source"]:
                if "use_jp_extra =" in line:
                    # Update JP extra if needed, or just keep it
                    if "use_jp_extra = True" in line:
                         new_source.append(line.replace("True", "False"))
                    else:
                         new_source.append(line)
                    
                    # Add PT extra with same indentation
                    indent = line[:line.find("use_jp_extra")]
                    new_source.append(f"{indent}use_pt_extra = True\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            found_config = True
            break

if not found_config:
    print("WARNING: Could not find config cell to update use_jp_extra.")

# 2. Update esd.list example in Cell 11 (Markdown)
found_esd = False
for cell in notebook["cells"]:
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "path/to/audio.wav" in source and "ID_do_idioma" in source:
            print("Found esd.list example cell. Updating language ID example.")
            new_source = []
            for line in cell["source"]:
                if "ID_do_idioma, ZH, JP ou EN" in line:
                    new_source.append(line.replace("ZH, JP ou EN", "ZH, JP, EN ou PT"))
                elif "foo.wav|hanako|JP|Olá, como vai?" in line:
                     new_source.append(line.replace("JP", "PT"))
                elif "bar.wav|taro|JP|Sim, estou ouvindo" in line:
                     new_source.append(line.replace("JP", "PT"))
                else:
                    new_source.append(line)
            cell["source"] = new_source
            found_esd = True
            break

if not found_esd:
    print("WARNING: Could not find esd.list example cell.")

# 3. Update preprocess_all call in Cell 15 (approximate)
found_preprocess = False
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "preprocess_all(" in source and "use_jp_extra=use_jp_extra," in source:
            print("Found preprocess_all cell. Adding use_pt_extra.")
            new_source = []
            for line in cell["source"]:
                if "use_jp_extra=use_jp_extra," in line:
                    new_source.append(line)
                    # Add PT extra with same indentation
                    indent = line[:line.find("use_jp_extra")]
                    new_source.append(f"{indent}use_pt_extra=use_pt_extra,\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            found_preprocess = True
            break

if not found_preprocess:
    print("WARNING: Could not find preprocess_all cell.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("colab.ipynb updated successfully.")
