import os
import json
import re

# Function to create a valid HTML anchor from a comment
def create_anchor(comment):
    return re.sub(r'\W+', '-', comment.strip().lower())

# Function to update a single notebook
def update_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Remove existing index cell(s)
    notebook['cells'] = [cell for cell in notebook['cells'] if not (
        cell['cell_type'] == 'markdown' and 
        "# Code Block Index" in ''.join(cell.get('source', []))
    )]

    # Create the index cell
    index_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Code Block Index\n\n"
        ]
    }

    # Track the cells to be added
    new_cells = []
    code_block_counter = 1

    # Track existing anchors
    existing_anchors = set()

    # Extract existing anchors from notebook
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            for line in cell['source']:
                match = re.search(r'<a id="(.*?)"></a>', line)
                if match:
                    existing_anchors.add(match.group(1))

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            if cell['source'] and cell['source'][0].startswith("#"):
                comment = cell['source'][0].strip("# ").strip()
                label = create_anchor(comment)
                if label not in existing_anchors:
                    # Add a new Markdown cell with the anchor before the code cell
                    anchor_cell = {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f'<a id="{label}"></a>'
                        ]
                    }
                    new_cells.append(anchor_cell)
                    existing_anchors.add(label)
                index_entry = f'- [{comment}](#{label})\n'
                index_cell['source'].append(index_entry)
            else:
                label = f'code-block-{code_block_counter}'
                if label not in existing_anchors:
                    # Add a new Markdown cell with the anchor before the code cell
                    anchor_cell = {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f'<a id="{label}"></a>'
                        ]
                    }
                    new_cells.append(anchor_cell)
                    existing_anchors.add(label)
                index_entry = f'- [Code Block {code_block_counter}](#{label})\n'
                index_cell['source'].append(index_entry)
                code_block_counter += 1
        new_cells.append(cell)

    # Insert the index cell at the beginning of the notebook
    new_cells.insert(0, index_cell)
    notebook['cells'] = new_cells

    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"Notebook updated and saved to {notebook_path}")

# Iterate over all ipynb files in the current directory and update them
for filename in os.listdir('.'):
    if filename.endswith('.ipynb'):
        update_notebook(filename)
