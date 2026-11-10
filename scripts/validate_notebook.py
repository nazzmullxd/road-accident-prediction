"""Check notebook structure and existing temporal protocol without running training."""

import ast
from dataclasses import dataclass
from pathlib import Path

import nbformat


def main():
    path = Path(__file__).resolve().parents[1] / 'road-accident-prediction.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    definitions = {}
    count = 0
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != 'code':
            continue
        if cell.outputs or cell.execution_count is not None:
            raise ValueError(f'Cell {index} contains saved execution state')
        tree = ast.parse(cell.source, filename=f'{path.name}:cell-{index}')
        compile(tree, f'{path.name}:cell-{index}', 'exec')
        count += 1
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name in definitions:
                    raise ValueError(f'Duplicate definition: {node.name}')
                definitions[node.name] = node

    # Execute only the existing pure configuration and temporal-window checks.
    # All training, I/O, imports, and orchestration cells remain unexecuted.
    names = ('UserConfig', 'intervals', 'valid_anchor', 'protocol_self_checks')
    subset = ast.Module(body=[definitions[name] for name in names], type_ignores=[])
    namespace = {'dataclass': dataclass, 'MANIFEST': None}
    exec(compile(subset, str(path), 'exec'), namespace)
    namespace['CFG'] = namespace['UserConfig']()
    namespace['protocol_self_checks']()
    print(f'Validated notebook schema and Python syntax for {count} code cells.')


if __name__ == '__main__':
    main()
