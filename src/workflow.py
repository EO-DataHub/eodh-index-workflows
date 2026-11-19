"""
Dispatcher: runs the product named by the `workflow` parameter.

Each module in `products/` is one product. It sets `NAME` (the value of `--workflow`) and
`run(inputs, output_dir) -> OutputSpec`. Adding a module adds a product; this file does not change.
"""

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Dict

import products
from eodh_workflow import OutputSpec, WorkflowInputs


def find_products() -> Dict[str, ModuleType]:
    """Every product module in `products/`, keyed by its `NAME`."""
    found: Dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(products.__path__):
        module = importlib.import_module(f"{products.__name__}.{info.name}")
        name = getattr(module, "NAME", None)
        if not name or not callable(getattr(module, "run", None)):
            continue
        if name in found:
            raise ValueError(f"Product name {name!r} is set by both {found[name].__name__} and {module.__name__}")
        found[name] = module
    return found


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    known = find_products()
    name = inputs.params.get("workflow")
    if not name:
        raise ValueError(f"No --workflow parameter. Known workflows: {sorted(known)}")
    if name not in known:
        raise ValueError(f"Unknown workflow {name!r}. Known workflows: {sorted(known)}")
    return known[name].run(inputs, output_dir)
