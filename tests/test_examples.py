"""Smoke tests: every example in examples/ must run its main() without error.

Keeps examples/ from silently rotting as internal APIs evolve.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / 'examples'
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob('*.py'))


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        msg = f'could not load a module spec for {path}.'
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('path', EXAMPLE_FILES, ids=lambda p: p.stem)
def test_example_runs(path):
    module = _load_module(path)
    assert hasattr(module, 'main'), f'{path.name} must define a main() function'
    module.main()
