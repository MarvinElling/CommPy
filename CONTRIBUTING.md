# Contributing to CommPy

Thank you for your interest in contributing to CommPy! This guide explains how to contribute code, documentation, and improvements.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on code quality and clarity
- Help others learn and improve

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/CommPy.git
cd CommPy
```

### 2. Create a Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

CommPy follows PEP 8 with these conventions:

- **Indentation**: 4 spaces
- **Line length**: 100 characters maximum
- **Naming**: 
  - `Classes`: PascalCase (e.g., `BPSK_Modulator`)
  - `functions`: snake_case (e.g., `shannon_entropy`)
  - `constants`: UPPER_SNAKE_CASE (e.g., `PI = 3.14159`)

### Type Hints

All functions must include complete type hints:

```python
def my_function(
    x: np.ndarray,
    p: float,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Brief description.
    
    Longer description here.
    
    Parameters:
    - x: Input array
    - p: Probability parameter in [0, 1]
    - rng: Optional random number generator
    
    Returns:
    - Output array
    """
    if rng is None:
        rng = np.random.default_rng()
    # Implementation
    return result
```

### Documentation

Every public function/class must have a docstring:

```python
class MyClass:
    """One-line summary.
    
    Longer description explaining purpose and usage.
    """
    
    def my_method(self, x: float) -> float:
        """Brief description.
        
        More detailed explanation if needed.
        
        Parameters:
        - x: Parameter description
        
        Returns:
        - Return value description
        
        Raises:
        - ValueError: When input is invalid
        
        Example:
        >>> obj = MyClass()
        >>> result = obj.my_method(5.0)
        """
        return x * 2
```

### Testing

Write tests for all new features:

```python
# In tests/test_my_feature.py
import numpy as np
from commpy import MyClass

def test_basic_functionality():
    """Test basic operation."""
    obj = MyClass()
    result = obj.my_method(5.0)
    assert result == 10.0

def test_edge_cases():
    """Test edge cases."""
    obj = MyClass()
    assert obj.my_method(0) == 0
    assert obj.my_method(-5) == -10

def test_with_rng():
    """Test with seeded RNG."""
    rng = np.random.default_rng(seed=42)
    obj = MyClass()
    result1 = obj.stochastic_method(rng=rng)
    
    rng = np.random.default_rng(seed=42)
    result2 = obj.stochastic_method(rng=rng)
    
    assert np.allclose(result1, result2)
```

Run tests:

```bash
pytest tests/
pytest tests/test_my_feature.py -v  # Verbose
pytest --cov=commpy tests/          # With coverage
```

### Imports

Organize imports in this order:

```python
# Standard library
import numpy as np
from typing import Optional

# Local/CommPy imports
from commpy._utils.maths import is_prime
```

## Making Changes

### 1. Implement Feature

Create or modify files in `src/commpy/`:

```python
# src/commpy/_mymodule/my_feature.py

"""Module docstring explaining purpose."""

from typing import Optional
import numpy as np


def my_function(x: np.ndarray, param: float) -> np.ndarray:
    """Function docstring with full description."""
    # Implementation
    return result


class MyClass:
    """Class docstring."""
    
    def __init__(self, param: float):
        """Initialize."""
        self.param = param
```

### 2. Update the top-level `__init__.py`

CommPy's subpackages (anything under `src/commpy/_*`) are PEP 420 implicit namespace
packages — they intentionally have **no** `__init__.py` of their own. The *only* public API
surface is the flat re-export list in `src/commpy/__init__.py`; everything under a `_`-prefixed
submodule is private and may be reorganized freely without it being a breaking change.

```python
# src/commpy/__init__.py
from ._mymodule.my_feature import my_function, MyClass

__all__ = [
    # ... existing exports ...
    'my_function',
    'MyClass',
]
```

### 3. Write Tests

```bash
# tests/test_my_feature.py
import pytest
import numpy as np
from commpy import MyClass, my_function
```

### 4. Document Changes

Update relevant documentation:

- **API Reference**: [docs/API.md](docs/API.md)
- **Getting Started**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **README**: [README.md](README.md)

### 5. Create Examples

If appropriate, create example scripts:

```python
# examples/example_my_feature.py

"""Example: Using MyClass for XYZ."""

import numpy as np
from commpy import MyClass

# Create an instance
obj = MyClass(param=0.5)

# Use the class
result = obj.my_method(x=10)
print(f"Result: {result}")
```

## Submitting Changes

### 1. Commit with Clear Messages

```bash
git add .
git commit -m "Add MyClass for feature XYZ

- Implement core functionality
- Add comprehensive documentation
- Add 10 test cases
- Update API reference

Fixes #123"
```

Good commit messages:
- Start with verb: Add, Fix, Remove, Update
- Be specific about what changed
- Reference issues: `Fixes #123`, `Related to #456`

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

On GitHub:
1. Click "New Pull Request"
2. Select `base: main` ← `compare: your-branch`
3. Write clear PR description
4. Link related issues
5. Request review

**PR Template:**

```markdown
## Description
Brief description of changes.

## Motivation
Why is this needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [x] Code follows style guidelines
- [x] Documentation updated
- [x] Tests added/updated
- [x] Type hints complete
- [x] No breaking changes
```

## Code Review Process

When reviewing code:

1. **Check functionality**: Does it work as intended?
2. **Check clarity**: Is code easy to understand?
3. **Check style**: Does it follow guidelines?
4. **Check tests**: Are they comprehensive?
5. **Check documentation**: Is it complete and correct?

Suggestions for improvement:

```markdown
### Style
Consider using snake_case for this function name per PEP 8.

### Implementation
This loop could be vectorized with NumPy for better performance.

### Testing
Adding a test for the edge case when x=0 would be helpful.
```

## Documentation Style Guide

### Docstring Format

```python
def function(param1: str, param2: int = 5) -> bool:
    """One-line summary of what the function does.
    
    Longer description providing context, use cases, and
    important implementation details. Should be clear and
    helpful for users unfamiliar with the code.
    
    Parameters:
    - param1: Description of param1 (type, range, etc.)
    - param2: Description of optional param2 (default: 5)
    
    Returns:
    - Description of return value and its meaning
    
    Raises:
    - ValueError: When param1 is empty or None
    - TypeError: When param2 is not numeric
    
    Example:
    >>> result = function("test", 10)
    >>> print(result)
    True
    """
```

### Documentation Files

- **Module docs**: Explain purpose and classes
- **API references**: List all public functions with signatures
- **Tutorials**: Show practical examples
- **Guides**: Explain concepts and workflows

### Writing Examples

Make examples:
- **Self-contained**: Run without external setup
- **Realistic**: Show actual use cases
- **Well-commented**: Explain each step
- **Progressive**: Build from simple to complex

```python
# Good example
import numpy as np
from commpy import BPSK_Modulator, Channels

# Create 100 random bits
bits = np.random.randint(0, 2, 100)

# Modulate to BPSK symbols
symbols = BPSK_Modulator.modulate(bits)

# Add AWGN at SNR=10dB
noisy = Channels.awgn(symbols, snr_db=10)

# Demodulate and measure errors
recovered = BPSK_Modulator.demodulate(noisy)
ber = np.mean(recovered != bits)
print(f"Bit error rate: {ber:.4f}")
```

## Reporting Issues

### Bug Reports

Include:
- **Title**: Concise description
- **Environment**: Python version, scipy version, OS
- **Reproducible**: Minimal code that triggers bug
- **Expected**: What should happen
- **Actual**: What actually happens
- **Traceback**: Full error message

```markdown
## Description
BPSK demodulation returns wrong values when input contains NaN.

## Environment
- Python 3.11
- NumPy 1.24
- CommPy 0.1.2

## Reproduction
```python
import numpy as np
from commpy import BPSK_Modulator

symbols = np.array([1+0j, np.nan, -1+0j])
recovered = BPSK_Modulator.demodulate(symbols)
print(recovered)  # Expected: [1, ?, 0]
```

## Feature Requests

Include:
- **Motivation**: Why is this needed?
- **Use case**: How would you use it?
- **Proposal**: How should it work?
- **Alternatives**: Other ways to achieve this?

## Setting Up IDE

### VS Code

```json
// .vscode/settings.json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100]
}
```

### PyCharm

1. Settings → Code Style → Python
2. Set line length to 100
3. Enable PEP 8 warnings
4. Right-click directory → Mark as Sources Root

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if exists)
3. Create git tag: `git tag v0.1.3`
4. Push and create GitHub Release

## Questions?

- **Documentation**: Check docs/ directory
- **Examples**: Look in examples/ directory  
- **Issues**: Search existing issues
- **Discussions**: Start new discussion thread

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- GitHub contributors page
- Release notes

Thank you for contributing to CommPy! 🎉

