# CommPy Documentation Index

Welcome to CommPy documentation! This guide helps you find what you need.

## Quick Navigation

### For New Users
1. **Start here**: [Getting Started Guide](GETTING_STARTED.md)
2. **Learn concepts**: [User Guide](USER_GUIDE.md)
3. **See examples**: `examples/` directory in repository
4. **Ask questions**: GitHub Discussions

### For Developers
1. **API Reference**: [API.md](API.md)
2. **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)
3. **Module structure**: Check `src/commpy/` directory
4. **Testing**: `tests/` directory

### For Specific Tasks

| Task | Document | Section |
|------|----------|---------|
| Install CommPy | [GETTING_STARTED.md](GETTING_STARTED.md) | Installation |
| Learn modulation | [USER_GUIDE.md](USER_GUIDE.md) | Modulation & Demodulation |
| Simulate channels | [USER_GUIDE.md](USER_GUIDE.md) | Channel Models |
| Generate waveforms | [USER_GUIDE.md](USER_GUIDE.md) | Waveform Generation |
| Find function details | [API.md](API.md) | Use Ctrl+F to search |
| Measure BER | [GETTING_STARTED.md](GETTING_STARTED.md) | Tutorial 2 |
| Contribute code | [CONTRIBUTING.md](../CONTRIBUTING.md) | Development Guidelines |
| Error correction (FEC) | [GETTING_STARTED.md](GETTING_STARTED.md) | Tutorials 7-8 |
| Generic M-QAM/M-PSK + soft demod | [GETTING_STARTED.md](GETTING_STARTED.md) | Tutorial 9 |
| OFDM | [GETTING_STARTED.md](GETTING_STARTED.md) | Tutorial 10 |
| Source coding / channel capacity | [GETTING_STARTED.md](GETTING_STARTED.md) | Tutorial 11 |
| Queuing theory | [GETTING_STARTED.md](GETTING_STARTED.md) | Tutorial 12 |
| Runnable end-to-end scripts | `examples/` directory | one file per feature + capstone |

---

## Documentation Files

### 📖 Main Documentation

#### [README.md](../README.md)
**Purpose:** Overview and quick start
**Contains:**
- Feature summary
- Installation instructions
- Quick start examples
- Module structure overview
- Basic API reference
- Example code snippets

**When to read:**
- First time using CommPy
- Want overview of capabilities
- Looking for installation help

---

#### [GETTING_STARTED.md](GETTING_STARTED.md)
**Purpose:** Beginner-friendly tutorials
**Contains:**
- Installation verification
- Basic concepts explained
- 6 hands-on tutorials
- Common patterns
- Troubleshooting guide
- Quick reference cheat sheet

**When to read:**
- New to digital communications
- Want step-by-step examples
- Need to troubleshoot issues
- Looking for quick reference

**Tutorials included:**
1. Simple modulation & demodulation
2. Bit error rate (BER) simulation
3. Channel comparison
4. Different modulation schemes
5. IQ waveform generation
6. Reproducible results with RNG

---

#### [API.md](API.md)
**Purpose:** Complete API documentation
**Contains:**
- All public functions/classes
- Parameter descriptions
- Return value specs
- Code examples
- Use cases
- Type hints

**Organized into sections:**
- Modulation (BPSK, QPSK, ASK, PSK-8, OOK)
- Channels (BSC, BEC, AWGN)
- Information Theory (Shannon entropy)
- Waveforms (IQWaveform)
- Utilities (Prime field, math functions)

**When to read:**
- Looking up specific function
- Need parameter details
- Want type information
- Seeing code examples

---

#### [USER_GUIDE.md](USER_GUIDE.md)
**Purpose:** Comprehensive conceptual guide
**Contains:**
- Theory explanations
- Visual constellation diagrams
- When to use each scheme
- Detailed parameter guidance
- Practical applications
- Best practices
- Real-world examples

**Sections:**
1. Introduction
2. Modulation & Demodulation (detailed explanations)
3. Channel Models (theory + code)
4. Information Theory
5. Waveform Generation
6. Practical Applications
7. Best Practices

**When to read:**
- Want to understand concepts deeply
- Choosing which modulation to use
- Need guidance on parameters
- Looking for practical examples
- Want performance guidance

---

#### [CONTRIBUTING.md](../CONTRIBUTING.md)
**Purpose:** Guide for contributors
**Contains:**
- Setup instructions
- Code style guidelines
- Documentation standards
- Testing requirements
- PR process
- Code review guidelines
- Documentation examples

**When to read:**
- Want to contribute to CommPy
- Fixing a bug
- Adding a feature
- Writing documentation
- Improving the library

---

## Learning Paths

### Path 1: "I Want to Get Started Quickly"
1. Read: [README.md](../README.md) - Overview (5 min)
2. Install CommPy (2 min)
3. Read: [GETTING_STARTED.md](GETTING_STARTED.md) - Quick Start (10 min)
4. Run: Tutorial 1 - Basic modulation (10 min)
5. Try: Run Tutorial 2 - BER simulation (15 min)

**Total: ~40 minutes**

---

### Path 2: "I Need to Understand Digital Communications"
1. Read: [USER_GUIDE.md](USER_GUIDE.md) - Introduction (10 min)
2. Read: [USER_GUIDE.md](USER_GUIDE.md) - Modulation section (30 min)
3. Run: [GETTING_STARTED.md](GETTING_STARTED.md) - Tutorials 1-3 (30 min)
4. Read: [USER_GUIDE.md](USER_GUIDE.md) - Channel Models section (20 min)
5. Run: [GETTING_STARTED.md](GETTING_STARTED.md) - Tutorial 2 (20 min)

**Total: ~2 hours**

---

### Path 3: "I Want to Use CommPy for Research"
1. Read: [README.md](../README.md) (5 min)
2. Read: [USER_GUIDE.md](USER_GUIDE.md) - All sections (2 hours)
3. Read: [API.md](API.md) - Reference your specific modules (30 min)
4. Create simulation script using patterns from [USER_GUIDE.md](USER_GUIDE.md) (1 hour)
5. Run and refine simulation (varies)

**Total: ~4 hours + simulation time**

---

### Path 4: "I Want to Contribute Code"
1. Read: [CONTRIBUTING.md](../CONTRIBUTING.md) - Full document (30 min)
2. Set up development environment (15 min)
3. Read: [README.md](../README.md) - Module structure (10 min)
4. Explore: `/src/commpy/` source code (30 min)
5. Start contributing! (varies)

**Total: ~1.5 hours + development time**

---

## Document Quick Reference

### By Topic

**Modulation:**
- Overview: [README.md](../README.md#modulation)
- Tutorial: [GETTING_STARTED.md](GETTING_STARTED.md) Tutorial 1
- Deep dive: [USER_GUIDE.md](USER_GUIDE.md) - Modulation section
- API: [API.md](API.md) - Modulation classes

**Channels:**
- Overview: [README.md](../README.md#channel-simulation)
- Tutorial: [GETTING_STARTED.md](GETTING_STARTED.md) Tutorial 3
- Deep dive: [USER_GUIDE.md](USER_GUIDE.md) - Channel Models section
- API: [API.md](API.md) - Channels class

**BER Simulation:**
- Tutorial: [GETTING_STARTED.md](GETTING_STARTED.md) Tutorial 2
- Practical example: [USER_GUIDE.md](USER_GUIDE.md) - Application 1
- Reference: [API.md](API.md) - Search for `awgn`

**Waveforms:**
- Example: [GETTING_STARTED.md](GETTING_STARTED.md) Tutorial 5
- Deep dive: [USER_GUIDE.md](USER_GUIDE.md) - Waveform Generation
- API: [API.md](API.md) - IQWaveform class

**Best Practices:**
- Quick tips: [GETTING_STARTED.md](GETTING_STARTED.md) - Common Patterns
- Detailed guide: [USER_GUIDE.md](USER_GUIDE.md) - Best Practices
- Code standards: [CONTRIBUTING.md](../CONTRIBUTING.md) - Code Style

**Type Hints & Docstrings:**
- Examples: [CONTRIBUTING.md](../CONTRIBUTING.md) - Documentation section
- Code style: [CONTRIBUTING.md](../CONTRIBUTING.md) - Code Style section

---

## Using the Documentation

### Finding Information

**Using GitHub:**
- Search docs/ folder for keywords
- Use repository search feature
- Check README.md for quick answers

**Using Text Editor:**
- Open all docs in VS Code
- Use Ctrl+Shift+F for full-text search
- Open specific section links

**Common Searches:**
```
"how to modulate" → GETTING_STARTED.md Tutorial 1
"SNR" → USER_GUIDE.md, API.md
"example" → GETTING_STARTED.md, README.md
"parameters" → API.md
"reproduce" → GETTING_STARTED.md Tutorial 6
```

---

## Getting Help

### Documentation Not Clear?
1. Check [GETTING_STARTED.md](GETTING_STARTED.md) - Troubleshooting section
2. Search [API.md](API.md) for specific function details
3. Look at example in [README.md](../README.md)
4. Open GitHub issue with question

### Want to Report a Bug?
1. Check [GETTING_STARTED.md](GETTING_STARTED.md) - Troubleshooting
2. Search existing GitHub issues
3. Create new issue with details from [CONTRIBUTING.md](../CONTRIBUTING.md) - Reporting section

### Want to Contribute?
Start with [CONTRIBUTING.md](../CONTRIBUTING.md) - Getting Started section

---

## Documentation Maintenance

Last Updated: March 2026

**Covered CommPy Version:** 0.1.2+

**Sections Status:**
- ✅ Installation - Current
- ✅ Core API - Current  
- ✅ Examples - Current
- ✅ Tutorials - Current
- ✅ Advanced topics - Current

---

## Feedback

Have suggestions for improving documentation?
- Submit issues on GitHub
- Create pull requests with improvements
- Provide feedback in discussions

---

## Related Resources

### External Resources
- **NumPy**: https://numpy.org/doc/
- **Matplotlib**: https://matplotlib.org/
- **Digital Communications**: [Your recommended textbooks]

### Example Projects
Examples can be found in the `examples/` directory of the repository.

