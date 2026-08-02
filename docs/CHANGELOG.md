# CommPy Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

#### MIMO — `commpy` (multiple-antenna support)
- `rayleigh_channel_matrix` / `mimo_awgn` / `mimo_noise_variance`: an i.i.d.
  Rayleigh MIMO channel matrix and a flat-fading `y = H x + n` channel.
- `alamouti_encode` / `alamouti_decode`: rate-1 Alamouti space-time block coding
  (2 transmit antennas, any number of receive antennas) for transmit diversity.
- Spatial-multiplexing detectors: `zf_detector`, `mmse_detector`, `ml_detector`
  (exhaustive maximum-likelihood), and `kbest_detector` (a K-best sphere decoder,
  equal to ML for a large enough list). K-best is cross-validated against ML.
- `mimo_capacity` / `ergodic_mimo_capacity`: deterministic and fading-averaged
  MIMO channel capacity.

#### AI-for-wireless (optional PyTorch layer) — `commpy.ml`
- A new optional subpackage (`pip install "commpy[ml]"`), deliberately **not**
  imported by `commpy/__init__` so the base install stays NumPy/SciPy-only and
  `import commpy` never pulls in PyTorch. Import it explicitly as `commpy.ml`.
- `awgn` / `normalize_power`: a differentiable complex AWGN channel and transmit
  power constraint (autograd-friendly `(..., 2)` real/imag tensors).
- `Autoencoder` (+ `train_autoencoder`, `block_error_rate`): an end-to-end
  learned transmitter/receiver that learns a constellation and its detector by
  training through the differentiable channel.
- `NeuralDemapper` (+ `train_demapper`): a learned soft demapper for a
  `Modulator`'s constellation, exposing the same `soft_demodulate` interface.
- `NeuralMinSumDecoder` (+ `train_neural_min_sum`): an LDPC belief-propagation
  decoder unrolled into a trainable weighted-min-sum network, reusing an
  existing `LDPCCode`'s Tanner graph (unit weights reproduce classical min-sum).
- CI gains a `test-ml-extra` job that installs CPU-only PyTorch and runs the
  `commpy.ml` tests and `mypy` with torch present.

#### MCP server (optional) — `commpy-mcp`
- A Model Context Protocol server (`pip install "commpy[mcp]"`, then run
  `commpy-mcp`) exposing CommPy to AI agents: tools to list capabilities,
  compute AWGN/BSC channel capacity, and run uncoded or coded
  (LDPC/polar/turbo) BER sweeps. The tool logic is plain, importable functions
  (`commpy.mcp_server`), fully tested without the `mcp` package.

#### Documentation & tooling
- A **Gallery** docs page with generated coding-gain and learned-constellation
  figures.
- A **benchmark suite** (`benchmarks/`, pytest-benchmark) for the LDPC, polar,
  turbo, and Viterbi decoders and QAM (run with
  `pytest benchmarks/ --benchmark-only --no-cov`); bare `pytest` runs the
  correctness suite only (`testpaths = ["tests"]`).

## [1.1.0] - 2026-07-26

Adds the three modern standard forward-error-correcting codes — **LDPC**,
**polar**, and **turbo** — that underpin 5G-NR/LTE/Wi-Fi/DVB, all consuming the
existing soft-decision LLRs from `Modulator.soft_demodulate`. No breaking
changes to the public API (`commpy.*`).

### Added

#### Channel coding (FEC) — `_channelCoding/`
- **LDPC codes** (`ldpc/`): `LDPCCode` with belief-propagation decoding
  (sum-product and normalized min-sum, fully vectorized). Constructors for
  Gallager's regular ensemble (`from_gallager`) and quasi-cyclic protograph
  lifting (`from_base_graph`), a systematic GF(2) generator that handles
  rank-deficient parity-check matrices, and a ready rate-1/2 QC code
  (`standards.rate_one_half_ldpc`).
- **Polar codes** (`polar/`): `PolarCode` with successive-cancellation (SC) and
  CRC-aided SC-*list* (CA-SCL) decoding, reusing the `CRC` class for the
  list-selection check. Frozen-set construction by Bhattacharyya parameters or
  Gaussian-approximation density evolution. The list decoder accumulates exact
  log-domain path metrics, verified equal to maximum-likelihood decoding in the
  test suite.
- **Turbo codes** (`turbo/`): `TurboCode`, a rate-1/3 parallel-concatenated code
  with recursive-systematic constituent encoders (`RSCTrellis`) and iterative
  log-MAP (BCJR) decoding that exchanges extrinsic information through an
  interleaver.

#### Link-level simulation — `_simulation/`
- `simulate_coded_ber`: Monte-Carlo BER sweep for any soft-input block code
  (modulate → channel → `soft_demodulate` → decode), the coded counterpart of
  `simulate_ber`.

### Changed
- `plot_waterfall` now clamps confidence-interval error bars at zero, so the
  very small error rates typical of coded curves no longer trigger a matplotlib
  "negative yerr" error.

### Examples
- `ldpc_coding_gain_demo.py`, `polar_scl_demo.py`, `turbo_coding_gain_demo.py`
  — coded-vs-uncoded BER waterfalls demonstrating each code's coding gain.

## [1.0.0] - 2026-07-21

A major expansion from a small modulation/channel-model library into a
comprehensive communications-engineering toolkit. No breaking changes to the
public API (`commpy.*`); three previously-broken imports left over from an
earlier package rename are fixed as part of this release.

### Added

#### Channel coding (FEC) — `_channelCoding/`
- **Block codes** (`block/`): `CRC` (CRC-8/CRC-16-XMODEM/CRC-32 presets, plus a custom `CRCConfig`), `HammingCode`, generic `CyclicCode`, `BCHCode`, `ReedSolomonCode` (error decoding via Berlekamp-Massey/Chien search/Forney, plus a separate erasures-only decoder correcting up to `n-k` erasures).
- **Convolutional codes** (`convolutional/`): `Trellis`, `ConvolutionalEncoder`, `viterbi_decode` (hard- and soft-decision, zero-tail termination, optional Numba JIT acceleration on the add-compare-select trellis traversal).
- **Interleaving** (`interleaving/`): `BlockInterleaver`, `ConvolutionalInterleaver` (Ramsey type-II).

#### Galois field arithmetic — `_fields/`
- `PrimeField` (GF(p)) generalized onto a new `FiniteField` abstract base.
- `GF2m`: binary extension field GF(2^m) via log/antilog tables, fully vectorized, used by BCH/Reed-Solomon.
- Polynomial arithmetic over any `FiniteField` (`poly_add`/`poly_mul`/`poly_divmod`/`poly_eval`).

#### Digital modulation & PHY — `_modulation/`
- Generic, Gray-coded `Modulator` base class with `MPSKModulator`, `MQAMModulator` (square constellations), `MPAMModulator`, and generic soft-decision (LLR) demodulation.
- Original per-scheme classes (`OOK_Modulator`, `BPSK_Modulator`, `ASK_2_Modulator`, `ASK_4_Modulator`, `QPSK_Modulator`, `PSK_8_Modulator`) kept for backward compatibility; the `QPSK_Modulator` list-vs-array bug is fixed.
- `raised_cosine_filter` / `root_raised_cosine_filter`, drop-in compatible with `IQWaveform`'s `pulse_shape` parameter.
- `zf_equalizer` / `mmse_equalizer` linear FIR channel equalizers.
- `gardner_timing_error`, `estimate_cfo_mth_power`, `costas_loop_bpsk` synchronization primitives.

#### OFDM — `_ofdm/`
- `OFDMModulator` / `OFDMDemodulator` with configurable active subcarriers and cyclic prefix.
- `papr`, `papr_db`, `papr_ccdf` for peak-to-average power ratio analysis.

#### Information theory — `_informationTheory/`
- `binary_entropy`, `mutual_information`, `channel_capacity_bsc`, `channel_capacity_awgn`, `channel_capacity_dmc` (Blahut-Arimoto).
- `huffman_codes`/`huffman_encode`/`huffman_decode`, `arithmetic_encode`/`arithmetic_decode` (exact-rational arithmetic coding).
- `rate_distortion_binary` (closed-form binary rate-distortion function).

#### Networking — `_networking/`
- `MM1Queue`, `MM1KQueue`, `MMcQueue`: closed-form M/M/1-family queuing performance models.

#### Infrastructure
- CI (`.github/workflows/ci.yml`) now runs `ruff`, `mypy --strict`, and `pytest` on every push/PR across Python 3.10-3.12, including a dedicated job for the numba-absent fallback path.
- `scipy` is now a hard dependency (FFT for OFDM, `solve_toeplitz` for MMSE equalization, `erfc` for BER reference curves in tests). `numba` is an optional `commpy[fast]` extra for JIT-accelerated Viterbi decoding.
- Test suite grew from 3 tests to 300+, including exhaustive brute-force cross-validation for BCH/Reed-Solomon decoding and Viterbi-vs-maximum-likelihood-search checks.

### Fixed
- Three stale `CommPy` → `commpy` imports (from an earlier package rename) that broke `PrimeField`, the test suite, and `IQWaveform`'s demo block.

### Changed
- Internal package layout reorganized (`_channelCoding/` split into `_channels/`, `_fields/`, `_modulation/`, plus the new `_channelCoding/{block,convolutional,interleaving}/`). This only affects private (`_`-prefixed) submodules; the public `commpy.*` API is unchanged.

---

## [0.1.2] - Previous Release

### Added

#### Documentation
- **Comprehensive API Reference** (`docs/API.md`)
  - Complete documentation for all modulation classes
  - Channel model details and usage examples
  - Information theory functions
  - Waveform generation guide
  - Utility functions reference

- **Getting Started Guide** (`docs/GETTING_STARTED.md`)
  - Beginner-friendly installation guide
  - 6 hands-on tutorials with code examples
  - Common patterns and best practices
  - Troubleshooting guide
  - Quick reference cheat sheet

- **User Guide** (`docs/USER_GUIDE.md`)
  - Conceptual explanations of all features
  - Visual constellation diagrams
  - Modulation scheme comparisons
  - Channel model theory and practice
  - Practical applications and examples
  - Best practices for simulations

- **Contributing Guide** (`CONTRIBUTING.md`)
  - Development environment setup
  - Code style guidelines
  - Documentation standards
  - Testing requirements
  - Pull request process
  - Code review guidelines

- **Documentation Index** (`docs/index.md`)
  - Quick navigation guide
  - Learning paths for different user types
  - Document organization
  - Topic cross-reference

- **Enhanced README.md**
  - Feature list and badges
  - Quick start examples for all major features
  - Installation instructions
  - Module structure overview
  - Complete API summary
  - Multiple practical examples

### Documentation Structure

```
CommPy/
├── README.md              # Overview and quick start
├── CONTRIBUTING.md        # Contribution guidelines
├── docs/
│   ├── index.md          # Documentation index and navigation
│   ├── API.md            # Complete API reference
│   ├── GETTING_STARTED.md # Beginner tutorials
│   ├── USER_GUIDE.md     # Comprehensive user guide
│   └── CHANGELOG.md      # This file
```

---

## [0.1.1] - Previous Release

### Added
- Core modulation classes (BPSK, QPSK, ASK, PSK-8, OOK)
- Channel models (BSC, BEC, AWGN)
- IQ waveform generation
- Shannon entropy calculation
- Utility functions (prime checking, modular inverse)
- PrimeField arithmetic

---

## [0.1.0] - Initial Release

### Added
- Initial CommPy library structure
- Basic modulation support
- Channel modeling framework
- Information theory foundation

---

## Documentation Updates Timeline

### Version 0.1.2 Documentation
- Created comprehensive API documentation
- Wrote 6 tutorials covering main features
- Documented all classes and functions
- Added multiple practical examples
- Created contribution guidelines
- Set up documentation structure

### Future Improvements

Planned documentation enhancements:
- [ ] Video tutorials
- [ ] Interactive Jupyter notebooks
- [ ] Performance benchmarks
- [ ] Extended examples gallery
- [ ] Textbook-style theory chapters
- [ ] API stability guarantees document

---

## For Users

### New in 0.1.2
If upgrading from 0.1.1, new documentation is fully backward compatible with existing APIs.

**No breaking changes** - all code from 0.1.1 works as-is in 0.1.2.

### Starting with 0.1.2
Read [docs/index.md](index.md) to find the best starting point for your needs.

---

## For Contributors

### Documentation Consistency
All documentation follows these standards:
- Clear, concise writing
- Comprehensive code examples
- Type hints throughout
- Practical use cases
- Cross-references to related topics

### Adding to Documentation
See [CONTRIBUTING.md](https://github.com/MarvinElling/CommPy/blob/main/CONTRIBUTING.md) for guidelines on:
- Code documentation style
- Example standards
- API documentation format
- Testing documentation

---

## Known Documentation Gaps

Items on the roadmap:
- Advanced filter design for pulse shaping
- Custom modulation creation
- Performance optimization guide
- Visualization best practices
- Integration with simulink/other tools

Please open an issue if you find gaps!

---

## Documentation Statistics

**Comprehensive Documentation Package:**
- **400+ KB** of documentation files
- **2000+ lines** of API documentation
- **20+ code examples** across all docs
- **6 full tutorials** with step-by-step explanations
- **4 learning paths** for different user types
- **Complete API coverage** of all public functions

---

## Questions About Changes?

- Check [docs/index.md](index.md) for navigation help
- Review specific documentation files for detailed information
- Open an issue on GitHub for clarifications
- See [CONTRIBUTING.md](https://github.com/MarvinElling/CommPy/blob/main/CONTRIBUTING.md) for dev questions

---

## Acknowledgments

Documentation created with focus on:
- Accessibility for beginners
- Depth for advanced users
- Clarity and consistency
- Practical, real-world examples

---

*Last updated: July 2026*

