# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows semantic versioning for software releases.

## [Unreleased]

## [4.2.1] - 2026-08-01

### Added

- Trace-derived regression tests for VITA decision routing, citation normalization, precedence, residual-risk handling, and PRO selection.
- `--task-offset` / `TASK_OFFSET` support for evaluating a held-out benchmark range.
- Separate advisory findings in VITA and verification-envelope certificates.

### Changed

- Calibrated VITA decision semantics to distinguish categorical prohibitions, satisfiable controls, and mandatory human review.
- Stopped moderate residual search mass from automatically forcing every VITA result to `needs_review`.
- Restricted precedence resolution to explicit conflicts and genuine alternatives from the same policy family.
- Required semantic relevance before inline citations can strengthen claim support.
- Made VITA-PRO inspect the complete candidate universe and favor decisive current-version evidence over stale or unrelated diversity.
- Added task-aware generation guidance and grounded citation repair for VITA-family methods.

## [4.2.0] - 2026-07-31

### Added

- Provenance-Robust Optimization evidence selection.
- Strict Ollama retry and paper-validity controls.
- Claim-level citation coverage metrics.
- Source-group evidence-cut evaluation.
- Regression tests for PRO selection and corrected metrics.
- Continuous integration for Python 3.11 and 3.12.
- Container-build validation.
- Contributor, citation, security, and repository-maintenance documentation.

### Changed

- Reworked the README around installation, architecture, reproducibility, outputs, and research limitations.
- Updated architecture and benchmark documentation for VeriWeave-VITA-PRO.
- Restricted package discovery to the maintained `veriweave` package.
- Consolidated environment configuration into `.env.example`.
- Improved Docker and ignore-file hygiene.

### Removed

- Legacy ClaimGraph source package and benchmark copy.
- Generated policy-graph exports.
- Redundant placeholder and dependency files.
