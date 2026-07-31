# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows semantic versioning for software releases.

## [Unreleased]

### Added

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

## [4.2.0] - 2026-07-31

### Added

- Provenance-Robust Optimization evidence selection.
- Strict Ollama retry and paper-validity controls.
- Claim-level citation coverage metrics.
- Source-group evidence-cut evaluation.
- Regression tests for PRO selection and corrected metrics.
