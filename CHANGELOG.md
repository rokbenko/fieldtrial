# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0,
minor releases may contain breaking changes.

## [Unreleased]

## [0.1.0a1] - 2026-10-01

First alpha: the statistics core and the calculator commands.

### Added

- `fieldtrial.stats`, the statistics core (numpy and scipy only):
  - one-arm intervals: Wilson (default), Clopper–Pearson, Jeffreys, Agresti–Coull, and an
    exact binomial test against a threshold
  - two independent arms: Newcombe difference CI, Boschloo exact test (primary), Fisher's
    exact test and the conditional odds ratio
  - paired designs: exact McNemar, Tango score CI, Cochran's Q with pairwise McNemar
  - multiplicity adjustments: Holm, Bonferroni, Benjamini–Hochberg
  - planning: sample size, power and minimum detectable effect (pooled-z, Fleiss
    continuity-corrected, arcsine; unequal allocation), exact power for Boschloo and
    McNemar, and a seeded simulation cross-check
  - Bayesian summaries (descriptive only): P(p_B > p_A) and credible intervals
- Calculator commands `fieldtrial ci`, `compare`, `paired`, `power`, `mde` and `adjust`,
  each with `--json` output.
- Documentation: a statistics reference page per method, a command-line page and the API
  reference.
- Project skeleton: packaging, `fieldtrial --version`, CI and the documentation site.

[Unreleased]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/rokbenko/fieldtrial/releases/tag/v0.1.0a1
