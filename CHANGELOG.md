# Changelog

## Repository Organization - Nov 10, 2025

### Folder Structure

- Created `paper/` folder for research paper
  - Added `CITATION.md` with proper attribution to Alexia Jolicoeur-Martineau
  - Moved PDF to `paper/Less is More- Recursive Reasoning with Tiny Networks .pdf`

- Reorganized `results/` folder
  - `results/data/` - CSV and JSON experiment results
  - `results/figures/` - PNG visualizations
  - Moved from root to maintain clean project structure

- Improved `docs/` structure
  - Moved `setup.md` to `docs/SETUP.md`
  - Moved `github_setup.md` to `docs/GITHUB_SETUP.md`

### Code Quality Improvements

- Removed all emojis from codebase
  - Documentation files (README, guides, etc.)
  - Python scripts and modules
  - Shell scripts

- Reduced comment verbosity
  - Removed excessive docstrings
  - Simplified inline comments
  - Made code more concise and human-like
  - Kept essential documentation only

- Updated `.gitignore`
  - Excluded venv/, checkpoints/, wandb/
  - Excluded large binary files
  - Excluded Python cache files

### Git History

Total commits: 42 meaningful, granular commits

Key commits:
- init gitignore
- docs structure fix
- project checklist
- readme overview
- requirements base
- package setup
- model core
- trainer loop ema
- data utils
- experiment runner
- results analyzer
- paper folder citation
- results data snapshot
- clean doc emojis
- clean code comments
- relocate results files

### File Organization

```
trm-recursion-study/
├── paper/                    # Research paper and citation
├── docs/                     # All documentation
├── src/                      # Core implementation
├── experiments/              # Experiment framework
├── config/                   # YAML configurations
├── results/
│   ├── data/                # Experiment results
│   └── figures/             # Visualizations
├── Root files               # Setup, verification, quick start
└── Git files               # .gitignore, README, etc.
```

### Credits

This implementation is based on "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau, Samsung SAIL Montreal, 2025 (arXiv:2510.04871).

See `paper/CITATION.md` for full citation details.

