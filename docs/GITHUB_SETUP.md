# GitHub Repository Setup Guide

Step-by-step instructions to create and populate your TRM repository on GitHub.

## Step 1: Create Repository on GitHub

1. Go to [github.com](https://github.com)
2. Click the "+" icon → "New repository"
3. Fill in details:
   - **Repository name**: `trm-recursion-study`
   - **Description**: "Empirical study of recursion vs scale in Tiny Recursive Models (TRM)"
   - **Visibility**: Public (recommended) or Private
   - **Initialize**: Do NOT check any boxes (we have files already)
4. Click "Create repository"

## Step 2: Initialize Local Repository

```bash
# Navigate to your project directory
cd /path/to/trm-recursion-study

# Initialize git
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: TRM recursion vs scale study

- Complete TRM implementation (src/)
- Experiment runner with configuration
- Documentation and setup scripts
- Based on paper by Jolicoeur-Martineau (2025)"
```

## Step 3: Connect to GitHub

```bash
# Add remote (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/trm-recursion-study.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Add Your Results (Optional)

If you have experimental results to include:

```bash
# Add your CSV and JSON results
cp /path/to/experiment_results.csv results/
cp /path/to/detailed_results.json results/
cp /path/to/*.png results/figures/

# Commit results
git add results/
git commit -m "Add experimental results

- Sudoku-Extreme: 87.4% accuracy with 2L, n=6
- Complete experiment matrix (1L, 2L, 4L)
- Recursion depth analysis (n=2,4,6,8)
- Visualization figures"

git push
```

## Step 5: Configure Repository Settings

### Add Topics (for discoverability)
1. Go to repository on GitHub
2. Click "" next to "About"
3. Add topics:
   - `deep-learning`
   - `pytorch`
   - `recursive-neural-networks`
   - `reasoning`
   - `neural-networks`
   - `sudoku`
   - `arc-agi`
   - `parameter-efficiency`

### Enable GitHub Pages (optional)
1. Settings → Pages
2. Source: Deploy from branch
3. Branch: main, folder: /docs
4. Save

### Add Collaborators (if needed)
1. Settings → Collaborators
2. Add people by username

## Step 6: Create README Badges

Add these to the top of README.md for a professional look:

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
```

## Step 7: Create Releases (when ready)

1. Go to "Releases" → "Create a new release"
2. Tag version: `v1.0.0`
3. Release title: "TRM Recursion Study v1.0 - Initial Release"
4. Description:
```markdown
# TRM Recursion Study v1.0

Initial release of Tiny Recursive Model experiments.

## Features
- Complete TRM implementation
- Systematic recursion vs scale experiments
- Sudoku-Extreme benchmark
- Comprehensive documentation

## Key Results
- 87.4% test accuracy on Sudoku-Extreme
- 2-layer networks outperform 4-layer
- Optimal recursion depth: n=6

## Installation
See [SETUP.md](docs/SETUP.md) for complete instructions.
```

## Step 8: Share Your Repository

### Add to Paper with Code
1. Go to [paperswithcode.com](https://paperswithcode.com)
2. Search for "Less is More: Recursive Reasoning"
3. Click "Add Code"
4. Submit your repository

### Share on Social Media
```
 New open-source implementation of Tiny Recursive Models (TRM)!

 Key findings:
- 87.4% accuracy with only 7M parameters
- Smaller networks + deep recursion > large networks
- Full replication & extension experiments

 https://github.com/yourusername/trm-recursion-study

#DeepLearning #PyTorch #RecursiveNeuralNetworks
```

## File Checklist

Before pushing, ensure you have:

- [ ] README.md (main documentation)
- [ ] LICENSE (MIT)
- [ ] requirements.txt (dependencies)
- [ ] .gitignore (ignore patterns)
- [ ] setup.sh (installation script)
- [ ] src/model.py (TRM implementation)
- [ ] src/trainer.py (training loop)
- [ ] src/__init__.py (package init)
- [ ] experiments/run_experiments.py (experiment runner)
- [ ] config/sudoku_config.yaml (full config)
- [ ] config/quick_test.yaml (test config)
- [ ] docs/SETUP.md (setup guide)
- [ ] quick_start.py (quick test)
- [ ] results/.gitkeep (preserve directory)

## Common Git Commands

```bash
# Check status
git status

# View changes
git diff

# Add specific files
git add src/model.py

# Commit with message
git commit -m "Add feature X"

# Push to GitHub
git push

# Pull latest changes
git pull

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Merge branch
git merge feature-name
```

## Troubleshooting

### Authentication Issues
If using HTTPS, you may need a Personal Access Token:
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

Or switch to SSH:
```bash
git remote set-url origin git@github.com:yourusername/trm-recursion-study.git
```

### Large Files
If you have large checkpoint files (>100MB):
```bash
# Install git-lfs
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes
git commit -m "Add git-lfs tracking"
```

### Merge Conflicts
```bash
# If you get merge conflicts
git status  # See conflicted files
# Edit files to resolve conflicts
git add <resolved-files>
git commit -m "Resolve merge conflicts"
```

## Maintenance

### Keep Dependencies Updated
```bash
# Update requirements.txt
pip list --outdated
pip install --upgrade <package>
pip freeze > requirements.txt

# Commit updates
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### Add New Experiments
```bash
# Create new branch
git checkout -b experiment-maze-hard

# Add new code/configs
git add config/maze_config.yaml
git commit -m "Add Maze-Hard experiments"

# Push and create pull request
git push -u origin experiment-maze-hard
```

## Next Steps

After setup:
1.  Verify repository on GitHub
2.  Test clone on another machine
3.  Run quick_start.py to verify
4.  Add to your CV/portfolio
5.  Share with community

---

**Need help?** Open an issue on GitHub or refer to [Git documentation](https://git-scm.com/doc).
