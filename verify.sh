#!/bin/bash

# TRM Repository Verification Script
# Checks that all required files are present

echo "============================================"
echo "TRM Repository Verification"
echo "============================================"
echo ""

error_count=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo "✓ $1"
    else
        echo "✗ $1 (MISSING)"
        ((error_count++))
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo "✓ $1/"
    else
        echo "✗ $1/ (MISSING)"
        ((error_count++))
    fi
}

echo "Checking root files..."
check_file "README.md"
check_file "LICENSE"
check_file "requirements.txt"
check_file ".gitignore"
check_file "setup.sh"
check_file "quick_start.py"
check_file "CHECKLIST.md"
check_file "PACKAGE_README.md"

echo ""
echo "Checking directories..."
check_dir "src"
check_dir "config"
check_dir "experiments"
check_dir "docs"
check_dir "results"

echo ""
echo "Checking src/ files..."
check_file "src/__init__.py"
check_file "src/model.py"
check_file "src/trainer.py"

echo ""
echo "Checking config/ files..."
check_file "config/sudoku_config.yaml"
check_file "config/quick_test.yaml"

echo ""
echo "Checking experiments/ files..."
check_file "experiments/run_experiments.py"

echo ""
echo "Checking docs/ files..."
check_file "docs/SETUP.md"
check_file "docs/GITHUB_SETUP.md"

echo ""
echo "============================================"
if [ $error_count -eq 0 ]; then
    echo "✓ All files present! Repository is complete."
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "1. Run: python quick_start.py"
    echo "2. Or: ./setup.sh (if not done already)"
    exit 0
else
    echo "✗ $error_count file(s) missing!"
    echo "============================================"
    echo ""
    echo "Please ensure all files are in the correct locations."
    exit 1
fi
