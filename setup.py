"""
Setup script for TRM Recursion Study package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="trm-recursion-study",
    version="0.1.0",
    author="Ilakian Puvanendra",
    description="Tiny Recursive Models: Recursion vs Scale Study",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trm-recursion-study",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.20.0",
            "notebook>=6.5.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trm-quick-start=quick_start:main",
            "trm-run-experiments=experiments.run_experiments:main",
            "trm-analyze-results=experiments.analyze_results:main",
        ],
    },
)
