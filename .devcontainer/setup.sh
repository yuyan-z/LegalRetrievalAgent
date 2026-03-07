#!/bin/bash
#!/usr/bin/env bash
set -e

echo ">>> Setting up development environment..."

python -m pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
  echo ">>> Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
  echo ">>> Installing project in editable mode..."
  pip install -e .
fi

echo ">>> Installing development tools..."
pip install pytest ipykernel

python -m ipykernel install --user --name lexcite --display-name "Python (lexcite)" || true

echo ">>> Setup completed."
