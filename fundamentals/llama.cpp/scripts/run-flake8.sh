#!/bin/bash
# pip install "flake8>=6.0.0"

source venv/bin/activate

flake8 $1

deactivate
