#!/usr/bin/env bash
set -e
python -V
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
