#!/usr/bin/env bash
# Exit on error
set -o errexit

pip install -r requirements.txt
python dataanalyzer/manage.py collectstatic --no-input
python dataanalyzer/manage.py migrate

