#!/usr/bin/env bash
# Generated with AI assistance (Claude Sonnet, 2026-11-10).
# Prompt: "Write report/build.sh (tectonic build of main.tex to F20DL-Group16-report.pdf) and report/check_report.py (page limit via bodyend label, no hand-typed result numbers, every DISCLOSURES.csv item present, no undefined refs, figures exist, forbidden-claim grep)."
set -euo pipefail
cd "$(dirname "$0")"
tectonic --keep-intermediates --keep-logs main.tex > build.log 2>&1 || { echo "tectonic failed; see report/build.log" >&2; tail -20 build.log >&2; exit 1; }
cp main.pdf F20DL-Group16-report.pdf
echo "built report/F20DL-Group16-report.pdf"
