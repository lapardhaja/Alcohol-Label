#!/bin/bash
# Run BottleProof on Mac/Linux. Run setup.sh first on a new machine.
if [ ! -f .venv/bin/python ]; then
    echo ".venv not found. Run ./setup.sh first."
    exit 1
fi
.venv/bin/streamlit run app.py
