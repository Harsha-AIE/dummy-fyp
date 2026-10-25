# Run from project root in PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_trf
# Neo4j Desktop / Docker must be installed and running separately.
# PaddleOCR on first run will download model weights (requires internet once).
