# Local Academic Paper Chatbot

A local-first academic paper analysis chatbot for Windows x64 and Python 3.12 x64.

## Development setup

Create a Python 3.12 virtual environment, then install the hash-locked dependencies and
the project package:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Run the quality baseline:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m mypy src
```
