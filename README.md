# Local Academic Paper Chatbot

A local-first academic paper analysis chatbot for Windows x64 and Python 3.12 x64.

## Development setup

From the repository root, create a Python 3.12 virtual environment and install the lock
tool and build prerequisites:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip pip-tools
```

The committed lock is ready to use. If declared dependencies change, regenerate it before
installing:

```powershell
.venv\Scripts\pip-compile.exe --extra dev --generate-hashes --output-file requirements.lock pyproject.toml
```

Install the locked environment and project package:

```powershell
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Run the quality baseline:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_package.py -q
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pip check
```
