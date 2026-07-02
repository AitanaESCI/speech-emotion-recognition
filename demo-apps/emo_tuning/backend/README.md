# EmoTuning Backend

FastAPI backend server for coordinate tuning.

## Run Development Server

To run the backend with hot-reload enabled (port 8000):
```bash
uv run uvicorn main:app --port 8000 --reload
```

## Run Production Server

To run the backend without reload (port 8000):
```bash
uv run uvicorn main:app --port 8000
```

## Run Tests

To run the automated test suite:
```bash
uv run python test_backend.py
```
