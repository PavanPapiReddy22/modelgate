# Contributing to modelgate

## Setup

```bash
git clone https://github.com/PavanPapiReddy22/modelgate.git
cd modelgate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

The OpenAI adapter tests require an `OPENAI_API_KEY` environment variable (or a `.env` file at the repo root).

## Adding a new provider

1. Create `src/modelgate/providers/<name>.py` subclassing `BaseProvider` from [src/modelgate/providers/base.py](src/modelgate/providers/base.py)
2. Implement `complete()` and `stream()`
3. Register the provider in [src/modelgate/client.py](src/modelgate/client.py)
4. Add tests in `tests/test_<name>_adapter.py`

## Submitting changes

- Open a PR against `main`
- Make sure all tests pass
- Keep changes focused — one feature or fix per PR
