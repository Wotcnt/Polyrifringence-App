![CI](https://github.com/Wotcnt/Polyrifringence-App/actions/workflows/ci.yml/badge.svg)

# Polyrifringence App (Streamlit)

Streamlit web interface for interacting with the bundled `poly_engine` package.

This repository is the **application layer**: UI, run orchestration, and diagnostics. It is not a substitute for the canonical documentation suite.

## Canon and scope
- **Strict Canon Mode** is enabled by default in the UI and locks:
  - ΔΩ to 1/7 (0.142857)
  - λ_cycle to the Appendix C definition (`ratio_abs`)
  - Δt_cycle to the empirical envelope (0.00035–0.00040 s)
  - the ΔΩ recursion depth bound via `enforce_delta_omega_bound`
- When Strict Canon Mode is disabled, the app operates in exploratory diagnostics mode and must be treated as non-canonical.

See:
- `README_UNIFIED_SUITE.md`
- `CANON_MAP.md`

## Run locally (Windows)
```powershell
./scripts/run_local.ps1
```

## Run locally (macOS/Linux)
```bash
bash ./scripts/run_local.sh
```

## Run tests
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
pytest -q
```

## Run with Docker
```bash
docker build -t polyrifringence-app .
docker run --rm -p 8501:8501 polyrifringence-app
```
