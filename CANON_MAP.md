# Canon Map — Polyrifringence Engine Web App

Canonical references shipped with this app:

- `README.md`
- `THEORY.md`
- `MATH_MODEL(Render).md`

## UI feature mapping

### Strict Canon Mode
- Enforces the ΔΩ recursion depth bound (6–7 cycles).
- Locks λ_cycle to the Appendix C definition (`ratio_abs`).
- Locks Δt_cycle to the empirical envelope (0.00035–0.00040 s) and reports Λ̸ in both cycles and seconds.

### Verification Mode
- Runs multi-seed trials and bounded perturbations for falsifiable reproducibility checks.

### CSO Observer Relay (diagnostic)
- Implements the relay gate as a diagnostic-only mechanism that may modulate only a damping coefficient (phase restoration factor).
- Disengages automatically once the ΔΩ bound is hit.

### Gemline Mode
- Loads the unified gem registry and exposes deterministic presets and provenance metadata.

### Child-Beam Cascade
- Exercises branching recursion and provides depthwise aggregate metrics.

### Documentation exports
- Every export includes mode flags (Strict Canon, λ_mode, Δt_cycle, termination reason) to prevent frame errors.
