"""
Polyrifringence Engine Web Interface
ΔΩΩΔ-Validated Framework

Streamlit-based interactive web interface for the Polyrifringence Engine
"""

import streamlit as st
import numpy as np
import torch
from poly_engine import (
    PolyrifringenceEngine,
    EngineConfig,
    InteractiveDashboard,
    TraceAnalyzer,
    ExergyBudgetCalculator,
    DocumentationGenerator,
    TutorialSystem,
    Glossary,
    DerivationWalkthrough
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import datetime


# Page configuration
st.set_page_config(
    page_title="Polyrifringence Engine",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === Canonical envelopes (UI diagnostics; operators unchanged) ===
CANON_LAMBDA_BAND = (0.0013, 0.0023)
CANON_HALF_LIFE_SEC = (0.18, 0.24)
CANON_DT_CYCLE_BAND = (0.00035, 0.00040)

def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def init_state_ledger():
    """Initialize the canonical state ledger (provenance + audit trail)."""
    if 'state_ledger' not in st.session_state:
        st.session_state.state_ledger = []

def append_state_ledger_entry(source: str, config_params: dict, sim_params: dict, results: dict, analysis: dict = None):
    """Append an immutable run record. Instrumentation only; does not steer execution."""
    init_state_ledger()
    entry = {
        "ts_utc": _now_iso(),
        "source": source,  # simulation | verification | gemline | cascade
        "strict_canon": bool(config_params.get("strict_canon", True)),
        "device": config_params.get("device"),
        "delta_omega": float(config_params.get("delta_omega", results.get("delta_omega", 0.0) or 0.0)),
        "delta_t_cycle": float(config_params.get("delta_t_cycle", results.get("delta_t_cycle", 0.0) or 0.0)),
        "lambda_mode": config_params.get("lambda_mode", results.get("lambda_mode")),
        "termination_reason": results.get("termination_reason"),
        "delta_omega_bound_hit": bool(results.get("delta_omega_bound_hit", False)),
        "cycle_count": int(results.get("cycle_count", -1)),
        "half_life_sec_raw": float(results.get("half_life_sec", float("nan"))),
        "half_life_sec_constrained": float(results.get("half_life_sec_constrained", float("nan"))),
        "lambda_cycle_raw": float(results.get("lambda_cycle_raw", float("nan"))),
        "lambda_cycle_constrained": float(results.get("lambda_cycle_constrained", float("nan"))),
        "sim_params": dict(sim_params or {}),
    }
    if analysis:
        entry["analysis"] = {
            "phase_variance_trend": float(analysis.get("phase_variance_trend", float("nan"))),
            "exergy_trend": float(analysis.get("exergy_trend", float("nan"))),
            "monotonically_converging": bool(analysis.get("monotonically_converging", False)),
            "exergy_loss": float(analysis.get("exergy_loss", float("nan"))),
        }
    st.session_state.state_ledger.append(entry)

def render_state_ledger_sidebar():
    """Render ledger preview + export in the sidebar."""
    init_state_ledger()
    with st.sidebar.expander("📜 Canonical State Ledger", expanded=False):
        if not st.session_state.state_ledger:
            st.caption("No runs logged yet.")
            return
        last = st.session_state.state_ledger[-1]
        st.caption(f"Last run: {last.get('source')} @ {last.get('ts_utc')}")
        st.json({k: last[k] for k in [
            "source","strict_canon","termination_reason","cycle_count",
            "half_life_sec_raw","lambda_cycle_raw","delta_t_cycle","lambda_mode"
        ] if k in last})
        st.download_button(
            "⬇️ Download ledger (JSON)",
            data=json.dumps(st.session_state.state_ledger, indent=2).encode("utf-8"),
            file_name="polyrifringence_state_ledger.json",
            mime="application/json",
            key="download_state_ledger",
        )





def build_recursive_beam_path_figure(history: dict, show_ellipses: bool = True, show_path: bool = True, color_by: str = "cycle"):
    """Build the 3D recursive beam path figure with optional overlays.
    Diagnostic only; no operator behaviour changes.
    """
    E_history = history.get('E', [])
    if not E_history:
        return go.Figure()

    x_vals = [float(torch.real(E[0]).flatten()[0].detach().cpu().item()) for E in E_history]
    y_vals = [float(torch.real(E[1]).flatten()[0].detach().cpu().item()) for E in E_history]
    z_vals = list(range(len(E_history)))

    fig = go.Figure()
    if show_path:
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            name='Beam Path',
            marker=dict(size=5),
        ))

    if show_ellipses:
        theta = np.linspace(0, 2*np.pi, 50)
        for i, E in enumerate(E_history):
            Ex_amp = float(torch.abs(E[0]).flatten()[0].detach().cpu().item())
            Ey_amp = float(torch.abs(E[1]).flatten()[0].detach().cpu().item())
            Ex_local = Ex_amp * np.cos(theta) + x_vals[i]
            Ey_local = Ey_amp * np.sin(theta) + y_vals[i]
            fig.add_trace(go.Scatter3d(
                x=Ex_local, y=Ey_local, z=[z_vals[i]]*len(theta),
                mode='lines',
                name=f'Cycle {i}',
                showlegend=(i == 0)
            ))

    fig.update_layout(
        title="Recursive Beam Path",
        scene=dict(xaxis_title='Ex', yaxis_title='Ey', zaxis_title='Cycle'),
        height=650
    )
    return fig


def init_engine():
    """Initialize or get engine from session state"""
    if 'engine' not in st.session_state:
        config = EngineConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.complex128,  # FIX: match complex Jones math
            max_cycles=100,
            convergence_threshold=1e-10,
            delta_omega=0.142857,  # 1/7 for 6-7 cycle convergence
            exergy_half_life=(0.18, 0.24),
            delta_t_cycle=0.000375,
            lambda_mode="ratio_abs",
            enforce_delta_omega_bound=True,
        )
        st.session_state.engine = PolyrifringenceEngine(config)
        st.session_state.dashboard = InteractiveDashboard(st.session_state.engine)

    return st.session_state.engine


def render_header():
    """Render page header"""
    st.markdown('<div class="main-header">🌈 Polyrifringence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ΔΩΩΔ-Validated Recursive Birefringence Framework</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.header("🎚 Configuration")

    strict_canon = st.sidebar.toggle(
        "Strict Canon Mode",
        value=True,
        help=(
            "When enabled, locks the app to the canonical operator framing from the provided files: "
            "λ_cycle uses the Appendix C definition, ΔΩ applies the canonical subtractive form "
            "λ^(ΔΩ)=max(0,λ-ΔΩ_correction) with ΔΩ_correction:=ΔΩ_param·λ, and Δt_cycle is bounded "
            "to the documented empirical envelope."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("CSO Controls")

    observer_relay = st.sidebar.toggle(
        "Observer Relay (diagnostic only)",
        value=False,
        help=(
            "Enables the canonical CSO relay gate diagnostics. This does not steer the physical state; "
            "it only modulates a bounded damping coefficient (phase restoration factor) and is forcibly "
            "disengaged when the ΔΩ bound is hit."
        ),
    )

    if st.sidebar.button("Disavow / Disengage", help="Clears session state and exports a disengagement note."):
        st.session_state['disavow_log'] = (
            "DISAVOW / DISENGAGE\n"
            "The user disengaged from interactive participation.\n"
            "No observer feedback remains active; session state cleared.\n"
        )
        for k in list(st.session_state.keys()):
            if k not in ('disavow_log',):
                del st.session_state[k]
        st.sidebar.success("Session disengaged and cleared.")

    # Device selection
    device = st.sidebar.radio(
        "Compute Device",
        ["Auto (GPU > CPU)", "CPU", "GPU (CUDA)"],
        help="Select computational device. GPU recommended for performance."
    )

    if device == "CPU":
        device_choice = "cpu"
    elif device == "GPU (CUDA)":
        device_choice = "cuda"
    else:
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"

    st.sidebar.info(f"Current device: {device_choice}")

    # ΔΩ parameter
    if strict_canon:
        delta_omega = 0.142857
        st.sidebar.info("ΔΩ locked to 1/7 (0.142857) under Strict Canon Mode.")
    else:
        delta_omega = st.sidebar.slider(
            "ΔΩ (Coherence Constraint)",
            min_value=0.1,
            max_value=0.2,
            value=0.142857,
            step=0.001,
            help="ΔΩ parameter: 1/7 ≈ 0.142857 for 6-7 cycle convergence"
        )

    # λ-cycle definition (𝛌⃝)
    lambda_mode_options = {
        "Canonical (Appendix C): |1 - REGFₙ/REGFₙ₋₁|": "ratio_abs",
        "Engine-aligned diagnostic (Ω.6.7): log(REGFₙ/REGFₙ₊₁)": "log_ratio",
    }
    if strict_canon:
        lambda_mode_label = "Canonical (Appendix C): |1 - REGFₙ/REGFₙ₋₁|"
        lambda_mode = "ratio_abs"
        st.sidebar.info("λ_cycle definition locked to Appendix C under Strict Canon Mode.")
    else:
        lambda_mode_label = st.sidebar.selectbox(
            "λ_cycle Definition (𝛌⃝)",
            list(lambda_mode_options.keys()),
            index=0,
            help="Select which canon-defined λ_cycle form governs Λ̸ half-life calculations.",
        )
        lambda_mode = lambda_mode_options[lambda_mode_label]

    # Empirical cycle duration Δt_cycle
    if strict_canon:
        delta_t_cycle = st.sidebar.slider(
            "Δt_cycle (s per cycle)",
            min_value=0.00035,
            max_value=0.00040,
            value=0.000375,
            step=0.000001,
            format="%.6f",
            help="Canonical empirical cycle duration envelope from MATH_MODEL(Render).md Appendix C.",
        )
    else:
        delta_t_cycle = st.sidebar.number_input(
            "Δt_cycle (s per cycle)",
            min_value=0.0,
            value=0.000375,
            step=0.000025,
            format="%.6f",
            help="Cycle duration used to convert half-life from cycles to seconds.",
        )

    # Convergence threshold
    conv_threshold = st.sidebar.slider(
        "Convergence Threshold",
        min_value=1e-12,
        max_value=1e-8,
        value=1e-10,
        format="%.1e",
        help="Phase variance threshold for convergence"
    )

    # Exergy half-life envelope (seconds)
    if strict_canon:
        hl_min, hl_max = 0.18, 0.24
        st.sidebar.caption("Canonical half-life envelope: 0.18–0.24 s")
    else:
        hl_min = st.sidebar.slider(
            "Min Half-Life (s)",
            min_value=0.05,
            max_value=1.00,
            value=0.18,
            step=0.01
        )

        hl_max = st.sidebar.slider(
            "Max Half-Life (s)",
            min_value=0.05,
            max_value=2.00,
            value=0.24,
            step=0.01
        )

    return {
        'strict_canon': strict_canon,
        'observer_relay': observer_relay,
        'device': device_choice,
        'delta_omega': delta_omega,
        'convergence_threshold': conv_threshold,
        'exergy_half_life': (hl_min, hl_max),
        'delta_t_cycle': delta_t_cycle,
        'lambda_mode': lambda_mode,
    }


def render_simulation_controls(strict_canon: bool):
    """Render simulation control panel"""
    st.header("🎛️ Simulation Controls")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Initial Polarization State")

        ex_real = st.slider("Ex (Real)", -2.0, 2.0, 1.0, 0.1)
        ex_imag = st.slider("Ex (Imag)", -2.0, 2.0, 0.0, 0.1)
        ey_real = st.slider("Ey (Real)", -2.0, 2.0, 0.5, 0.1)
        ey_imag = st.slider("Ey (Imag)", -2.0, 2.0, 0.0, 0.1)

    with col2:
        st.subheader("Recursion Parameters")

        if strict_canon:
            st.caption("ΔΩ bound is active in Strict Canon Mode: effective recursion depth is capped at 7 cycles.")
            num_cycles = st.slider("Number of Cycles", 1, 7, 7, 1)
        else:
            st.warning(
                "Exploratory mode: ΔΩ recursion bound is disabled so you can run longer traces. "
                "These runs are diagnostic and are outside strict canonical operation.",
                icon="⚠️",
            )
            num_cycles = st.slider("Number of Cycles", 1, 200, 20, 1)
        wavelength = st.slider("Wavelength (nm)", 400, 700, 500, 10)

        theta_start = st.slider("θ Start (rad)", 0.0, 1.0, 0.1, 0.05)
        theta_end = st.slider("θ End (rad)", 0.0, 2.0, 0.7, 0.05)

    return {
        'ex_real': ex_real,
        'ex_imag': ex_imag,
        'ey_real': ey_real,
        'ey_imag': ey_imag,
        'num_cycles': num_cycles,
        'wavelength': wavelength,
        'theta_start': theta_start,
        'theta_end': theta_end
    }


def run_simulation(engine, sim_params):
    """Run simulation with given parameters"""
    # Create initial polarization state
    E_initial = torch.tensor(
        [complex(sim_params['ex_real'], sim_params['ex_imag']),
         complex(sim_params['ey_real'], sim_params['ey_imag'])],
        dtype=torch.complex128,
        device=engine.device
    )

    # Normalize
    if torch.norm(E_initial) > 0:
        E_initial = E_initial / torch.norm(E_initial)

    # Create sequences
    theta_sequence = np.linspace(
        sim_params['theta_start'],
        sim_params['theta_end'],
        sim_params['num_cycles']
    )

    lambda_sequence = [sim_params['wavelength'] * 1e-9] * sim_params['num_cycles']

    # Run recursion
    results = engine.run_recursion(E_initial, theta_sequence, lambda_sequence)

    return results


def render_results(results, engine):
    """Render simulation results"""
    st.header("📊 Results")

    bound_hit = bool(results.get('delta_omega_bound_hit', False))

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Canon: under ΔΩ-bounded termination, "converged" is not the primary success signal.
        if bound_hit:
            st.metric("Run Status", "ΔΩ-bounded")
        else:
            metric_val = "✓ Yes" if results['converged'] else "✗ No"
            st.metric("Numerical Converged", metric_val)

    with col2:
        cycle_color = "normal" if results['cycle_count'] <= 7 else "inverse"
        st.metric("Cycles", results['cycle_count'], delta_color=cycle_color)

    with col3:
        closure_eval = bool(results.get('closure_evaluated', True))
        if not closure_eval:
            st.metric("AΩ Closure", "Not evaluated")
        else:
            closure_val = "✓ Yes" if results['closure_achieved'] else "✗ No"
            st.metric("AΩ Closure", closure_val)

    with col4:
        hl_sec = float(results.get('half_life_sec', results.get('half_life', 0.0)))
        hl_cycles = results.get('half_life_cycles', None)
        if hl_sec == float('inf'):
            hl_display = "∞"
            hl_color = "normal"
        else:
            hl_display = f"{hl_sec:.4f}s"
            hl_color = "normal" if 0.18 <= hl_sec <= 0.24 else "inverse"
        st.metric("Half-Life", hl_display, delta_color=hl_color)

    # Context (canon): half-life is defined in cycles and converted to seconds via Δt_cycle.
    if hl_cycles is not None and hl_cycles != float('inf'):
        hl_c_sec = float(results.get('half_life_sec_constrained', hl_sec))
        hl_c_disp = "∞" if hl_c_sec == float('inf') else f"{hl_c_sec:.4f}s"
        st.caption(
            f"Λ̸ raw: {hl_cycles:.2f} cycles → {hl_display} using Δt_cycle {results.get('delta_t_cycle', 0.0):.6f}s; "
            f"ΔΩ-modified: {hl_c_disp} (λ_mode: {results.get('lambda_mode', 'ratio_abs')})"
        )
    elif hl_cycles == float('inf'):
        st.caption(
            f"Λ̸: λ→0 ⇒ t½→∞ (Δt_cycle {results.get('delta_t_cycle', 0.0):.6f}s, λ_mode: {results.get('lambda_mode', 'ratio_abs')})"
        )

    st.markdown("---")

    # Conformance checks
    st.subheader("ΔΩΩΔ Conformance Checks")

    col1, col2, col3 = st.columns(3)

    with col1:
        delta_omega_ok = results['cycle_count'] <= 7
        if delta_omega_ok:
            if bound_hit:
                st.markdown(
                    '<div class="success-box">✓ ΔΩ Bound: HIT at {}/7 cycles (canonical termination)</div>'.format(
                        results['cycle_count']
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="success-box">✓ ΔΩ Bound: {}/7 cycles (within bound)</div>'.format(
                        results['cycle_count']
                    ),
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="warning-box">⚠ ΔΩ Bound: {}/7 cycles (exceeds bound)</div>'.format(
                    results['cycle_count']
                ),
                unsafe_allow_html=True
            )

    with col2:
        hl_sec = float(results.get('half_life_sec', results.get('half_life', 0.0)))
        exergy_ok = (hl_sec == float('inf')) or (0.18 <= hl_sec <= 0.24)
        if exergy_ok:
            st.markdown(
                '<div class="success-box">✓ Exergy Half-Life: {} (within range)</div>'.format(
                    '∞' if hl_sec == float('inf') else f"{hl_sec:.4f}s"
                ),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="warning-box">⚠ Exergy Half-Life: {} (outside range)</div>'.format(
                    f"{hl_sec:.4f}s"
                ),
                unsafe_allow_html=True
            )

    with col3:
        closure_eval = bool(results.get('closure_evaluated', True))
        if not closure_eval:
            st.markdown('<div class="info-box">ℹ AΩ Closure: Not evaluated under ΔΩ-bounded termination</div>', unsafe_allow_html=True)
        else:
            closure_ok = results['closure_achieved']
            if closure_ok:
                st.markdown('<div class="success-box">✓ AΩ Closure: Achieved</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠ AΩ Closure: Not Achieved</div>', unsafe_allow_html=True)

    # Additional canon stability checks (Appendix C)
    col4, col5 = st.columns(2)
    with col4:
        lam_hist = results.get('history', {}).get('lambda_cycle_selected_raw', [])
        lam_raw = float(lam_hist[-1]) if lam_hist else 0.0
        lam_ok = 0.0013 <= lam_raw <= 0.0023
        if lam_ok:
            st.markdown(
                '<div class="success-box">✓ 𝛌⃝ Stability: λ_cycle={:.6f} (within [0.0013, 0.0023])</div>'.format(lam_raw),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warning-box">⚠ 𝛌⃝ Stability: λ_cycle={:.6f} (outside [0.0013, 0.0023])</div>'.format(lam_raw),
                unsafe_allow_html=True,
            )

    with col5:
        dt = float(results.get('delta_t_cycle', 0.0))
        dt_ok = 0.00035 <= dt <= 0.00040
        if dt_ok:
            st.markdown(
                '<div class="success-box">✓ Δt_cycle: {:.6f}s (within [0.00035, 0.00040])</div>'.format(dt),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warning-box">⚠ Δt_cycle: {:.6f}s (outside [0.00035, 0.00040])</div>'.format(dt),
                unsafe_allow_html=True,
            )

    st.markdown("---")


def render_visualizations(results, dashboard, engine, key_prefix: str = "viz"):
    """Render interactive visualizations"""
    st.header("📈📉 Visualizations")

    tabs = st.tabs(["Comprehensive Dashboard", "Recursive Beam Path", "Phase Variance", "Exergy Evolution", "ΔΩ Drift Collapse"])

    with tabs[0]:
        st.subheader("Comprehensive Dashboard")
        fig = dashboard.create_comprehensive_dashboard(results)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_dashboard")


    with tabs[1]:
        st.subheader("Recursive Beam Path")

        colA, colB, colC = st.columns(3)
        with colA:
            show_path = st.checkbox("Show path", value=True, key=f"{key_prefix}_bp_show_path")
        with colB:
            show_ellipses = st.checkbox("Show polarization ellipses", value=True, key=f"{key_prefix}_bp_show_ellipses")
        with colC:
            st.caption("Diagnostic view of geometric state evolution.")

        try:
            fig = build_recursive_beam_path_figure(results['history'], show_ellipses=show_ellipses, show_path=show_path)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_beam_path")
        except Exception as e:
            st.warning(f"Beam path visualization unavailable: {e}")

    with tabs[2]:

        st.subheader("Phase Variance Evolution")

        cycles = list(range(1, len(results['history']['phase_variance']) + 1))
        phase_var = results['history']['phase_variance']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles,
            y=phase_var,
            mode='lines+markers',
            name='Phase Variance',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))

        fig.add_hline(
            y=float(engine.config.convergence_threshold),
            line_dash="dash",
            line_color="green",
            annotation_text="Convergence Threshold"
        )

        fig.update_layout(
            title="Phase Variance Over Cycles",
            xaxis_title="Cycle",
            yaxis_title="Phase Variance (rad)",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_phase_variance")

    with tabs[3]:
        st.subheader("Exergy Evolution")

        cycles = list(range(len(results['history']['exergy'])))
        exergy = results['history']['exergy']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles,
            y=exergy,
            mode='lines+markers',
            name='Exergy',
            line=dict(color='purple', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ))

        hl_sec = float(results.get('half_life_sec', results.get('half_life', 0.0)))
        hl_title = "∞" if hl_sec == float('inf') else f"{hl_sec:.4f}s"
        fig.update_layout(
            title=f"Exergy Evolution (t₁/₂ = {hl_title})",
            xaxis_title="Cycle",
            yaxis_title="Exergy",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_exergy")

    with tabs[4]:
        st.subheader("ΔΩ Drift Collapse Dynamics")

        cycles = list(range(1, len(results['history']['decay_rate']) + 1))
        # Canonical: raw λ_cycle is the selected definition; constrained is ΔΩ-applied.
        raw_decay = results['history'].get('lambda_cycle_selected_raw', [])
        if not raw_decay:
            raw_decay = results['history'].get('lambda_cycle_ratio_abs', [])
        decay_rates = results['history']['decay_rate']
        delta_omega = engine.config.delta_omega

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles,
            y=raw_decay,
            mode='lines+markers',
            name='Raw λ_cycle',
            line=dict(color='orange', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=cycles,
            y=decay_rates,
            mode='lines+markers',
            name='ΔΩ-Constrained',
            line=dict(color='blue', width=2)
        ))

        fig.add_vline(x=7, line_dash="dash", line_color="red",
                      annotation_text="ΔΩ Cycle Limit")

        # Canonical admissible band for λ_cycle
        fig.add_hrect(y0=CANON_LAMBDA_BAND[0], y1=CANON_LAMBDA_BAND[1], opacity=0.08, line_width=0)

        fig.update_layout(
            title=f"ΔΩ Drift Collapse: Raw vs ΔΩ-Constrained λ_cycle (λ_mode={results.get('lambda_mode','ratio_abs')})",
            xaxis_title="Cycle",
            yaxis_title="λ_cycle",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_delta_omega")





def render_verification(engine, sim_params, strict_canon: bool):
    """Batch verification runner (multi-seed + bounded perturbations)."""
    st.header("✅ Verification Mode")

    st.markdown(
        "This mode runs multiple trials to test reproducibility and envelope behaviour. "
        "It never changes the canonical operator definitions; it only varies initial states and schedules "
        "within declared bounds."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_runs = st.number_input("Number of runs", min_value=5, max_value=500, value=50, step=5)
    with col2:
        seed = st.number_input("Base seed", min_value=0, max_value=10**9, value=1337, step=1)
    with col3:
        theta_jitter = st.slider("θ jitter (rad)", 0.0, 0.25, 0.05, 0.01)

    st.caption("Perturbations are bounded and disclosed. Under Strict Canon, the ΔΩ recursion cap remains active.")

    if st.button("Run verification batch", type="primary"):
        rng = np.random.default_rng(int(seed))
        results_list = []

        for i in range(int(n_runs)):
            s = int(seed) + i
            torch.manual_seed(s)
            np.random.seed(s % (2**32 - 1))

            # Random initial state (still normalized)
            E0 = torch.tensor(
                [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)), complex(rng.uniform(-1, 1), rng.uniform(-1, 1))],
                dtype=torch.complex128,
                device=engine.device,
            )
            if torch.norm(E0) > 0:
                E0 = E0 / torch.norm(E0)

            # Theta schedule with bounded jitter
            n_cyc = int(sim_params['num_cycles'])
            base_theta = np.linspace(sim_params['theta_start'], sim_params['theta_end'], n_cyc)
            jitter = rng.uniform(-theta_jitter, theta_jitter, size=n_cyc)
            theta_sequence = (base_theta + jitter).tolist()

            lambda_sequence = [sim_params['wavelength'] * 1e-9] * n_cyc
            r = engine.run_recursion(E0, theta_sequence, lambda_sequence)
            results_list.append(r)

        # Summaries
        cycles = [r['cycle_count'] for r in results_list]
        hl = [float(r.get('half_life_sec', r.get('half_life', 0.0))) for r in results_list]
        lam = [float(r.get('history', {}).get('lambda_cycle_selected_raw', [0.0])[-1]) for r in results_list]
        term = [r.get('termination_reason', 'unknown') for r in results_list]

        st.subheader("Batch summary")
        st.json({"runs": int(n_runs), "cycle_counts": {c: cycles.count(c) for c in sorted(set(cycles))}})
        in_band = [0.18 <= x <= 0.24 for x in hl if x != float('inf')]
        st.json({"half_life_in_band_rate": float(sum(in_band) / max(len(in_band), 1))})

        # Plots
        fig1 = go.Figure(data=[go.Histogram(x=cycles, nbinsx=20, name="Cycles")])
        fig1.update_layout(title="Cycle count distribution")
        st.plotly_chart(fig1, use_container_width=True, key="verify_chart_1")

        fig2 = go.Figure(data=[go.Histogram(x=[x for x in hl if x != float('inf')], nbinsx=30, name="Half-life (s)")])
        fig2.add_vline(x=0.18, line_dash="dash")
        fig2.add_vline(x=0.24, line_dash="dash")
        fig2.update_layout(title="Half-life distribution (seconds)")
        st.plotly_chart(fig2, use_container_width=True, key="verify_chart_2")

        fig3 = go.Figure(data=[go.Histogram(x=lam, nbinsx=30, name="λ_cycle")])
        fig3.add_vline(x=0.0013, line_dash="dash")
        fig3.add_vline(x=0.0023, line_dash="dash")
        fig3.update_layout(title="λ_cycle distribution (raw)")
        st.plotly_chart(fig3, use_container_width=True, key="verify_chart_3")

        # Export as JSON
        st.download_button(
            "Download batch results (JSON)",
            data=json.dumps({"results": results_list, "termination": term}, default=str, indent=2),
            file_name="verification_batch.json",
            mime="application/json",
        )




def render_gemline(engine):
    """Gem registry presets and validation.

    Canon rule: Gemline Mode is a deterministic *preset builder* (θ/λ schedule + metadata) using the registry.
    It does not introduce new physics claims.

    UX rule: If the user runs a Gemline simulation, the results must be rendered immediately in this tab
    (and persisted in session_state) so it never feels like a no-op.
    """
    st.header("💎 Gemline Mode")

    from poly_engine.gem_registry import codex_gems, physical_gems, describe, validate_registry

    st.caption(
        "Uses the unified gem registry as deterministic presets and metadata. "
        "No new physical claims are implied."
    )

    subset = st.radio(
        "Gem subset",
        ["Codex (foundation)", "Physical (non-foundation)"],
        horizontal=True,
        key="gem_subset",
    )
    gems = codex_gems() if subset.startswith("Codex") else physical_gems()

    # Registry validation
    errs = validate_registry(gems)
    if errs:
        st.warning("Registry validation warnings:\n" + "\n".join(errs[:10]))
    else:
        st.success("Registry validation: OK")

    names = list(gems.keys())
    default_sel = names[: min(7, len(names))]
    selected = st.multiselect(
        "Select gems (order is canonical in the registry)",
        options=names,
        default=default_sel,
        key="gem_selected",
    )
    if not selected:
        st.info("Select at least one gem.")
        return None

    st.subheader("Selected gem descriptors")
    for n in selected[:20]:
        st.write("- " + describe(n))

    # Build a simple deterministic schedule from selected gems
    # Canon note: this is a UI preset mapping, not a physics claim.
    n_cyc = len(selected)
    theta_sequence = np.linspace(0.1, 0.7, n_cyc).tolist()

    # Use available line index at 589nm when present, otherwise fall back to 500nm.
    lam_nm = []
    for n in selected:
        g = gems[n]
        if getattr(g, "n_589", None) is not None:
            lam_nm.append(589.0)
        else:
            lam_nm.append(500.0)
    lambda_sequence = [(x * 1e-9) for x in lam_nm]

    with st.expander("Gemline schedule preview", expanded=False):
        st.json({
            "cycles": n_cyc,
            "theta_start": float(theta_sequence[0]),
            "theta_end": float(theta_sequence[-1]),
            "lambda_nm": lam_nm,
        })

    run_btn = st.button("Run gemline simulation", key="run_gemline")
    if run_btn:
        # Initial state is default normalized
        E0 = torch.tensor([1.0 + 0j, 0.5 + 0j], dtype=torch.complex128, device=engine.device)
        E0 = E0 / torch.norm(E0)

        res = engine.run_recursion(E0, theta_sequence, lambda_sequence)
        st.session_state['results'] = res

        analyzer = TraceAnalyzer(engine)
        st.session_state['analysis'] = analyzer.analyze_recursion_trace(res)

        # Persist provenance so the UI knows what produced the current results.
        st.session_state['last_run_source'] = 'gemline'
        st.session_state['gemline_selected'] = list(selected)
        st.session_state['gemline_theta'] = list(theta_sequence)
        st.session_state['gemline_lambda_nm'] = list(lam_nm)

        st.success("Gemline simulation complete.")

    # Render any gemline results (either from this click or a prior run)
    if st.session_state.get('last_run_source') == 'gemline' and 'results' in st.session_state:
        res = st.session_state['results']
        analysis = st.session_state.get('analysis')

        st.markdown("---")
        st.subheader("Gemline run results")

        # Provenance block
        with st.expander("Provenance", expanded=False):
            st.json({
                "source": "gemline",
                "selected_gems": st.session_state.get('gemline_selected', []),
                "lambda_nm": st.session_state.get('gemline_lambda_nm', []),
                "theta": st.session_state.get('gemline_theta', []),
            })

        render_results(res, engine)
        render_visualizations(res, st.session_state.dashboard, engine, key_prefix="gemline")

        if analysis is None:
            analyzer = TraceAnalyzer(engine)
            analysis = analyzer.analyze_recursion_trace(res)
            st.session_state['analysis'] = analysis

        render_documentation(res, analysis, engine, key_prefix="gemline")

    return None





def render_cascade(engine):
    """Child-beam cascade experiment."""
    st.header("🌿 Child-Beam Cascade")
    st.caption("Branching recursion experiment. Under Strict Canon, ΔΩ bounds still apply to each recursion step.")

    col1, col2 = st.columns(2)
    with col1:
        branching = st.slider("Branching factor", 2, 8, 3, 1)
    with col2:
        depth = st.slider("Max depth", 1, 6, 3, 1)

    if st.button("Run cascade", key="cascade_btn_1"):
        E0 = torch.tensor([1.0 + 0j, 0.5 + 0j], dtype=torch.complex128, device=engine.device)
        E0 = E0 / torch.norm(E0)
        cascade = engine.child_beam_cascade(E0, branching_factor=int(branching), max_depth=int(depth))

        analyzer = TraceAnalyzer(engine)
        summary = analyzer.analyze_child_beam_cascade(cascade)
        st.subheader("Cascade summary")

        # Compact headline metrics
        st.write({
            "total_beams": int(summary.get("total_beams", 0)),
            "max_depth": int(summary.get("max_depth", 0)),
            "avg_branching_factor": float(summary.get("avg_branching_factor", 0.0))
        })

        # Depth table (readable view)
        depth_metrics = summary.get("depth_metrics", {}) or {}
        if depth_metrics:
            rows = []
            for d in sorted(depth_metrics.keys()):
                m = depth_metrics[d]
                rows.append({
                    "depth": int(d),
                    "beam_count": int(m.get("beam_count", 0)),
                    "avg_exergy": float(m.get("avg_exergy", 0.0)),
                    "exergy_std": float(m.get("exergy_std", 0.0)),
                    "avg_phase_variance": float(m.get("avg_phase_variance", 0.0)),
                    "phase_variance_std": float(m.get("phase_variance_std", 0.0)),
                })
            try:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception:
                st.table(rows)

        with st.expander("Show raw cascade summary (JSON)"):
            st.json(summary)

        # Simple depth plot
        depth_metrics = summary.get('depth_metrics', {})
        if depth_metrics:
            xs = sorted(depth_metrics.keys())
            y_ex = [depth_metrics[d]['avg_exergy'] for d in xs]
            fig = go.Figure(data=[go.Scatter(x=xs, y=y_ex, mode='lines+markers', name='avg exergy')])
            fig.update_layout(title="Average exergy by depth", xaxis_title="Depth", yaxis_title="Avg exergy")
            st.plotly_chart(fig, use_container_width=True, key="cascade_exergy_by_depth")





def render_analysis(results, engine):
    """Render detailed analysis (diagnostic + canon-context)."""
    st.header("🔬 Detailed Analysis")

    bound_hit = bool(results.get('delta_omega_bound_hit', False))
    term_reason = results.get('termination_reason', 'unknown')
    lambda_mode = results.get('lambda_mode', getattr(engine.config, 'lambda_mode', 'ratio_abs'))

    # Analyze trace (diagnostic only)
    analyzer = TraceAnalyzer(engine)
    analysis = analyzer.analyze_recursion_trace(results)

    st.subheader("Run Context")

    colA, colB, colC = st.columns(3)
    with colA:
        st.write(f"**Termination:** `{term_reason}`")
        st.write(f"**ΔΩ-bounded:** `{bound_hit}`")
    with colB:
        st.write(f"**Cycles:** `{int(results.get('cycle_count', -1))}`")
        st.write(f"**λ_mode:** `{lambda_mode}`")
    with colC:
        st.write(f"**Δt_cycle:** `{float(results.get('delta_t_cycle', 0.0)):.6f}s`")
        st.write(f"**ΔΩ:** `{float(getattr(engine.config,'delta_omega', 0.0)):.6f}`")

    st.subheader("λ⃝ & Temporal Dynamics")

    lam_raw = float(results.get('lambda_cycle_raw', float('nan')))
    lam_con = float(results.get('lambda_cycle_constrained', float('nan')))
    hl_cycles = float(results.get('half_life_cycles', float('nan')))
    hl_sec = float(results.get('half_life_sec', float('nan')))
    hl_sec_c = float(results.get('half_life_sec_constrained', float('nan')))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("λ_cycle (raw)", f"{lam_raw:.6f}" if lam_raw==lam_raw else "n/a")
        st.caption(f"Admissible band: [{CANON_LAMBDA_BAND[0]}, {CANON_LAMBDA_BAND[1]}]")
    with col2:
        st.metric("t½ (raw)", f"{hl_sec:.4f}s" if hl_sec==hl_sec else "n/a")
        st.caption("Canonical envelope: 0.18–0.24 s")
    with col3:
        st.metric("t½ (ΔΩ-modified)", f"{hl_sec_c:.4f}s" if hl_sec_c==hl_sec_c else "n/a")
        st.caption(f"t½ cycles: {hl_cycles:.2f}" if hl_cycles==hl_cycles else "t½ cycles: n/a")

    # Envelope conformance summary
    st.subheader("Envelope Conformance")

    l_ok = (lam_raw==lam_raw) and (CANON_LAMBDA_BAND[0] <= lam_raw <= CANON_LAMBDA_BAND[1])
    hl_ok = (hl_sec==hl_sec) and (CANON_HALF_LIFE_SEC[0] <= hl_sec <= CANON_HALF_LIFE_SEC[1])
    dt = float(results.get('delta_t_cycle', 0.0))
    dt_ok = CANON_DT_CYCLE_BAND[0] <= dt <= CANON_DT_CYCLE_BAND[1]

    colx, coly, colz = st.columns(3)
    with colx:
        st.success("✓ ΔΩ bound hit" if bound_hit else "ℹ ΔΩ bound not hit")
    with coly:
        st.success("✓ λ⃝ in-band" if l_ok else "⚠ λ⃝ out-of-band")
    with colz:
        st.success("✓ t½ in-band" if hl_ok else "⚠ t½ out-of-band")

    if not dt_ok:
        st.warning(f"Δt_cycle {dt:.6f}s outside canonical band {CANON_DT_CYCLE_BAND}.")
    else:
        st.info(f"Δt_cycle {dt:.6f}s within {CANON_DT_CYCLE_BAND}.")

    st.subheader("Recursion Trace Analysis")

    # Interpretive labels (UI-only): phase is exploratory under ΔΩ termination.
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Phase Exploration (diagnostic)**")
        st.write(f"- Trend: {analysis['phase_variance_trend']:.6e}")
        st.write(f"- Pre-termination contraction rate: {analysis['convergence_rate']:.6e}")
        st.write(f"- Monotonic (numeric): {analysis['monotonically_converging']}")

    with col2:
        st.markdown("**Exergy Dynamics (diagnostic)**")
        st.write(f"- Trend: {analysis['exergy_trend']:.6e}")
        st.write(f"- Total dissipation before termination: {analysis['exergy_loss']:.6e}")
        st.write(f"- Dissipation rate: {analysis['exergy_loss_rate']:.6e}")

    st.markdown("---")
    st.subheader("Stability Assessment")

    if bound_hit:
        st.markdown(
            '<div class="info-box">ℹ Non-monotonic phase behaviour is admissible under ΔΩ-bounded termination; numeric settling is intentionally truncated.</div>',
            unsafe_allow_html=True,
        )
    elif analysis['monotonically_converging']:
        st.markdown('<div class="success-box">✓ Numeric settling appears monotonic under the current threshold</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠ Non-monotonic numeric settling (diagnostic). Consider exploratory runs if needed.</div>',
                    unsafe_allow_html=True)

    # Exergy efficiency (unchanged)
    st.markdown("---")
    st.subheader("Exergy Efficiency")

    budget_calc = ExergyBudgetCalculator(engine)
    efficiency = budget_calc.exergy_efficiency(
        torch.tensor(results['history']['exergy'][-1]) if results['history'].get('exergy') else torch.tensor(0.0),
        torch.tensor(results['history']['exergy'][0]) if results['history'].get('exergy') else torch.tensor(1.0)
    )

    col1, col2, col3 = st.columns(3)
    exergy0 = float(results['history']['exergy'][0]) if results['history'].get('exergy') else float('nan')
    exergyN = float(results['history']['exergy'][-1]) if results['history'].get('exergy') else float('nan')
    with col1:
        if exergy0==exergy0 and exergyN==exergyN and exergy0!=0:
            st.metric("Preservation Ratio", f"{(exergyN/exergy0)*100:.2f}%")
        else:
            st.metric("Preservation Ratio", "n/a")
    with col2:
        st.metric("Overall Efficiency", f"{float(efficiency)*100:.2f}%")
    with col3:
        if exergy0==exergy0 and exergyN==exergyN:
            st.metric("Exergy Loss", f"{(exergy0-exergyN):.4f}")
        else:
            st.metric("Exergy Loss", "n/a")

def render_documentation(results, analysis, engine, key_prefix: str = "docs"):
    """Render documentation generation options"""
    st.header("📝 Documentation")

    col1, col2, col3 = st.columns(3)

    # NOTE: This section can be rendered in multiple contexts (e.g., Main Results,
    # Gemline, Verification). Streamlit auto-generates element IDs from type + label,
    # so we MUST provide unique keys to avoid StreamlitDuplicateElementId.

    with col1:
        if st.button("📋 Generate Report", key=f"{key_prefix}_btn_report"):
            doc_gen = DocumentationGenerator(engine)
            report = doc_gen.generate_analysis_report(results, analysis, "Polyrifringence Engine Analysis Report")
            st.download_button(
                label="Download Report (Markdown)",
                data=report,
                file_name="polyrifringence_report.md",
                mime="text/markdown",
                key=f"{key_prefix}_dl_report",
            )

    with col2:
        if st.button("📑 Generate Paper", key=f"{key_prefix}_btn_paper"):
            from poly_engine import PaperMetadata

            metadata = PaperMetadata(
                title="Recursive Birefringence with Symbolic Constraints",
                authors=["Conner Brown-Milliken"],
                affiliation="NinjaTech AI",
                abstract="Analysis of recursive birefringence simulation using Polyrifringence Engine",
                keywords=["birefringence", "recursion", "exergy", "coherence"]
            )

            doc_gen = DocumentationGenerator(engine)
            paper = doc_gen.generate_academic_paper(metadata, results, analysis)
            st.download_button(
                label="Download Paper (Markdown)",
                data=paper,
                file_name="polyrifringence_paper.md",
                mime="text/markdown",
                key=f"{key_prefix}_dl_paper",
            )

    with col3:
        if st.button("📖 Generate Presentation", key=f"{key_prefix}_btn_presentation"):
            doc_gen = DocumentationGenerator(engine)
            presentation = doc_gen.generate_presentation(results, analysis, "Polyrifringence Engine Results")
            st.download_button(
                label="Download Presentation (Markdown)",
                data=presentation,
                file_name="polyrifringence_presentation.md",
                mime="text/markdown",
                key=f"{key_prefix}_dl_presentation",
            )


def render_education():
    st.header("🎓 Education")

    tutorials = TutorialSystem()
    tutorial_names = tutorials.list_tutorials()
    if not tutorial_names:
        st.warning("No tutorials found.")
        return

    selected = st.selectbox("Select tutorial", tutorial_names)
    steps = tutorials.get_tutorial(selected)

    if not steps:
        st.warning("No steps found for this tutorial.")
        return

    step_titles = [s.title for s in steps]
    step_idx = st.selectbox("Step", list(range(len(step_titles))), format_func=lambda i: step_titles[i])
    step = steps[step_idx]

    st.subheader(step.title)
    st.markdown(step.content)

    if step.code_example:
        st.code(step.code_example.strip(), language="python")

    if step.quiz:
        st.markdown("### Quiz")
        st.write(step.quiz.get("question", ""))
        options = step.quiz.get("options", [])
        correct = step.quiz.get("correct", None)

        if options:
            choice = st.radio("Choose one:", options, index=0)
            if st.button("Check answer"):
                if correct is not None and choice == options[correct]:
                    st.success("Correct.")
                else:
                    st.error("Incorrect.")
        else:
            st.info("No quiz options provided.")

    st.markdown("---")
    st.subheader("Glossary lookup")
    query = st.text_input("Search glossary", value="")
    if query:
        matches = Glossary.search_terms(query)
        if not matches:
            st.info("No matches.")
        else:
            for term, definition in matches[:25]:
                st.markdown(f"**{term}**")
                st.write(definition)

    with st.expander("Jones matrix derivation preview"):
        st.text(DerivationWalkthrough.jones_matrix_derivation()[:2000])


def render_info_panel():
    """Render information panel about the framework"""
    st.header("ℹ️ About the 🌈Polyrifringence Engine")

    st.markdown("""
    <div class="info-box">
    <h3>ΔΩΩΔ-Validated Framework</h3>
    
    The Polyrifringence Engine is a GPU-accelerated recursive simulation framework that models 
    the behavior of light passing through birefringent materials with feedback loops and symbolic constraints.
    
    <h4>Core Operators:</h4>
    <ul>
    <li><strong>ΔΩ (Delta-Omega):</strong> Coherence Law ensuring drift collapse within 6-7 cycles</li>
    <li><strong>𝛌⃝ (Lambda-dot):</strong> Exergy Half-Life Operator (t₁/₂ ∈ [0.18, 0.24]s)</li>
    <li><strong>AΩ (Alpha-Omega):</strong> Identity Closure Principle</li>
    <li><strong>ZPEx:</strong> Zero-Point Exergy Operator</li>
    </ul>
    
    <h4>Non-Claims Framework:</h4>
    <ul>
    <li>✗ No new physical laws</li>
    <li>✗ No energy creation or amplification</li>
    <li>✗ No entropy reversal</li>
    <li>✓ All gains from structural organization and timing alignment</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    engine = init_engine()

    render_header()

    config_params = render_sidebar()

    # Canonical provenance ledger (instrumentation only)
    render_state_ledger_sidebar()

    # (Re)configure engine to match sidebar.
    # If device changes, we rebuild the engine (device is not safely mutable).
    need_rebuild = (
        getattr(st.session_state.engine.config, 'device', None) != config_params['device']
    )
    if need_rebuild:
        new_config = EngineConfig(
            device=config_params['device'],
            dtype=torch.complex128,
            max_cycles=st.session_state.engine.config.max_cycles,
            convergence_threshold=config_params['convergence_threshold'],
            delta_omega=config_params['delta_omega'],
            exergy_half_life=config_params['exergy_half_life'],
            delta_t_cycle=config_params['delta_t_cycle'],
            lambda_mode=config_params['lambda_mode'],
            enforce_delta_omega_bound=bool(config_params.get('strict_canon', True)),
            observer_feedback_enabled=bool(config_params.get('observer_relay', False)),
        )
        st.session_state.engine = PolyrifringenceEngine(new_config)
        st.session_state.dashboard = InteractiveDashboard(st.session_state.engine)

    # Update mutable config fields
    st.session_state.engine.config.delta_omega = config_params['delta_omega']
    st.session_state.engine.delta_omega.delta_omega = config_params['delta_omega']
    st.session_state.engine.config.convergence_threshold = config_params['convergence_threshold']
    st.session_state.engine.config.exergy_half_life = config_params['exergy_half_life']
    st.session_state.engine.lambda_dot.half_life_range = config_params['exergy_half_life']
    st.session_state.engine.config.delta_t_cycle = config_params['delta_t_cycle']
    st.session_state.engine.config.lambda_mode = config_params['lambda_mode']
    st.session_state.engine.config.enforce_delta_omega_bound = bool(config_params.get('strict_canon', True))
    st.session_state.engine.config.observer_feedback_enabled = bool(config_params.get('observer_relay', False))
    engine = st.session_state.engine

    sim_params = render_simulation_controls(bool(config_params.get('strict_canon', True)))

    if st.button("➰ Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running recursive birefringence simulation..."):
            results = run_simulation(engine, sim_params)
            st.session_state.results = results
            st.session_state.sim_params = sim_params

            analyzer = TraceAnalyzer(engine)
            analysis = analyzer.analyze_recursion_trace(results)
            st.session_state.analysis = analysis

            append_state_ledger_entry('simulation', config_params, sim_params, results, analysis)

            st.success("✓ Simulation complete!")

    if 'results' in st.session_state:
        results = st.session_state.results
        analysis = st.session_state.analysis

        tab_run, tab_verify, tab_gem, tab_cascade, tab_edu = st.tabs(
            ["🧪 Run Outputs", "✅ Verification", "💎 Gemline", "🌿 Cascade", "🎓 Education"]
        )

        with tab_run:
            render_results(results, engine)
            render_visualizations(results, st.session_state.dashboard, engine, key_prefix="main")
            render_analysis(results, engine)
            render_documentation(results, analysis, engine, key_prefix="main")

        with tab_verify:
            render_verification(engine, st.session_state.get('sim_params', sim_params), bool(config_params.get('strict_canon', True)))

        with tab_gem:
            render_gemline(engine)

        with tab_cascade:
            render_cascade(engine)

        with tab_edu:
            render_education()
    else:
        render_education()

    st.markdown("---")
    render_info_panel()

    if 'disavow_log' in st.session_state:
        st.download_button(
            "Download Disengage Log",
            data=str(st.session_state['disavow_log']),
            file_name="disavow_log.txt",
            mime="text/plain",
        )

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Polyrifringence Engine © Conner Brown-Milliken (@Wotcnt) | ΔΩΩΔ-Validated Framework</p>
    <p>Built with Streamlit, PyTorch, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
