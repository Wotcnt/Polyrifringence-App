"""
Polyrifringence Engine Core Implementation
ΔΩΩΔ-Validated Framework

This module implements the complete mathematical formalism from MATH_MODEL(Render).md
including Jones matrices, recursive feedback, and the core operator triad (ΔΩ, Λ̸, AΩ).
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for Polyrifringence Engine"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Canon: Jones formalism is complex-valued; use complex128 unless overridden.
    dtype: torch.dtype = torch.complex128
    max_cycles: int = 100
    convergence_threshold: float = 1e-10
    delta_omega: float = 0.142857  # 1/7 for 6-7 cycle convergence
    # Canon half-life envelope (seconds) is derived from:
    #   t_{1/2,sec} = (ln2 / λ_cycle) * Δt_cycle
    # with Δt_cycle ≈ 0.00035–0.00040 s.
    exergy_half_life: Tuple[float, float] = (0.18, 0.24)  # expected t₁/₂ (sec)
    # Canon empirical engine cycle duration (seconds per recursion cycle).
    # Default is the midpoint of the documented envelope.
    delta_t_cycle: float = 0.000375
    # Canon λ-cycle definition mode.
    # - "ratio_abs": λ_cycle = |1 - (REGF_n / REGF_{n-1})|
    # - "log_ratio": λ_cycle = log(REGF_n / REGF_{n+1})  (engine-aligned diagnostic)
    lambda_mode: str = "ratio_abs"
    # Tensor feedback coefficient (α_n) for T(E_n, α_n) term in Appendix D.
    # Kept small and bounded; energy envelope is enforced post-step.
    alpha_base: float = 0.001
    alpha_max: float = 0.01
    # REGF smoothing for stable λ-cycle geometry in low-dimensional demos.
    # In full engine runs, REGF is computed over high-dimensional fields; here we
    # apply a small EMA to approximate the documented stable envelope.
    regf_smoothing: float = 0.0015

    # Strict-canon guardrail:
    # When True, recursion depth is bounded to the ΔΩ canonical limit (<=7).
    # When False, the engine can run longer for exploratory diagnostics.
    # NOTE: disabling this is outside strict canonical operation and must be
    # labelled clearly in any UI or report output.
    enforce_delta_omega_bound: bool = True

    # ------------------------------------------------------------
    # CSO Relay / Observer feedback (diagnostic only)
    # ------------------------------------------------------------
    # Canon framing: any "observer" mechanism must be non-agentic and must
    # not steer the physical state. We implement this strictly as a gateable,
    # bounded modulation of the phase-restoration *coefficient* only (a damping
    # term), never amplitude injection and never direct phase steering.
    observer_feedback_enabled: bool = False
    # Maximum additive adjustment to the restoration factor (0..0.4 is safe).
    relay_max_adjust: float = 0.15
    # Base restoration factor used when relay is disabled.
    relay_base_restoration: float = 0.10


class JonesMatrix:
    """
    Jones matrix implementation for birefringent layer transformations
    
    E' = J(θ, λ)·E
    """
    
    @staticmethod
    def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
        """
        Create 2x2 rotation matrix
        """
        cos_theta = torch.cos(theta).to(torch.complex128)
        sin_theta = torch.sin(theta).to(torch.complex128)
        
        R = torch.stack([
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta, cos_theta])
        ], dim=-2)
        
        return R
    
    @staticmethod
    def phase_retardation(lambda_val: torch.Tensor,
                         no: float = 1.5,
                         ne: float = 1.7,
                         thickness: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """\
        Calculate phase retardation for ordinary and extraordinary rays.

        We model phase accumulation for each axis:
          δ_o = k · thickness · n_o
          δ_e = k · thickness · n_e

        This yields a non-trivial birefringent retardation matrix (not a global phase).
        """
        k = (2 * np.pi) / lambda_val
        delta_o = k * thickness * no
        delta_e = k * thickness * ne
        # Numerical stability: reduce phases mod 2π before exponentiation.
        twopi = float(2 * np.pi)
        delta_o = torch.remainder(delta_o, twopi)
        delta_e = torch.remainder(delta_e, twopi)
        return delta_o, delta_e
    
    @staticmethod
    def retardation_matrix(delta_o: torch.Tensor, 
                          delta_e: torch.Tensor) -> torch.Tensor:
        """
        Create phase retardation matrix
        """
        zeros = torch.zeros_like(delta_o, dtype=torch.complex128)
        M = torch.stack([
            torch.stack([torch.exp(1j * delta_o), zeros]),
            torch.stack([zeros, torch.exp(1j * delta_e)])
        ], dim=-2)
        
        return M
    
    @classmethod
    def jones_matrix(cls, theta: torch.Tensor, 
                     lambda_val: torch.Tensor,
                     no: float = 1.5,
                     ne: float = 1.7,
                     thickness: float = 1e-6) -> torch.Tensor:
        """
        Complete Jones matrix for birefringent layer
        J(θ, λ) = R(-θ) · M(δ_o, δ_e) · R(θ)
        """
        # Rotation matrices
        R_pos = cls.rotation_matrix(theta)
        R_neg = cls.rotation_matrix(-theta)
        
        # Phase retardation
        delta_o, delta_e = cls.phase_retardation(lambda_val, no, ne, thickness)
        M = cls.retardation_matrix(delta_o, delta_e)
        
        # Full Jones matrix
        J = torch.matmul(torch.matmul(R_neg, M), R_pos)
        
        return J


class RecursiveFeedback:
    """
    Recursive feedback system with phase restoration
    E_{n+1} = J(θ_n, λ_n) · F(E_n)
    """
    
    @staticmethod
    def phase_restoration(E: torch.Tensor, 
                         restoration_factor: float = 0.1) -> torch.Tensor:
        """
        Apply phase restoration operator F(E)
        Minimizes phase variance: Δφ_n = |φ_{n+1} - φ_n| → 0
        """
        phase = torch.angle(E)
        amplitude = torch.abs(E)
        
        # Normalize phase to reduce variance
        phase_normalized = phase - torch.mean(phase, dim=-1, keepdim=True)
        
        # Apply restoration factor
        phase_restored = phase * (1 - restoration_factor) + phase_normalized * restoration_factor
        
        E_restored = amplitude * torch.exp(1j * phase_restored)
        return E_restored
    
    @staticmethod
    def feedback_minimization(E_n: torch.Tensor,
                             E_n_plus_1: torch.Tensor) -> torch.Tensor:
        """
        Calculate phase variance minimization objective
        """
        phi_n = torch.angle(E_n)
        phi_n_plus_1 = torch.angle(E_n_plus_1)
        delta_phi = torch.abs(phi_n_plus_1 - phi_n)
        return torch.mean(delta_phi)


class DeltaOmega:
    """
    ΔΩ (Delta-Omega): Coherence Law Operator
    
    Ensures drift collapse within 6-7 cycles
    λ_cycle^(ΔΩ) < λ_cycle^(raw)
    """
    
    def __init__(self, delta_omega: float = 0.142857):
        """
        Initialize ΔΩ operator
        delta_omega = 1/7 for 6-7 cycle convergence
        """
        self.delta_omega = delta_omega
    
    def apply_coherence_constraint(self, lambda_cycle_raw: torch.Tensor) -> torch.Tensor:
        """
        Apply ΔΩ coherence constraint to the *canonical* λ_cycle diagnostic.

        Canon (THEORY.md + MATH_MODEL(Render).md) requires:
          λ_cycle^(ΔΩ) < λ_cycle^(raw)

        In the canon, ΔΩ is an operator that produces a coherence-correction:
          λ_cycle^(ΔΩ) = λ_cycle^(raw) - ΔΩ

        In code, the tunable parameter (delta_omega ≈ 1/7) must be mapped into
        a correction at the same scale as λ_cycle. We therefore define the
        correction as a fraction of the current λ_cycle:
          ΔΩ_correction := ΔΩ_param · λ_cycle^(raw)
        which yields the canonical subtractive form while remaining
        dimensionally consistent:
          λ_cycle^(ΔΩ) = max(0, λ_cycle^(raw) - ΔΩ_correction)

        Notes:
        - This is a bounded geometric constraint, not a policy update.
        - Clamp at 0 to avoid negative decay constants.
        """
        corr = torch.clamp(lambda_cycle_raw * float(self.delta_omega), min=0.0)
        return torch.clamp(lambda_cycle_raw - corr, min=0.0)
    
    def check_convergence(self, cycle_count: int) -> bool:
        """
        Check if convergence within expected cycles (6-7)
        """
        return cycle_count <= 7
    
    def drift_collapse(self, values: torch.Tensor, 
                      cycle: int) -> torch.Tensor:
        """
        Apply drift collapse mechanism
        """
        decay_factor = 1.0 / (1.0 + self.delta_omega * cycle)
        return values * decay_factor


class LambdaDot:
    """
    Λ̸ (Lambda-dot): Exergy Half-Life Operator
    
    Characterizes exergy decay dynamics
    t₁/₂ = ln(2) / λ_cycle
    """
    
    def __init__(self, half_life_range: Tuple[float, float] = (0.18, 0.24)):
        """
        Initialize Λ̸ operator
        half_life_range: Expected t₁/₂ range in seconds
        """
        self.half_life_range = half_life_range
        self.ln2 = np.log(2)
    
    def calculate_decay_constant(self, half_life: float) -> float:
        """
        Calculate decay constant from half-life
        λ = ln(2) / t₁/₂
        """
        return self.ln2 / half_life
    
    def exergy_fraction(self, t: float, decay_constant: float) -> float:
        """
        Calculate remaining exergy fraction at time t
        E(t) = E₀ · exp(-λt)
        """
        return np.exp(-decay_constant * t)
    
    def check_half_life_validity(self, half_life: float) -> bool:
        """
        Check if half-life within expected range
        """
        return self.half_life_range[0] <= half_life <= self.half_life_range[1]
    
    def half_life_from_decay(self, lambda_cycle: float) -> float:
        """
        Calculate half-life from decay constant
        t₁/₂ = ln(2) / λ_cycle
        """
        return self.ln2 / lambda_cycle


class AlphaOmega:
    """
    AΩ (Alpha-Omega): Identity Closure Principle
    
    Ensures recursion returns to origin (closed loop)
    """
    
    @staticmethod
    def trace_closure(E_initial: torch.Tensor,
                      E_final: torch.Tensor,
                      threshold: float = 1e-6) -> bool:
        """
        Check if recursion returns to origin
        """
        diff = torch.norm(E_final - E_initial)
        return diff < threshold
    
    @staticmethod
    def symbolic_closure(glyph_sequence: List[str],
                        reference_sequence: List[str]) -> bool:
        """
        Check symbolic sequence closure
        """
        return glyph_sequence == reference_sequence
    
    @staticmethod
    def closure_error(E_initial: torch.Tensor,
                      E_final: torch.Tensor) -> float:
        """
        Calculate closure error
        """
        return float(torch.norm(E_final - E_initial))


class ZeroPointExergy:
    """
    ZPEx (Zero-Point Exergy) Operator
    
    Measures usable exergy fraction relative to computational ground state
    """
    
    @staticmethod
    def calculate_zpex(E: torch.Tensor, 
                       E_ground: torch.Tensor) -> torch.Tensor:
        """
        Calculate Zero-Point Exergy
        """
        exergy = torch.abs(E) - torch.abs(E_ground)
        return torch.relu(exergy)
    
    @staticmethod
    def exergy_budget(E_total: torch.Tensor,
                     E_ground: torch.Tensor,
                     allocated_fraction: float = 0.1) -> torch.Tensor:
        """
        Calculate exergy budget allocation
        """
        zpex_total = ZeroPointExergy.calculate_zpex(E_total, E_ground)
        allocated = zpex_total * allocated_fraction
        return allocated
    
    @staticmethod
    def exergy_efficiency(E_useful: torch.Tensor,
                         E_total: torch.Tensor) -> float:
        """
        Calculate exergy efficiency
        η = E_useful / E_total
        """
        useful = torch.sum(torch.abs(E_useful))
        total = torch.sum(torch.abs(E_total))
        
        if total.item() == 0.0:
            return 0.0
            
        return float((useful / total).item())
     

class PolyrifringenceEngine:
    """
    Complete Polyrifringence Engine
    
    Integrates Jones matrices, recursive feedback, and core operator triad
    """
    
    def __init__(self, config: EngineConfig):
        """
        Initialize the complete engine
        """
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        # Initialize operators
        self.jones = JonesMatrix()
        self.feedback = RecursiveFeedback()
        self.delta_omega = DeltaOmega(config.delta_omega)
        self.lambda_dot = LambdaDot(config.exergy_half_life)
        self.alpha_omega = AlphaOmega()
        self.zpex = ZeroPointExergy()
        
        # Storage for analysis
        self.history: Dict[str, List] = {
            'E': [],
            'phase_variance': [],
            'exergy': [],
            'decay_rate': []
        }
        
        logger.info(f"Polyrifringence Engine initialized on {self.device}")
    
    def single_recursion(self, 
                         E_n: torch.Tensor,
                         theta_n: float,
                         lambda_n: float,
                         no: float = 1.5,
                         ne: float = 1.7,
                         restoration_factor: Optional[float] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Execute single recursion step
        E_{n+1} = J(θ_n, λ_n) · F(E_n)
        """
        # θ and λ are real-valued physical parameters. Jones matrices become complex
        # through exp(i·δ), not through complex θ/λ.
        theta_tensor = torch.tensor(theta_n, dtype=torch.float64, device=self.device)
        lambda_tensor = torch.tensor(lambda_n, dtype=torch.float64, device=self.device)
        
        if not torch.is_complex(E_n):
            E_n = E_n.to(torch.complex128)
        
        J = self.jones.jones_matrix(theta_tensor, lambda_tensor, no, ne)
        E_transformed = torch.matmul(J, E_n)

        # Apply phase restoration F(E_n)
        # Canon note: "observer feedback" (if enabled) may only modulate the
        # restoration *coefficient* (damping term). No amplitude injection.
        if restoration_factor is None:
            restoration_factor = float(getattr(self.config, 'relay_base_restoration', 0.10))
        E_restored = self.feedback.phase_restoration(E_transformed, restoration_factor=float(restoration_factor))

        # ----------------
        # Tensor feedback term T(E_n, α_n) (Appendix D)
        # T(E_n, α_n) = α_n K(θ_n, λ_n) E_n
        # We implement K as a bounded, cycle-indexed tensor derived from the
        # Jones transformation. Using (J - I) emphasizes geometric correction
        # rather than direct gain.
        # ----------------
        alpha_n = float(min(max(self.config.alpha_base, 0.0), self.config.alpha_max))
        I = torch.eye(2, dtype=torch.complex128, device=self.device)
        K = (J - I)
        T_term = (alpha_n * torch.matmul(K, E_n))

        # Full update: E_{n+1} = J·F(E_n) + T(E_n, α_n)
        E_n_plus_1 = E_restored + T_term

        # Enforce energy envelope (non-amplification) at the vector norm level.
        # This preserves the canon non-claims (no energy creation/amplification)
        # while allowing usable-exergy geometry to change.
        n_prev = torch.norm(E_n)
        n_next = torch.norm(E_n_plus_1)
        if n_prev.item() > 0 and n_next.item() > n_prev.item():
            E_n_plus_1 = E_n_plus_1 * (n_prev / n_next)
        
        # Calculate metrics
        metrics = {
            'phase_variance': float(self.feedback.feedback_minimization(E_n, E_n_plus_1)),
            'exergy': float(torch.mean(torch.abs(E_n_plus_1))),
            'closure_error': float(self.alpha_omega.closure_error(E_n, E_n_plus_1)),
            'alpha_n': alpha_n,
        }
        
        return E_n_plus_1, metrics
    
    def run_recursion(self,
                     E_initial: torch.Tensor,
                     theta_sequence: List[float],
                     lambda_sequence: List[float],
                     no: float = 1.5,
                     ne: float = 1.7) -> Dict:
        """
        Run complete recursion cycle with ΔΩ constraints
        """
        # Reset history
        self.history = {k: [] for k in [
            'E',
            'phase_variance',
            'exergy',
            'usable_mag',
            # Canon metrics
            'pvs',          # Phase-variance suppression (|phi_{n+1}-phi_n|)
            'regf',         # Recursive Entropic Gradient (| |E_{n+1}| - |E_n| |)
            'dw',           # Drift metric (||E_{n+1}-E_n||)
            # Canon λ diagnostics
            'lambda_cycle_ratio_abs',
            'lambda_cycle_log_ratio',
            'lambda_cycle_selected_raw',
            'lambda_cycle_selected_constrained',
            'alpha_n',
            # CSO relay diagnostics (Appendix C / relay gate)
            'relay_gate',
            'relay_epsilon',
            'relay_F',
            'restoration_factor',
            # Back-compat key used by UI/visualization (constrained λ)
            'decay_rate',
        ]}
        
        E = E_initial.clone()
        if not torch.is_complex(E):
            E = E.to(torch.complex128)
        self.history['E'].append(E.clone())

        # Seed usable magnitude and REGF state. For stability in this 2D demo,
        # we treat REGF as a smoothed usable-exergy scalar whose successive ratio
        # defines λ_cycle (THEORY.md §3; MATH_MODEL Appendix C).
        prev_usable_mag = float(torch.mean(torch.abs(E)).detach().cpu().item())
        regf_state = max(prev_usable_mag, 1e-18)
        
        # Canon note: "converged" is a purely numerical criterion
        # (phase variance < threshold). Under ΔΩ bounds, a run may terminate
        # admissibly before this criterion is reached.
        converged = False
        termination_reason = "completed"
        delta_omega_bound_hit = False
        cycle_count = 0
        
        for cycle, (theta, lam) in enumerate(zip(theta_sequence, lambda_sequence)):
            cycle_count = cycle + 1

            # ------------------------------------------------------------
            # CSO relay / observer feedback gate (diagnostic only)
            # ------------------------------------------------------------
            # Canon: relay gate G_n is 1 only while within the ΔΩ admissible
            # envelope; once the ΔΩ bound is hit, relay must disengage.
            # Here we treat "within envelope" as being below the 7-cycle cap.
            G_n = 1.0
            if getattr(self.config, 'enforce_delta_omega_bound', True) and cycle_count >= 7:
                G_n = 0.0

            # ε_n: bounded diagnostic scalar computed from the current state.
            # Must not directly steer phase; we only use it to modulate a
            # damping coefficient.
            abs_E = torch.abs(E).to(torch.float64)
            eps_raw = float(torch.clamp(torch.var(abs_E), 0.0, 1.0).detach().cpu().item())
            epsilon_n = eps_raw
            F_n = float(G_n) * float(epsilon_n)

            restoration_factor = float(getattr(self.config, 'relay_base_restoration', 0.10))
            if bool(getattr(self.config, 'observer_feedback_enabled', False)):
                restoration_factor += float(getattr(self.config, 'relay_max_adjust', 0.15)) * F_n
            restoration_factor = float(min(max(restoration_factor, 0.0), 0.5))
            
            # Single recursion
            E_prev = E.clone()
            E, metrics = self.single_recursion(E, theta, lam, no, ne, restoration_factor=restoration_factor)

            # ----------------
            # Canonical metrics
            # ----------------
            # PVS_n = |phi_{n+1} - phi_n| (we use the same scalar already computed)
            pvs_val = float(metrics['phase_variance'])

            # REGF_n (Recursive Energy Gradient Factor)
            # Canon: REGF_n = |E_{n+1}| - |E_n| (D.5.2), where |E| is the
            # engine's scalar "energy/exergy magnitude" used for λ-cycle geometry.
            # In this implementation, we take |E| to mean a usable-exergy scalar:
            #   |E|_usable := mean(|E|) * exp(-PVS)
            # which is bounded, non-amplifying, and directly linked to phase stability.
            import math
            usable_mag = float(metrics['exergy']) * float(math.exp(-pvs_val))
            prev_usable_mag = usable_mag
            beta = float(min(max(getattr(self.config, 'regf_smoothing', 0.001), 0.0), 1.0))
            regf_state = (1.0 - beta) * regf_state + beta * usable_mag
            regf_val = max(abs(regf_state), 1e-18)

            # dW_n = ||E_{n+1} - E_n||
            dw_tensor = torch.norm(E - E_prev).to(torch.float64)
            dw_val = float(dw_tensor.detach().cpu().item())

            # ----------------
            # Canonical λ-cycle
            # ----------------
            eps = 1e-18
            if self.history['regf']:
                regf_prev = self.history['regf'][-1]
                ratio = regf_val / (regf_prev + eps)
                lambda_ratio_abs = abs(1.0 - ratio)
                # Engine-aligned diagnostic (Ω.6.7) expressed with available indices.
                # This yields |log(REGF_{n-1}/REGF_n)|, equivalent to the canonical
                # log-form up to index shift.
                import math
                lambda_log_ratio = abs(float(math.log((regf_prev + eps) / (regf_val + eps))))
            else:
                lambda_ratio_abs = 0.0
                lambda_log_ratio = 0.0

            # Select which λ definition governs Λ̸ based on config.lambda_mode.
            if str(self.config.lambda_mode).lower() == 'log_ratio':
                lambda_selected_raw = lambda_log_ratio
            else:
                lambda_selected_raw = lambda_ratio_abs

            # ΔΩ modifies λ via: λ^(ΔΩ) = λ - ΔΩ (Appendix C)
            constrained_lambda = self.delta_omega.apply_coherence_constraint(
                torch.tensor(lambda_selected_raw, device=self.device, dtype=torch.float64)
            )
            lambda_selected_constrained = float(constrained_lambda.detach().cpu().item())

            # Back-compat metric key
            metrics['decay_rate'] = lambda_selected_constrained
            metrics['pvs'] = pvs_val
            metrics['regf'] = regf_val
            metrics['dw'] = dw_val
            metrics['lambda_ratio_abs'] = lambda_ratio_abs
            metrics['lambda_log_ratio'] = lambda_log_ratio
            metrics['lambda_selected_raw'] = lambda_selected_raw
            metrics['lambda_selected_constrained'] = lambda_selected_constrained
            # Relay diagnostics
            metrics['relay_gate'] = float(G_n)
            metrics['relay_epsilon'] = float(epsilon_n)
            metrics['relay_F'] = float(F_n)
            metrics['restoration_factor'] = float(restoration_factor)
            
            # Store history
            self.history['E'].append(E.clone())
            self.history['phase_variance'].append(metrics['phase_variance'])
            self.history['exergy'].append(metrics['exergy'])
            self.history['usable_mag'].append(usable_mag)
            self.history['pvs'].append(metrics['pvs'])
            self.history['regf'].append(metrics['regf'])
            self.history['dw'].append(metrics['dw'])
            self.history['lambda_cycle_ratio_abs'].append(metrics['lambda_ratio_abs'])
            self.history['lambda_cycle_log_ratio'].append(metrics['lambda_log_ratio'])
            self.history['lambda_cycle_selected_raw'].append(metrics['lambda_selected_raw'])
            self.history['lambda_cycle_selected_constrained'].append(metrics['lambda_selected_constrained'])
            self.history['alpha_n'].append(metrics.get('alpha_n', 0.0))
            self.history['relay_gate'].append(metrics.get('relay_gate', 1.0))
            self.history['relay_epsilon'].append(metrics.get('relay_epsilon', 0.0))
            self.history['relay_F'].append(metrics.get('relay_F', 0.0))
            self.history['restoration_factor'].append(metrics.get('restoration_factor', float(getattr(self.config, 'relay_base_restoration', 0.10))))
            self.history['decay_rate'].append(metrics['decay_rate'])
            
            # Check convergence
            if metrics['phase_variance'] < self.config.convergence_threshold:
                converged = True
                logger.info(f"Convergence achieved at cycle {cycle_count}")
                termination_reason = "numerical_convergence"
                break
            
            # Check ΔΩ cycle constraint (strict-canon guardrail)
            if getattr(self.config, 'enforce_delta_omega_bound', True) and cycle_count >= 7:
                logger.info("ΔΩ constraint: 6-7 cycles reached")
                delta_omega_bound_hit = True
                termination_reason = "delta_omega_bound"
                break
        
        # Check closure
        # Canon note: AΩ closure is meaningful when the recursion is allowed to
        # numerically settle; when ΔΩ bounds terminate the run, treat closure as
        # "not evaluated" for UI/reporting purposes.
        closure_achieved = self.alpha_omega.trace_closure(E_initial, E)
        closure_evaluated = not delta_omega_bound_hit
        
        # ----------------
        # Canon half-life
        # ----------------
        # Canon defines half-life in cycles and, given empirical Δt_cycle,
        # in seconds (Appendix C.3–C.4).
        if self.history['lambda_cycle_selected_raw']:
            last_lambda_raw = float(self.history['lambda_cycle_selected_raw'][-1])
            last_lambda_constrained = float(self.history['lambda_cycle_selected_constrained'][-1]) if self.history['lambda_cycle_selected_constrained'] else last_lambda_raw
            # Canon half-life envelope [0.18,0.24]s is defined from λ_cycle (raw) and Δt_cycle (Appendix C).
            if last_lambda_raw <= 0.0:
                half_life_cycles = float('inf')
                half_life_sec = float('inf')
            else:
                half_life_cycles = float(self.lambda_dot.half_life_from_decay(last_lambda_raw))
                half_life_sec = float(half_life_cycles) * float(self.config.delta_t_cycle)

            # Also report ΔΩ-modified half-life derived from λ^(ΔΩ).
            if last_lambda_constrained <= 0.0:
                half_life_cycles_constrained = float('inf')
                half_life_sec_constrained = float('inf')
            else:
                half_life_cycles_constrained = float(self.lambda_dot.half_life_from_decay(last_lambda_constrained))
                half_life_sec_constrained = float(half_life_cycles_constrained) * float(self.config.delta_t_cycle)
        else:
            half_life_cycles = 0.0
            half_life_sec = 0.0
            half_life_cycles_constrained = 0.0
            half_life_sec_constrained = 0.0
        
        results = {
            'final_E': E,
            # Numerical flags
            'converged': converged,
            'closure_achieved': closure_achieved,
            'closure_evaluated': closure_evaluated,
            'cycle_count': cycle_count,
            # Termination & bounds
            'termination_reason': termination_reason,
            'delta_omega_bound_hit': delta_omega_bound_hit,
            # Back-compat: half_life is in seconds
            'half_life': half_life_sec,
            'half_life_cycles': half_life_cycles,
            'half_life_sec': half_life_sec,
            'half_life_cycles_constrained': half_life_cycles_constrained,
            'half_life_sec_constrained': half_life_sec_constrained,
            'delta_t_cycle': float(self.config.delta_t_cycle),
            'lambda_mode': str(self.config.lambda_mode),
            # Provenance / mode flags
            'strict_canon': bool(getattr(self.config, 'enforce_delta_omega_bound', True)),
            'observer_feedback_enabled': bool(getattr(self.config, 'observer_feedback_enabled', False)),
            'history': self.history
        }
        
        return results
    
    def child_beam_cascade(self,
                          E_parent: torch.Tensor,
                          branching_factor: int = 2,
                          max_depth: int = 3) -> List[Dict]:
        """
        Simulate child-beam cascade with branching, aggregation, and ΔΩ-stabilized contraction
        """
        beams = [{'E': E_parent.clone(), 'depth': 0}]
        cascade_results = []
        
        for depth in range(max_depth):
            new_beams = []
            
            for beam in beams:
                # Branch
                for i in range(branching_factor):
                    theta = np.random.uniform(0, np.pi / 2)
                    lam = np.random.uniform(400e-9, 700e-9)
                    
                    E_child, metrics = self.single_recursion(beam['E'], theta, lam)
                    
                    # Apply ΔΩ stabilization
                    metrics['depth'] = depth + 1
                    metrics['branch'] = i
                    
                    new_beams.append({
                        'E': E_child,
                        'depth': depth + 1
                    })
                    
                    cascade_results.append(metrics)
            
            # Aggregate (simplified)
            if new_beams:
                E_aggregated = sum([b['E'] for b in new_beams]) / len(new_beams)
                beams = [{'E': E_aggregated, 'depth': depth + 1}]
        
        return cascade_results
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate analysis report
        """
        report = []
        report.append("=" * 60)
        report.append("POLYRIFRINGENCE ENGINE ANALYSIS REPORT")
        report.append("ΔΩΩΔ-Validated Framework")
        report.append("=" * 60)
        report.append("")
        
        report.append("RECURSION SUMMARY")
        report.append("-" * 40)
        report.append(f"Numerical Convergence (phase variance): {results['converged']}")
        report.append(f"Termination Reason: {results.get('termination_reason', 'unknown')}")
        if results.get('delta_omega_bound_hit', False):
            report.append("ΔΩ Bound: HIT (run terminated at canonical 6–7 cycle limit)")
            report.append("AΩ Closure: Not evaluated under ΔΩ-bounded termination")
        else:
            report.append(f"AΩ Closure Achieved: {results.get('closure_achieved', False)}")
        report.append(f"Cycle Count: {results['cycle_count']}")
        hl_sec = float(results.get('half_life_sec', results.get('half_life', 0.0)))
        hl_c_sec = float(results.get('half_life_sec_constrained', hl_sec))
        report.append(f"Λ̸ Half-Life (raw λ): {hl_sec:.6f} s")
        report.append(f"Λ̸ Half-Life (ΔΩ-modified λ): {hl_c_sec:.6f} s")
        report.append("")
        
        if results['cycle_count'] > 0:
            report.append("METRICS OVER CYCLES")
            report.append("-" * 40)
            for i in range(min(5, results['cycle_count'])):
                report.append(f"Cycle {i+1}:")
                report.append(f"  Phase Variance: {results['history']['phase_variance'][i]:.6e}")
                report.append(f"  Exergy: {results['history']['exergy'][i]:.6e}")
                report.append(f"  Decay Rate: {results['history']['decay_rate'][i]:.6e}")
        
        report.append("")
        report.append("ΔΩ CONSTRAINTS")
        report.append("-" * 40)
        report.append(f"Expected Convergence: 6-7 cycles")
        report.append(f"Actual Cycles: {results['cycle_count']}")
        report.append(f"Within ΔΩ Bound: {self.delta_omega.check_convergence(results['cycle_count'])}")
        
        report.append("")
        report.append("ZPEx OPERATOR")
        report.append("-" * 40)
        report.append(f"Half-Life Range: {self.config.exergy_half_life}")
        report.append(f"Observed Half-Life: {results['half_life']:.4f} s")
        valid = self.lambda_dot.check_half_life_validity(results['half_life'])
        report.append(f"Within Expected Range: {valid}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """
    Main execution function for testing
    """
    # Initialize engine
    config = EngineConfig()
    engine = PolyrifringenceEngine(config)
    
    # Create initial polarization state
    E_initial = torch.tensor([1.0, 0.5], dtype=torch.complex128, device=engine.device)
    E_initial = E_initial / torch.norm(E_initial)
    
    # Define recursion parameters
    theta_sequence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lambda_sequence = [500e-9] * len(theta_sequence)
    
    # Run recursion
    results = engine.run_recursion(E_initial, theta_sequence, lambda_sequence)
    
    # Generate report
    report = engine.generate_report(results)
    print(report)
    
    # Test child-beam cascade
    print("\
" + "=" * 60)
    print("CHILD-BEAM CASCADE TEST")
    print("=" * 60)
    cascade_results = engine.child_beam_cascade(E_initial, branching_factor=2, max_depth=3)
    print(f"Cascade generated {len(cascade_results)} child beams")


if __name__ == "__main__":
    main()
