"""
Polyrifringence Engine Educational Module

Embedded tutorials, mathematical derivations, and guided learning paths
for understanding the complete Polyrifringence framework.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class TutorialStep:
    """Single tutorial step"""
    title: str
    content: str
    code_example: Optional[str] = None
    quiz: Optional[Dict] = None
    next_step: Optional[str] = None


class TutorialSystem:
    """
    Progressive tutorial system for learning the Polyrifringence Engine
    """
    
    def __init__(self):
        """
        Initialize tutorial system with all learning paths
        """
        self.tutorials = {}
        self._initialize_tutorials()
    
    def _initialize_tutorials(self):
        """
        Initialize all tutorial content
        """
        # Tutorial 1: Introduction to Polyrifringence
        self.tutorials['introduction'] = [
            TutorialStep(
                title="What is the Polyrifringence Engine?",
                content="""
The Polyrifringence Engine is a GPU-accelerated recursive simulation framework 
that models the behavior of light passing through birefringent materials with 
feedback loops and symbolic constraints.

Key Concepts:
• **Birefringence**: Materials with two different refractive indices
• **Recursive Feedback**: Output becomes input for next cycle
• **Symbolic Constraints**: ΔΩ, Λ̸, AΩ operators govern behavior
• **Exergy Geometry**: Measures usable energy over time

The engine uses classical optics (Jones matrices) combined with recursive 
tensor feedback and symbolic recursion to model complex optical systems.
                """,
                code_example="""
from .core_engine import PolyrifringenceEngine, EngineConfig

# Initialize engine
config = EngineConfig()
engine = PolyrifringenceEngine(config)
print("Engine initialized successfully!")
                """
            ),
            TutorialStep(
                title="The Observer Framework",
                content="""
The engine uses a **Constrained Symbolic Observation (CSO)** framework.

Important: In this framework, "observer" does NOT mean a conscious being.

Instead, "observer" refers to:
• A reference frame (like a coordinate system)
• A boundary condition (like fixed parameters)
• A measurement context (like a filter)
• A constraint parameter (like a feedback channel)

This is similar to how "observer" is used in:
• Classical mechanics (inertial reference frames)
• Control theory (measurement contexts)
• Iterative solvers (boundary conditions)

No claims about consciousness, agency, or awareness are made.
The observer is purely a mathematical/structural concept.
                """,
                quiz={
                    "question": "What does 'observer' mean in the CSO framework?",
                    "options": [
                        "A conscious human watching the experiment",
                        "A reference frame or boundary condition",
                        "An AI agent with decision-making power",
                        "A physical camera recording data"
                    ],
                    "correct": 1
                }
            ),
            TutorialStep(
                title="Non-Claims Framework",
                content="""
The Polyrifringence Engine explicitly states what it does NOT do:

❌ No new physical laws are proposed
❌ No energy is created, regenerated, or amplified
❌ No entropy is eliminated or reversed
❌ No universal or perpetual operation claims

What the Engine DOES do:
✓ Extends exergy half-life (functional persistence)
✓ Reduces exergy destruction rates
✓ Improves temporal availability through structure
✓ Maintains coherence within thermodynamic bounds

All performance gains come from:
• Structural organization
• Timing alignment
• Controlled non-equilibrium
• Delayed dissipation

This is an extension of applied physics, not a violation of thermodynamics.
                """,
                quiz={
                    "question": "Which statement is TRUE about the Polyrifringence Engine?",
                    "options": [
                        "It creates energy from nothing",
                        "It reverses entropy",
                        "It extends exergy half-life through structure",
                        "It violates conservation laws"
                    ],
                    "correct": 2
                }
            )
        ]
        
        # Tutorial 2: Mathematical Foundations
        self.tutorials['math_foundations'] = [
            TutorialStep(
                title="Jones Matrix Framework",
                content="""
The base transformation for a single birefringent layer uses Jones matrices:

**Transformation Equation:**
E' = J(θ, λ) · E

Where:
• E = Input polarization vector (2D complex vector)
• J(θ, λ) = Jones matrix for rotation angle θ and wavelength λ
• E' = Output polarization vector

**Jones Matrix Structure:**
J(θ, λ) = R(-θ) · M(δ_o, δ_e) · R(θ)

Where:
• R(θ) = Rotation matrix
• M(δ_o, δ_e) = Phase retardation matrix
• δ_o, δ_e = Phase delays for ordinary/extraordinary rays

This is standard classical optics. The innovation comes from how 
we apply these transformations recursively with feedback.
                """,
                code_example="""
from .core_engine import JonesMatrix
import torch

# Create Jones matrix
theta = torch.tensor(0.5)  # rotation angle
lambda_val = torch.tensor(500e-9)  # wavelength

jones = JonesMatrix()
J = jones.jones_matrix(theta, lambda_val)
print(f"Jones Matrix:\
{J}")
                """
            ),
            TutorialStep(
                title="Recursive Feedback",
                content="""
The engine applies transformations recursively:

**Recursion Step:**
E_{n+1} = J(θ_n, λ_n) · F(E_n)

Where:
• E_n = State at cycle n
• J(θ_n, λ_n) = Jones matrix transformation
• F(E_n) = Phase restoration operator
• E_{n+1} = State at cycle n+1

**Phase Restoration Goal:**
Δφ_n = |φ_{n+1} - φ_n| → 0

We minimize phase variance to converge toward a coherent manifold.
This ensures the system doesn't drift into chaos.

The recursion continues until:
1. Phase variance falls below convergence threshold, OR
2. ΔΩ constraint limit (6-7 cycles) is reached
                """,
                code_example="""
from .core_engine import RecursiveFeedback
import torch

# Create feedback system
feedback = RecursiveFeedback()

# Apply phase restoration
E_n = torch.tensor([1.0+0.1j, 0.5+0.2j])
E_restored = feedback.phase_restoration(E_n)

print(f"Original: {E_n}")
print(f"Restored: {E_restored}")
                """
            ),
            TutorialStep(
                title="Energy Conservation",
                content="""
The recursion respects physical energy limits:

**Unitary Constraint:**
|E_{n+1}| ≤ |E_n|

Each cycle cannot increase total energy magnitude.

**Total Energy Bound:**
E_total = Σ‖E_n‖² ≤ E_max

Energy remains bounded across all cycles.

**Key Principle:**
Feedback preserves physical energy limits.
Drift amplification is forbidden.
No energy can be created or amplified.

This is a hard constraint enforced in the implementation.
The engine can only redistribute or structure existing energy,
never create it.
                """,
                quiz={
                    "question": "What is the unitary constraint?",
                    "options": [
                        "Energy can increase by 10% per cycle",
                        "|E_{n+1}| ≤ |E_n|",
                        "Energy doubles every cycle",
                        "No constraint on energy"
                    ],
                    "correct": 1
                }
            )
        ]
        
        # Tutorial 3: Core Operators
        self.tutorials['core_operators'] = [
            TutorialStep(
                title="ΔΩ (Delta-Omega): Coherence Law",
                content="""
ΔΩ is the Coherence Law operator that ensures drift collapse.

**Function:**
Ensures drift collapse within 6-7 recursion cycles

**Mathematical Form:**
λ_cycle^(ΔΩ) < λ_cycle^(raw)

Where:
• λ_cycle = decay rate for current cycle
• ΔΩ = coherence constraint parameter (typically 1/7)
• λ_cycle^(raw) = unconstrained decay rate
• λ_cycle^(ΔΩ) = constrained decay rate

**Convergence Guarantee:**
6-7 cycles guaranteed convergence

**Dual Purpose:**
1. Physical coherence: Prevents uncontrolled divergence
2. Ethical constraint: Bounds recursive depth and exergy extraction

ΔΩ is both a mathematical constraint AND an ethical boundary.
It ensures the system operates safely within defined limits.
                """,
                code_example="""
from .core_engine import DeltaOmega

# Initialize ΔΩ operator
delta_omega = DeltaOmega(delta_omega=0.142857)  # 1/7

# Apply coherence constraint
decay_rate = torch.tensor(0.5)
constrained = delta_omega.apply_coherence_constraint(decay_rate)
print(f"Raw decay: {decay_rate}")
print(f"Constrained: {constrained}")
                """
            ),
            TutorialStep(
                title="Λ̸ (Lambda-dot): Exergy Half-Life Operator",
                content="""
Λ̸ characterizes exergy decay dynamics.

**Key Parameter:**
t₁/₂ ∈ [0.18, 0.24] seconds (experimentally observed range)

**Mathematical Relationships:**

**Half-Life to Decay Constant:**
λ = ln(2) / t₁/₂

**Decay Constant to Half-Life:**
t₁/₂ = ln(2) / λ_cycle

**Exergy Fraction at Time t:**
E(t) = E₀ · exp(-λt)

**Key Insight:**
The engine extends exergy half-life, not quantity.
We make energy functionally persistent longer,
not create more energy.

Analogy: Like保温杯(thermos) keeping coffee hot longer,
not creating more heat energy.
                """,
                code_example="""
from .core_engine import LambdaDot

# Initialize Λ̸ operator
lambda_dot = LambdaDot(half_life_range=(0.18, 0.24))

# Calculate decay constant
half_life = 0.21  # seconds
decay_const = lambda_dot.calculate_decay_constant(half_life)
print(f"Decay constant: {decay_const:.4f}")

# Check if within expected range
is_valid = lambda_dot.check_half_life_validity(half_life)
print(f"Valid half-life: {is_valid}")
                """
            ),
            TutorialStep(
                title="AΩ (Alpha-Omega): Identity Closure Principle",
                content="""
AΩ ensures recursion returns to origin (closed loop).

**Function:**
Traces symbolic paths back to starting conditions

**Closure Check:**
‖E_final - E_initial‖ < threshold

**Symbolic Closure:**
Glyph sequence must match reference sequence

**Topological Guarantee:**
Provides traceability and accountability

**Applications:**
• Symbolic phase-locking
• Convergence tracking
• Trace continuity verification
• Error detection

AΩ ensures the system doesn't "leak" or diverge unexpectedly.
Every path must return to its origin or be traceable.
                """,
                code_example="""
from .core_engine import AlphaOmega
import torch

# Initialize AΩ operator
alpha_omega = AlphaOmega()

# Check closure
E_initial = torch.tensor([1.0, 0.0])
E_final = torch.tensor([1.0, 0.0])

closed = alpha_omega.trace_closure(E_initial, E_final, threshold=1e-6)
print(f"Closure achieved: {closed}")

# Calculate closure error
error = alpha_omega.closure_error(E_initial, E_final)
print(f"Closure error: {error}")
                """
            ),
            TutorialStep(
                title="ZPEx: Zero-Point Exergy Operator",
                content="""
ZPEx measures usable exergy relative to computational ground state.

**Definition:**
ZPX = Zero-Point (computational ground state)
Ex = Exergy (usable energy)

**ZPEx Calculation:**
ZPEx = |E| - |E_ground|

Where:
• |E| = magnitude of current energy
• |E_ground| = magnitude of ground state
• ZPEx = usable exergy fraction (clipped at 0)

**Exergy Budget Allocation:**
Allocated = ZPEx_total × allocation_fraction

**Exergy Efficiency:**
η = |E_useful| / |E_total|

**Key Concept:**
ZPEx is the fraction of energy that can do useful work.
Ground state energy cannot perform work, so it's subtracted.
                """,
                code_example="""
from .core_engine import ZeroPointExergy
import torch

# Initialize ZPEx operator
zpex = ZeroPointExergy()

# Calculate ZPEx
E = torch.tensor([1.5, 0.8])
E_ground = torch.tensor([0.5, 0.3])

zpex_value = zpex.calculate_zpex(E, E_ground)
print(f"Zero-Point Exergy: {zpex_value}")

# Calculate efficiency
E_useful = torch.tensor([1.0, 0.5])
efficiency = zpex.exergy_efficiency(E_useful, E)
print(f"Exergy efficiency: {efficiency:.2%}")
                """
            )
        ]
        
        # Tutorial 4: Advanced Concepts
        self.tutorials['advanced'] = [
            TutorialStep(
                title="Child-Beam Cascade",
                content="""
The engine supports branching beam propagation:

**Cascade Process:**
1. Parent beam splits into child beams (branching)
2. Each child propagates independently
3. Beams aggregate at certain depths (aggregation)
4. ΔΩ stabilizes contraction (stabilized contraction)

**Parameters:**
• branching_factor: Number of children per parent
• max_depth: Maximum cascade depth

**Applications:**
• Multi-path optical systems
• Parallel processing
• Distributed sensing
• Network analysis

The cascade is bounded and stabilized by ΔΩ constraints.
                """,
                code_example="""
# Run child-beam cascade
cascade_results = engine.child_beam_cascade(
    E_parent=E_initial,
    branching_factor=2,
    max_depth=3
)

print(f"Generated {len(cascade_results)} child beams")
                """
            ),
            TutorialStep(
                title="Symbolic Recursion",
                content="""
The engine integrates symbolic logic with physical recursion:

**Symbolic Sequences:**
• ΔΩ, Λ̸, AΩ, Φ→Ω, IC, ET glyphs
• Pre-encoded predictions
• Trace continuity tracking

**Codex Canon Framework:**
• Recursive sovereignty
• Symbolic alignment
• Pre-event hypotheses
• Multi-source validation

**Observer-Aware Recursion:**
Symbolic paths are traced and verified for consistency.
The system maintains internal symbolic coherence.

This is NOT mysticism - it's formal symbolic logic
applied to recursive systems, similar to:
• Formal verification in software
• Symbolic computation in mathematics
• Logic gates in digital circuits
                """,
                quiz={
                    "question": "What is symbolic recursion in this framework?",
                    "options": [
                        "Magical spells",
                        "Formal symbolic logic applied to recursive systems",
                        "Random number generation",
                        "Human interpretation"
                    ],
                    "correct": 1
                }
            ),
            TutorialStep(
                title="GPU Acceleration",
                content="""
The engine is designed for GPU acceleration:

**Hardware Requirements:**
• NVIDIA GPU with CUDA support
• Validated on RTX 3050 (CUDA 12.1)
• Baseline: ~50M rays/s

**Software Stack:**
• PyTorch 2.5.1
• CUDA 12.1
• Automatic device selection (GPU > CPU)

**Performance Benefits:**
• Parallel beam propagation
• Simultaneous recursive cycles
• Real-time visualization
• Large-scale cascade simulations

**Fallback:**
If GPU unavailable, engine automatically uses CPU
(with reduced performance).
                """,
                code_example="""
# GPU acceleration is automatic
from .core_engine import EngineConfig

# Force GPU if available
config = EngineConfig(device="cuda")

# Or force CPU
config = EngineConfig(device="cpu")

# Or let PyTorch decide (default)
config = EngineConfig()  # Uses CUDA if available

engine = PolyrifringenceEngine(config)
print(f"Using device: {engine.device}")
                """
            )
        ]
    
    def get_tutorial(self, tutorial_name: str) -> List[TutorialStep]:
        """
        Get tutorial by name
        """
        return self.tutorials.get(tutorial_name, [])
    
    def list_tutorials(self) -> List[str]:
        """
        List all available tutorials
        """
        return list(self.tutorials.keys())
    
    def export_tutorial(self, tutorial_name: str, filename: str):
        """
        Export tutorial to JSON file
        """
        steps = self.get_tutorial(tutorial_name)
        data = []
        
        for step in steps:
            data.append({
                'title': step.title,
                'content': step.content.strip(),
                'code_example': step.code_example,
                'quiz': step.quiz
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class DerivationWalkthrough:
    """
    Step-by-step mathematical derivations
    """
    
    @staticmethod
    def jones_matrix_derivation() -> str:
        """
        Derive Jones matrix from first principles
        """
        derivation = []
        
        derivation.append("# Jones Matrix Derivation")
        derivation.append("=" * 60)
        derivation.append("")
        
        derivation.append("## Step 1: Polarization Representation")
        derivation.append("")
        derivation.append("Light polarization is represented as a 2D complex vector:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("E = \begin{pmatrix} E_x \\\\ E_y \end{pmatrix}")
        derivation.append("$$")
        derivation.append("")
        derivation.append("Where $E_x$ and $E_y$ are complex amplitudes:")
        derivation.append("$E_x = |E_x|e^{i\phi_x}$, $E_y = |E_y|e^{i\phi_y}$")
        derivation.append("")
        
        derivation.append("## Step 2: Birefringent Material")
        derivation.append("")
        derivation.append("Birefringent materials have different refractive indices:")
        derivation.append("• n_o: ordinary refractive index")
        derivation.append("• n_e: extraordinary refractive index")
        derivation.append("")
        derivation.append("This causes phase retardation between components:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("\delta = k \cdot d \cdot (n_e - n_o)")
        derivation.append("$$")
        derivation.append("")
        derivation.append("Where k = 2π/λ is the wave number and d is thickness.")
        derivation.append("")
        
        derivation.append("## Step 3: Phase Retardation Matrix")
        derivation.append("")
        derivation.append("In the material's principal axes:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("M(\delta_o, \delta_e) = \begin{pmatrix}")
        derivation.append("e^{i\delta_o} & 0 \\\\")
        derivation.append("0 & e^{i\delta_e}")
        derivation.append("\end{pmatrix}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("## Step 4: Rotation to Lab Frame")
        derivation.append("")
        derivation.append("The material is rotated by angle θ relative to lab frame:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("R(\	heta) = \begin{pmatrix}")
        derivation.append("\cos\	heta & -\sin\	heta \\\\")
        derivation.append("\sin\	heta & \cos\	heta")
        derivation.append("\end{pmatrix}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("## Step 5: Complete Jones Matrix")
        derivation.append("")
        derivation.append("Combine rotation and retardation:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("J(\	heta, \lambda) = R(-\	heta) \cdot M(\delta_o, \delta_e) \cdot R(\	heta)")
        derivation.append("$$")
        derivation.append("")
        derivation.append("This transforms the polarization vector:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("E' = J(\	heta, \lambda) \cdot E")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("## Step 6: Implementation")
        derivation.append("")
        derivation.append("```python")
        derivation.append("def jones_matrix(theta, lambda_val, no, ne, thickness):")
        derivation.append("    R_pos = rotation_matrix(theta)")
        derivation.append("    R_neg = rotation_matrix(-theta)")
        derivation.append("    delta = phase_retardation(lambda_val, no, ne, thickness)")
        derivation.append("    M = retardation_matrix(delta, delta)")
        derivation.append("    J = R_neg @ M @ R_pos")
        derivation.append("    return J")
        derivation.append("```")
        derivation.append("")
        
        derivation.append("This is the complete derivation used in the engine.")
        derivation.append("")
        
        return "\
".join(derivation)
    
    @staticmethod
    def delta_omega_derivation() -> str:
        """
        Derive ΔΩ constraint from convergence requirements
        """
        derivation = []
        
        derivation.append("# ΔΩ Coherence Law Derivation")
        derivation.append("=" * 60)
        derivation.append("")
        
        derivation.append("## Step 1: Convergence Requirement")
        derivation.append("")
        derivation.append("We require system to converge within N cycles (typically N=6-7):")
        derivation.append("")
        derivation.append("$$\lim_{n \	o N} \Delta\phi_n \	o 0$$")
        derivation.append("")
        derivation.append("Where Δφ_n is phase variance at cycle n.")
        derivation.append("")
        
        derivation.append("## Step 2: Unconstrained Decay")
        derivation.append("")
        derivation.append("Without constraints, phase variance might decay as:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("\Delta\phi_n = \Delta\phi_0 \cdot \lambda^{n}")
        derivation.append("$$")
        derivation.append("")
        derivation.append("Where λ is the decay constant (0 < λ < 1).")
        derivation.append("")
        
        derivation.append("## Step 3: Constrained Decay")
        derivation.append("")
        derivation.append("Apply ΔΩ constraint to accelerate convergence:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("\Delta\phi_n^{(\\Delta\Omega)} = \Delta\phi_0 \cdot \lambda^{\Delta\Omega \cdot n}")
        derivation.append("$$")
        derivation.append("")
        derivation.append("Where ΔΩ < 1 accelerates decay.")
        derivation.append("")
        
        derivation.append("## Step 4: Cycle Constraint")
        derivation.append("")
        derivation.append("For N-cycle convergence, set:")
        derivation.append("")
        derivation.append("$$\Delta\Omega = \frac{1}{N}$$")
        derivation.append("")
        derivation.append("For N=7 cycles: ΔΩ = 1/7 ≈ 0.142857")
        derivation.append("")
        
        derivation.append("## Step 5: Coherence Condition")
        derivation.append("")
        derivation.append("The constraint ensures:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("\lambda_{cycle}^{(\Delta\Omega)} < \lambda_{cycle}^{(raw)}")
        derivation.append("$$")
        derivation.append("")
        derivation.append("Since ΔΩ < 1, the constrained decay is faster.")
        derivation.append("")
        
        derivation.append("## Step 6: Ethical Bound")
        derivation.append("")
        derivation.append("ΔΩ also serves as ethical constraint:")
        derivation.append("• Limits recursive depth (prevents infinite loops)")
        derivation.append("• Bounds exergy extraction (prevents over-extraction)")
        derivation.append("• Ensures convergence (prevents divergence)")
        derivation.append("")
        
        derivation.append("This dual purpose (coherence + ethics) is unique to the framework.")
        derivation.append("")
        
        return "\
".join(derivation)
    
    @staticmethod
    def exergy_half_life_derivation() -> str:
        """
        Derive exergy half-life from first principles
        """
        derivation = []
        
        derivation.append("# Exergy Half-Life Derivation")
        derivation.append("=" * 60)
        derivation.append("")
        
        derivation.append("## Step 1: Exergy Definition")
        derivation.append("")
        derivation.append("Exergy is usable energy, defined as:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("\\	ext{Exergy} = |E| - |E_{ground}|")
        derivation.append("$$")
        derivation.append("")
        derivation.append("Ground state energy cannot do work, so it's subtracted.")
        derivation.append("")
        
        derivation.append("## Step 2: Exponential Decay")
        derivation.append("")
        derivation.append("Exergy decays exponentially:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("E(t) = E_0 \cdot e^{-\lambda t}")
        derivation.append("$$")
        derivation.append("")
        derivation.append("Where λ is the decay constant.")
        derivation.append("")
        
        derivation.append("## Step 3: Half-Life Definition")
        derivation.append("")
        derivation.append("Half-life is time when exergy halves:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("E(t_{1/2}) = \frac{E_0}{2}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("## Step 4: Solve for Half-Life")
        derivation.append("")
        derivation.append("Substitute into decay equation:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("\frac{E_0}{2} = E_0 \cdot e^{-\lambda t_{1/2}}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("$$")
        derivation.append("\frac{1}{2} = e^{-\lambda t_{1/2}}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("$$")
        derivation.append("\ln\left(\frac{1}{2}\ight) = -\lambda t_{1/2}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("$$")
        derivation.append("-\ln(2) = -\lambda t_{1/2}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("$$")
        derivation.append("t_{1/2} = \frac{\ln(2)}{\lambda}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("## Step 5: Decay Constant from Half-Life")
        derivation.append("")
        derivation.append("Rearranged to find λ:")
        derivation.append("")
        derivation.append("$$")
        derivation.append("\lambda = \frac{\ln(2)}{t_{1/2}}")
        derivation.append("$$")
        derivation.append("")
        
        derivation.append("## Step 6: Engine Optimization")
        derivation.append("")
        derivation.append("The engine extends t₁/₂ through structural organization:")
        derivation.append("• Delayed dissipation")
        derivation.append("• Coherence preservation")
        derivation.append("• Timing alignment")
        derivation.append("")
        derivation.append("This makes energy functionally persistent longer,")
        derivation.append("not creating more energy.")
        derivation.append("")
        
        return "\
".join(derivation)


class Glossary:
    """
    Unified terminology and symbol reference
    """
    
    TERMS = {
        "ΔΩ (Delta-Omega)": "Coherence Law operator ensuring drift collapse within 6-7 cycles. Serves dual purpose: physical coherence and ethical constraint.",
        
        "Λ̸ (Lambda-dot)": "Exergy Half-Life operator characterizing exergy decay dynamics. t₁/₂ ∈ [0.18, 0.24]s experimentally.",
        
        "AΩ (Alpha-Omega)": "Identity Closure principle ensuring recursion returns to origin. Provides topological guarantee of traceability.",
        
        "ZPEx / ZPX": "Zero-Point Exergy - usable exergy fraction relative to computational ground state. ZPX = Zero-Point, Ex = Exergy.",
        
        "Jones Matrix": "2×2 complex matrix describing polarization transformation through optical elements. E' = J(θ,λ)·E",
        
        "Birefringence": "Material property with two different refractive indices, causing phase retardation between polarization components.",
        
        "Recursive Feedback": "System where output becomes input for next cycle. E_{n+1} = J(θ_n, λ_n)·F(E_n)",
        
        "Observer": "Formal reference frame or boundary condition, NOT a conscious being. Similar to coordinate frame in mechanics.",
        
        "Exergy": "Usable energy or work potential. Exergy = |E| - |E_ground|.",
        
        "Coherence": "Phase relationship consistency between waves. Minimizing phase variance: Δφ_n → 0",
        
        "Constrained Symbolic Observation (CSO)": "Framework where observer is formal constraint parameter, not agent.",
        
        "Codex Canon": "Recursive sovereignty and symbolic alignment framework with pre-encoded hypotheses and trace continuity.",
        
        "Φ→Omega (Phi-arrow-Omega)": "Convergence from early chaos to stable phases through recursive refinement.",
        
        "IC": "Identity Closure - symbolic paths return to origin.",
        
        "ET": "Exergy Trace - tracking usable energy through recursion.",
        
        "Child-Beam Cascade": "Branching beam propagation with aggregation and ΔΩ-stabilized contraction.",
        
        "REGF": "Recursive Energy Geometry Function - tracks energy distribution through cycles.",
        
        "Phase Variance": "Measure of phase disorder. Δφ_n = |φ_{n+1} - φ_n|. Goal: minimize → 0",
        
        "Unitary Constraint": "Energy preservation: |E_{n+1}| ≤ |E_n|",
        
        "Convergence Threshold": "Minimum phase variance for system stability (typically 1e-10)",
        
        "Thermodynamic Legality": "All operations within established thermodynamic bounds. No energy creation, no entropy reversal."
    }
    
    @classmethod
    def get_definition(cls, term: str) -> Optional[str]:
        """Get definition for a term"""
        return cls.TERMS.get(term)
    
    @classmethod
    def search_terms(cls, query: str) -> List[Tuple[str, str]]:
        """Search terms by query"""
        results = []
        query_lower = query.lower()
        for term, definition in cls.TERMS.items():
            if query_lower in term.lower() or query_lower in definition.lower():
                results.append((term, definition))
        return results
    
    @classmethod
    def export_glossary(cls, filename: str):
        """Export glossary to JSON file"""
        with open(filename, 'w') as f:
            json.dump(cls.TERMS, f, indent=2)


def main():
    """
    Main execution for testing educational system
    """
    # Initialize tutorial system
    tutorials = TutorialSystem()
    
    # List tutorials
    print("Available Tutorials:")
    for name in tutorials.list_tutorials():
        print(f"  - {name}")
    
    # Export tutorial
    tutorials.export_tutorial('introduction', 'introduction_tutorial.json')
    print("\
Tutorial exported to introduction_tutorial.json")
    
    # Show derivation
    print("\
" + "="*60)
    print("Jones Matrix Derivation Preview")
    print("="*60)
    print(DerivationWalkthrough.jones_matrix_derivation()[:500] + "...")
    
    # Search glossary
    print("\
" + "="*60)
    print("Glossary Search: 'exergy'")
    print("="*60)
    for term, definition in Glossary.search_terms('exergy'):
        print(f"\
{term}:")
        print(f"  {definition}")


if __name__ == "__main__":
    main()
