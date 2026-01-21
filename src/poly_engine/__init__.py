"""
Polyrifringence Engine - Unified Suite

ΔΩΩΔ-Validated Framework
© Conner Brown-Milliken (@Wotcnt)

A comprehensive GPU-accelerated recursive simulation framework combining:
- Classical optical physics (Jones matrices, birefringence)
- Recursive tensor feedback (phase minimization, coherence restoration)
- Codex symbolic recursion (observer-state integration, ΔΩ ethics)
- Exergy geometry (Λ̸ decay constant, usable-exergy half-life)

This unified suite provides:
1. Core physics engine with complete mathematical formalism
2. Interactive 3D visualizations and real-time dashboards
3. Educational system with tutorials and derivations
4. Analysis toolkit for optimization and trace analysis
5. Documentation generator for reports, papers, and presentations
6. Web interface for accessible interaction

Non-Claims Framework:
• No new physical laws
• No energy creation or amplification
• No entropy reversal
• All gains from structural organization and timing alignment
"""

__version__ = "1.0.0"
__author__ = "Conner Brown-Milliken (@Wotcnt)"

from .core_engine import (
    EngineConfig,
    JonesMatrix,
    RecursiveFeedback,
    DeltaOmega,
    LambdaDot,
    AlphaOmega,
    ZeroPointExergy,
    PolyrifringenceEngine
)

from .visualization import (
    BirefringenceVisualizer,
    DeltaOmegaVisualizer,
    ExergyVisualizer,
    SymbolicTraceMapper,
    InteractiveDashboard
)

from .educational import (
    TutorialSystem,
    TutorialStep,
    DerivationWalkthrough,
    Glossary
)

from .analysis_toolkit import (
    ParameterOptimizer,
    TraceAnalyzer,
    ExergyBudgetCalculator,
    PerformanceBenchmark,
    OptimizationResult
)

from .documentation_generator import (
    DocumentationGenerator,
    PaperMetadata,
    CitationGenerator
)

__all__ = [
    # Core Engine
    'EngineConfig',
    'JonesMatrix',
    'RecursiveFeedback',
    'DeltaOmega',
    'LambdaDot',
    'AlphaOmega',
    'ZeroPointExergy',
    'PolyrifringenceEngine',
    
    # Visualization
    'BirefringenceVisualizer',
    'DeltaOmegaVisualizer',
    'ExergyVisualizer',
    'SymbolicTraceMapper',
    'InteractiveDashboard',
    
    # Educational
    'TutorialSystem',
    'TutorialStep',
    'DerivationWalkthrough',
    'Glossary',
    
    # Analysis
    'ParameterOptimizer',
    'TraceAnalyzer',
    'ExergyBudgetCalculator',
    'PerformanceBenchmark',
    'OptimizationResult',
    
    # Documentation
    'DocumentationGenerator',
    'PaperMetadata',
    'CitationGenerator'
]


def create_engine(config: EngineConfig = None) -> PolyrifringenceEngine:
    """
    Factory function to create a configured Polyrifringence Engine
    
    Args:
        config: Optional EngineConfig object. If None, uses defaults.
    
    Returns:
        Configured PolyrifringenceEngine instance
    """
    if config is None:
        config = EngineConfig()
    return PolyrifringenceEngine(config)


def create_dashboard(engine: PolyrifringenceEngine) -> InteractiveDashboard:
    """
    Factory function to create an interactive dashboard
    
    Args:
        engine: PolyrifringenceEngine instance
    
    Returns:
        InteractiveDashboard instance
    """
    return InteractiveDashboard(engine)


def quick_demo():
    """
    Quick demonstration of the Polyrifringence Engine
    """
    print("=" * 60)
    print("POLYRIFRINGENCE ENGINE QUICK DEMO")
    print("ΔΩΩΔ-Validated Framework")
    print("=" * 60)
    print()
    
    # Create engine
    print("Initializing engine...")
    config = EngineConfig()
    engine = create_engine(config)
    print(f"✓ Engine created on {engine.device}")
    print()
    
    # Create initial state
    print("Creating initial polarization state...")
    import torch
    E_initial = torch.tensor([1.0, 0.5], dtype=torch.complex128, device=engine.device)
    E_initial = E_initial / torch.norm(E_initial)
    print(f"✓ Initial state: {E_initial}")
    print()
    
    # Run recursion
    print("Running recursive birefringence simulation...")
    theta_seq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lambda_seq = [500e-9] * len(theta_seq)
    
    results = engine.run_recursion(E_initial, theta_seq, lambda_seq)
    print(f"✓ Simulation complete")
    print()
    
    # Display results
    print("RESULTS")
    print("-" * 40)
    print(f"Converged: {results['converged']}")
    print(f"Cycle Count: {results['cycle_count']}")
    print(f"Closure Achieved: {results['closure_achieved']}")
    print(f"Exergy Half-Life: {results['half_life']:.4f}s")
    print()
    
    # Generate report
    print("Generating analysis report...")
    report = engine.generate_report(results)
    print(report)
    
    print()
    print("Demo complete!")
    print()
    print("To explore further:")
    print("  • Run tutorials: python -m poly_engine.educational")
    print("  • Create visualizations: python -m poly_engine.visualization")
    print("  • Optimize parameters: python -m poly_engine.analysis_toolkit")
    print("  • Generate documentation: python -m poly_engine.documentation_generator")


if __name__ == "__main__":
    quick_demo()
