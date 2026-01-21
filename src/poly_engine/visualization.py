"""
Polyrifringence Engine Visualization Module

Creates interactive 3D visualizations of birefringence dynamics,
ΔΩ convergence tracking, and symbolic trace mapping.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


class BirefringenceVisualizer:
    """
    3D Birefringence Explorer
    Visualizes recursive beam paths and polarization evolution
    """
    
    def __init__(self, engine):
        """
        Initialize visualizer with Polyrifringence Engine
        """
        self.engine = engine
        self.figures = {}
    
    def plot_polarization_ellipse(self, 
                                 Ex: np.ndarray, 
                                 Ey: np.ndarray,
                                 title: str = "Polarization Ellipse") -> go.Figure:
        """
        Plot polarization ellipse from Ex and Ey components
        """
        fig = go.Figure()
        
        # Plot ellipse
        fig.add_trace(go.Scatter(
            x=Ex,
            y=Ey,
            mode='lines',
            name='Polarization',
            line=dict(color='blue', width=2)
        ))
        
        # Mark electric field vector
        fig.add_trace(go.Scatter(
            x=[Ex[0]],
            y=[Ey[0]],
            mode='markers',
            name='E₀',
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Ex',
            yaxis_title='Ey',
            showlegend=True,
            width=600,
            height=600
        )
        
        return fig
    
    def plot_recursive_beam_path_3d(self,
                                    history: Dict,
                                    title: str = "Recursive Beam Path") -> go.Figure:
        """
        3D visualization of recursive beam propagation
        """
        fig = go.Figure()
        
        E_history = history['E']
        
        # Extract real parts for visualization
        x_vals = [float(torch.real(E[0]).flatten()[0].detach().cpu().item()) for E in E_history]
        y_vals = [float(torch.real(E[1]).flatten()[0].detach().cpu().item()) for E in E_history]
        z_vals = list(range(len(E_history)))
        
        # Plot beam path
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='lines+markers',
            name='Beam Path',
            line=dict(color='cyan', width=4),
            marker=dict(size=5)
        ))
        
        # Add polarization ellipses at each step
        for i, E in enumerate(E_history):
            theta = np.linspace(0, 2*np.pi, 50)
            Ex_local = float(torch.abs(E[0]).flatten()[0].detach().cpu().item()) * np.cos(theta) + x_vals[i]
            Ey_local = float(torch.abs(E[1]).flatten()[0].detach().cpu().item()) * np.sin(theta) + y_vals[i]
            
            fig.add_trace(go.Scatter3d(
                x=Ex_local,
                y=Ey_local,
                z=[z_vals[i]] * 50,
                mode='lines',
                name=f'Cycle {i}',
                line=dict(width=1),
                showlegend=(i == 0)
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Ex',
                yaxis_title='Ey',
                zaxis_title='Cycle'
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def plot_phase_variance_evolution(self,
                                      history: Dict,
                                      title: str = "Phase Variance Evolution") -> go.Figure:
        """
        Plot phase variance over recursion cycles
        """
        fig = go.Figure()
        
        cycles = list(range(1, len(history['phase_variance']) + 1))
        phase_var = history['phase_variance']
        
        fig.add_trace(go.Scatter(
            x=cycles,
            y=phase_var,
            mode='lines+markers',
            name='Phase Variance',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Add convergence threshold line
        if self.engine.config.convergence_threshold:
            fig.add_hline(
                y=self.engine.config.convergence_threshold,
                line_dash="dash",
                line_color="green",
                annotation_text="Convergence Threshold"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Cycle',
            yaxis_title='Phase Variance (rad)',
            width=800,
            height=500
        )
        
        return fig


class DeltaOmegaVisualizer:
    """
    ΔΩ Convergence Visualizer
    Shows drift collapse dynamics and coherence constraints
    """
    
    @staticmethod
    def plot_drift_collapse(history: Dict,
                           delta_omega: float) -> go.Figure:
        """
        Visualize drift collapse under ΔΩ constraint
        """
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Raw Decay Rate', 'ΔΩ-Constrained Decay'),
                           vertical_spacing=0.15)
        
        cycles = list(range(1, len(history.get('decay_rate', [])) + 1))
        constrained_decay = history.get('decay_rate', [])

        # Canon: raw λ is the selected definition (ratio-abs or log-ratio),
        # and constrained λ is λ^(ΔΩ) = max(0, λ - ΔΩ).
        raw_decay = history.get('lambda_cycle_selected_raw', [])

        # Backward/robust fallbacks
        if not raw_decay:
            raw_decay = history.get('lambda_cycle_ratio_abs', constrained_decay)
        
        # Plot raw decay
        fig.add_trace(go.Scatter(
            x=cycles,
            y=raw_decay,
            mode='lines+markers',
            name='Raw Decay',
            line=dict(color='orange', width=2)
        ), row=1, col=1)
        
        # Plot constrained decay
        fig.add_trace(go.Scatter(
            x=cycles,
            y=constrained_decay,
            mode='lines+markers',
            name='ΔΩ-Constrained',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        # Add ΔΩ cycle limit
        fig.add_vline(x=7, line_dash="dash", line_color="red",
                     annotation_text="ΔΩ Cycle Limit (7)")
        
        fig.update_xaxes(title_text="Cycle")
        fig.update_yaxes(title_text="Decay Rate")
        fig.update_layout(height=600, width=800,
                         title_text="ΔΩ Drift Collapse Dynamics")
        
        return fig
    
    @staticmethod
    def plot_convergence_cycle_distribution(results_list: List[Dict]) -> go.Figure:
        """
        Plot distribution of convergence cycles across multiple runs
        """
        cycles = [r['cycle_count'] for r in results_list]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=cycles,
            nbinsx=20,
            name='Convergence Cycles',
            marker_color='lightblue'
        ))
        
        # Add ΔΩ expected range
        fig.add_vline(x=6, line_dash="dash", line_color="green",
                     annotation_text="ΔΩ Min (6)")
        fig.add_vline(x=7, line_dash="dash", line_color="green",
                     annotation_text="ΔΩ Max (7)")
        
        fig.update_layout(
            title="Distribution of Convergence Cycles",
            xaxis_title="Cycle Count",
            yaxis_title="Frequency",
            width=700,
            height=500
        )
        
        return fig


class ExergyVisualizer:
    """
    Exergy Geometry Visualizer
    Shows exergy half-life dynamics and ZPEx evolution
    """
    
    @staticmethod
    def plot_exergy_evolution(history: Dict,
                             half_life: float) -> go.Figure:
        """
        Plot exergy evolution over recursion cycles
        """
        fig = go.Figure()
        
        cycles = list(range(len(history['exergy'])))
        exergy = history['exergy']
        
        # Plot actual exergy
        fig.add_trace(go.Scatter(
            x=cycles,
            y=exergy,
            mode='lines+markers',
            name='Exergy',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ))
        
        # Plot theoretical decay curve
        if len(cycles) > 1:
            t = np.array(cycles)
            decay_const = np.log(2) / half_life if half_life > 0 else 1.0
            theoretical_exergy = exergy[0] * np.exp(-decay_const * t)
            
            fig.add_trace(go.Scatter(
                x=cycles,
                y=theoretical_exergy,
                mode='lines',
                name='Theoretical Decay',
                line=dict(color='gray', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f"Exergy Evolution (t₁/₂ = {half_life:.4f}s)",
            xaxis_title='Cycle',
            yaxis_title='Exergy',
            width=800,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_exergy_half_life_distribution(half_lives: List[float]) -> go.Figure:
        """
        Plot distribution of exergy half-lives
        """
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=half_lives,
            nbinsx=30,
            name='Half-Life Distribution',
            marker_color='lightgreen'
        ))
        
        # Add expected range
        fig.add_vline(x=0.18, line_dash="dash", line_color="blue",
                     annotation_text="Min (0.18s)")
        fig.add_vline(x=0.24, line_dash="dash", line_color="blue",
                     annotation_text="Max (0.24s)")
        
        fig.update_layout(
            title="Exergy Half-Life Distribution",
            xaxis_title="Half-Life (s)",
            yaxis_title="Frequency",
            width=700,
            height=500
        )
        
        return fig


class SymbolicTraceMapper:
    """
    Symbolic Trace Mapper
    Visualizes glyph sequences and symbolic paths
    """
    
    @staticmethod
    def plot_glyph_sequence(glyphs: List[str],
                           positions: Optional[List[Tuple[float, float]]] = None) -> go.Figure:
        """
        Plot symbolic glyph sequence
        """
        if positions is None:
            # Generate positions
            n = len(glyphs)
            positions = [(i * 2, np.sin(i * 0.5) * 2) for i in range(n)]
        
        fig = go.Figure()
        
        # Plot connections
        x_pos = [p[0] for p in positions]
        y_pos = [p[1] for p in positions]
        
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='lines',
            name='Symbolic Path',
            line=dict(color='gray', width=1)
        ))
        
        # Plot glyphs
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            name='Glyphs',
            marker=dict(size=20, color='gold'),
            text=glyphs,
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        # Mark start and end
        fig.add_trace(go.Scatter(
            x=[x_pos[0]],
            y=[y_pos[0]],
            mode='markers',
            name='Start',
            marker=dict(size=30, color='green', symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=[x_pos[-1]],
            y=[y_pos[-1]],
            mode='markers',
            name='End',
            marker=dict(size=30, color='red', symbol='circle')
        ))
        
        fig.update_layout(
            title="Symbolic Glyph Trace",
            xaxis_title='Symbolic Dimension',
            yaxis_title='Trace Amplitude',
            width=900,
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_codex_alignment(alignments: Dict[str, float]) -> go.Figure:
        """
        Plot Codex alignment metrics
        """
        fig = go.Figure()
        
        categories = list(alignments.keys())
        values = list(alignments.values())
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['red' if v < 0.5 else 'green' for v in values]
        ))
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Alignment Threshold")
        
        fig.update_layout(
            title="Codex Alignment Metrics",
            xaxis_title='Category',
            yaxis_title='Alignment Score',
            width=800,
            height=500
        )
        
        return fig


class InteractiveDashboard:
    """
    Unified Interactive Dashboard
    Combines all visualizations in one interface
    """
    
    def __init__(self, engine):
        """
        Initialize dashboard with Polyrifringence Engine
        """
        self.engine = engine
        self.biref_viz = BirefringenceVisualizer(engine)
        self.delta_omega_viz = DeltaOmegaVisualizer()
        self.exergy_viz = ExergyVisualizer()
        self.symbolic_mapper = SymbolicTraceMapper()
    
    def create_comprehensive_dashboard(self, 
                                      results: Dict,
                                      glyphs: Optional[List[str]] = None) -> go.Figure:
        """
        Create comprehensive dashboard with all visualizations
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Recursive Beam Path', 'Phase Variance',
                          'Exergy Evolution', 'ΔΩ Drift Collapse'),
            specs=[[{'type': 'scene'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        history = results['history']
        
        # 1. Recursive Beam Path (3D)
        E_history = history['E']
        x_vals = [float(torch.real(E[0]).flatten()[0].detach().cpu().item()) for E in E_history]
        y_vals = [float(torch.real(E[1]).flatten()[0].detach().cpu().item()) for E in E_history]
        z_vals = list(range(len(E_history)))
        
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            line=dict(color='cyan', width=3),
            marker=dict(size=4),
            name='Beam Path'
        ), row=1, col=1)
        
        # 2. Phase Variance
        cycles = list(range(1, len(history['phase_variance']) + 1))
        fig.add_trace(go.Scatter(
            x=cycles, y=history['phase_variance'],
            mode='lines+markers',
            line=dict(color='red', width=2),
            name='Phase Variance'
        ), row=1, col=2)
        
        # 3. Exergy Evolution
        fig.add_trace(go.Scatter(
            x=list(range(len(history['exergy']))),
            y=history['exergy'],
            mode='lines+markers',
            line=dict(color='purple', width=2),
            name='Exergy'
        ), row=2, col=1)
        
        # 4. ΔΩ Drift Collapse
        delta_omega = self.engine.config.delta_omega
        # Canonical raw λ_cycle comes from successive REGF ratios; do not invert ΔΩ mapping
        raw_decay = history.get('lambda_cycle_raw', history['decay_rate'])
        
        fig.add_trace(go.Scatter(
            x=cycles, y=raw_decay,
            mode='lines+markers',
            line=dict(color='orange', width=2),
            name='Raw Decay'
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=cycles, y=history['decay_rate'],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            name='ΔΩ-Constrained'
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Polyrifringence Engine Comprehensive Dashboard",
            height=900,
            width=1200,
            showlegend=True
        )
        
        # Update scene
        fig.update_scenes(
            xaxis_title="Ex",
            yaxis_title="Ey",
            zaxis_title="Cycle"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Cycle", row=1, col=2)
        fig.update_yaxes(title_text="Variance", row=1, col=2)
        fig.update_xaxes(title_text="Cycle", row=2, col=1)
        fig.update_yaxes(title_text="Exergy", row=2, col=1)
        fig.update_xaxes(title_text="Cycle", row=2, col=2)
        fig.update_yaxes(title_text="Decay Rate", row=2, col=2)
        
        return fig
    
    def export_to_html(self, fig: go.Figure, filename: str = "dashboard.html"):
        """
        Export visualization to HTML
        """
        fig.write_html(filename)
        return filename


def main():
    """
    Main execution for testing visualizations
    """
    from core_engine import PolyrifringenceEngine, EngineConfig
    
    # Initialize engine
    config = EngineConfig()
    engine = PolyrifringenceEngine(config)
    
    # Create initial state
    E_initial = torch.tensor([1.0, 0.5], dtype=torch.complex128)
    E_initial = E_initial / torch.norm(E_initial)
    
    # Run recursion
    theta_seq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lambda_seq = [500e-9] * len(theta_seq)
    
    results = engine.run_recursion(E_initial, theta_seq, lambda_seq)
    
    # Create visualizations
    dashboard = InteractiveDashboard(engine)
    fig = dashboard.create_comprehensive_dashboard(results)
    
    # Export
    dashboard.export_to_html(fig, "polyrifringence_dashboard.html")
    print("Dashboard exported to polyrifringence_dashboard.html")


if __name__ == "__main__":
    main()
