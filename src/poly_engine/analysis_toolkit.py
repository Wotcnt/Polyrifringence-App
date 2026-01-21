"""
Polyrifringence Engine Analysis Toolkit

Parameter optimization, trace analysis, exergy budgeting, and advanced analysis tools
for understanding and optimizing Polyrifringence Engine performance.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_params: Dict[str, float]
    best_score: float
    history: List[Dict]
    converged: bool


class ParameterOptimizer:
    """
    Parameter optimization toolkit for ΔΩ, Λ̸, AΩ parameters
    Uses grid search and gradient-free optimization methods
    """
    
    def __init__(self, engine):
        """
        Initialize optimizer with Polyrifringence Engine
        """
        self.engine = engine
        self.optimization_history = []
    
    def grid_search_delta_omega(self,
                               theta_sequence: List[float],
                               lambda_sequence: List[float],
                               delta_omega_range: Tuple[float, float, int] = (0.1, 0.2, 20)
                               ) -> OptimizationResult:
        """
        Grid search for optimal ΔΩ parameter
        
        Args:
            theta_sequence: Rotation angles for recursion
            lambda_sequence: Wavelengths for recursion
            delta_omega_range: (min, max, num_points) for ΔΩ
        
        Returns:
            OptimizationResult with best parameters
        """
        print(f"Starting ΔΩ grid search with {delta_omega_range[2]} points...")
        
        min_val, max_val, num_points = delta_omega_range
        delta_omega_values = np.linspace(min_val, max_val, num_points)
        
        best_score = float('inf')
        best_delta_omega = min_val
        history = []
        
        for delta_omega in delta_omega_values:
            # Update engine config
            self.engine.config.delta_omega = delta_omega
            self.engine.delta_omega.delta_omega = delta_omega
            
            # Run simulation
            E_initial = torch.tensor([1.0, 0.5], dtype=torch.complex128, device=self.engine.device)
            E_initial = E_initial / torch.norm(E_initial)
            
            try:
                results = self.engine.run_recursion(E_initial, theta_sequence, lambda_sequence)
                
                # Calculate score (lower is better)
                # Score = weighted sum of: cycle count + closure_error + (1 - convergence)
                score = (
                    results['cycle_count'] * 0.4 +
                    results['history']['phase_variance'][-1] * 1e10 +
                    (0 if results['converged'] else 10)
                )
                
                history.append({
                    'delta_omega': delta_omega,
                    'score': score,
                    'cycle_count': results['cycle_count'],
                    'converged': results['converged']
                })
                
                if score < best_score:
                    best_score = score
                    best_delta_omega = delta_omega
                    
            except Exception as e:
                print(f"Error at ΔΩ={delta_omega:.4f}: {e}")
                continue
        
        result = OptimizationResult(
            best_params={'delta_omega': best_delta_omega},
            best_score=best_score,
            history=history,
            converged=True
        )
        
        print(f"Optimal ΔΩ: {best_delta_omega:.6f} (score: {best_score:.4f})")
        return result
    
    def optimize_exergy_half_life(self,
                                 half_life_range: Tuple[float, float, int] = (0.15, 0.30, 30)
                                 ) -> OptimizationResult:
        """
        Optimize exergy half-life parameter
        
        Args:
            half_life_range: (min, max, num_points) for t₁/₂
        
        Returns:
            OptimizationResult with best half-life
        """
        print(f"Optimizing exergy half-life in range {half_life_range[:2]}s...")
        
        min_val, max_val, num_points = half_life_range
        half_life_values = np.linspace(min_val, max_val, num_points)
        
        best_score = float('inf')
        best_half_life = min_val
        history = []
        
        for half_life in half_life_values:
            # Update engine config
            self.engine.config.exergy_half_life = (half_life, half_life)
            
            # Calculate decay constant
            decay_const = np.log(2) / half_life
            
            # Score based on alignment with expected range [0.18, 0.24]
            # and decay constant magnitude
            expected_min, expected_max = 0.18, 0.24
            
            if expected_min <= half_life <= expected_max:
                in_range_bonus = 0
            else:
                in_range_bonus = 10  # Penalty for being out of range
            
            score = abs(decay_const - 3.0) + in_range_bonus  # Target decay constant ~3.0
            
            history.append({
                'half_life': half_life,
                'decay_constant': decay_const,
                'score': score,
                'in_expected_range': expected_min <= half_life <= expected_max
            })
            
            if score < best_score:
                best_score = score
                best_half_life = half_life
        
        result = OptimizationResult(
            best_params={'half_life': best_half_life},
            best_score=best_score,
            history=history,
            converged=True
        )
        
        print(f"Optimal half-life: {best_half_life:.4f}s (score: {best_score:.4f})")
        return result
    
    def multi_objective_optimization(self,
                                    objectives: Dict[str, Callable],
                                    param_ranges: Dict[str, Tuple[float, float]],
                                    num_iterations: int = 100) -> OptimizationResult:
        """
        Multi-objective optimization using weighted sum method
        
        Args:
            objectives: Dict of objective functions to minimize
            param_ranges: Dict of parameter ranges (min, max)
            num_iterations: Number of optimization iterations
        
        Returns:
            OptimizationResult with Pareto-optimal parameters
        """
        print(f"Starting multi-objective optimization ({num_iterations} iterations)...")
        
        best_score = float('inf')
        best_params = {}
        history = []
        
        for iteration in range(num_iterations):
            # Sample parameters
            params = {
                name: np.random.uniform(low, high)
                for name, (low, high) in param_ranges.items()
            }
            
            # Calculate weighted score
            score = 0
            for obj_name, obj_func in objectives.items():
                try:
                    obj_value = obj_func(params)
                    weight = 1.0  # Equal weights by default
                    score += weight * obj_value
                except Exception as e:
                    score = float('inf')
                    break
            
            history.append({
                'iteration': iteration,
                'params': params.copy(),
                'score': score
            })
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            converged=True
        )
        
        print(f"Best score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        return result


class TraceAnalyzer:
    """
    Analyze symbolic traces and recursion paths
    """
    
    def __init__(self, engine):
        """
        Initialize trace analyzer with Polyrifringence Engine
        """
        self.engine = engine
    
    def analyze_recursion_trace(self, results: Dict) -> Dict:
        """
        Analyze complete recursion trace
        
        Args:
            results: Results from engine.run_recursion()
        
        Returns:
            Analysis dictionary with trace metrics
        """
        history = results['history']
        
        # Phase variance trend
        phase_var = history['phase_variance']
        phase_trend = np.polyfit(range(len(phase_var)), phase_var, 1)[0]
        
        # Exergy trend
        exergy = history['exergy']
        exergy_trend = np.polyfit(range(len(exergy)), exergy, 1)[0]
        
        # Convergence rate
        if len(phase_var) > 1:
            convergence_rate = (phase_var[0] - phase_var[-1]) / len(phase_var)
        else:
            convergence_rate = 0
        
        # Stability check
        stable = all(
            phase_var[i] > phase_var[i+1]
            for i in range(len(phase_var)-1)
        )
        
        analysis = {
            'phase_variance_trend': float(phase_trend),
            'exergy_trend': float(exergy_trend),
            'convergence_rate': float(convergence_rate),
            'monotonically_converging': stable,
            'final_phase_variance': phase_var[-1],
            'total_cycles': len(phase_var),
            'exergy_loss': exergy[0] - exergy[-1] if exergy else 0,
            'exergy_loss_rate': (exergy[0] - exergy[-1]) / len(exergy) if exergy else 0
        }
        
        return analysis
    
    def analyze_child_beam_cascade(self, 
                                   cascade_results: List[Dict]) -> Dict:
        """
        Analyze child-beam cascade
        
        Args:
            cascade_results: Results from engine.child_beam_cascade()
        
        Returns:
            Analysis dictionary with cascade metrics
        """
        if not cascade_results:
            return {}
        
        # Group by depth
        depth_groups = {}
        for result in cascade_results:
            depth = result.get('depth', 0)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(result)
        
        # Calculate metrics per depth
        depth_metrics = {}
        for depth, results in depth_groups.items():
            phase_vars = [r.get('phase_variance', 0) for r in results]
            exergies = [r.get('exergy', 0) for r in results]
            
            depth_metrics[depth] = {
                'beam_count': len(results),
                'avg_phase_variance': np.mean(phase_vars),
                'avg_exergy': np.mean(exergies),
                'phase_variance_std': np.std(phase_vars),
                'exergy_std': np.std(exergies)
            }
        
        analysis = {
            'total_beams': len(cascade_results),
            'max_depth': max(depth_groups.keys()) if depth_groups else 0,
            'depth_metrics': depth_metrics,
            'avg_branching_factor': np.mean([len(results) for results in depth_groups.values()])
        }
        
        return analysis
    
    def detect_anomalies(self, results: Dict, 
                        threshold_std: float = 2.0) -> List[Dict]:
        """
        Detect anomalies in recursion trace
        
        Args:
            results: Results from engine.run_recursion()
            threshold_std: Standard deviation threshold for anomaly detection
        
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        history = results['history']
        
        # Phase variance anomalies
        phase_var = np.array(history['phase_variance'])
        phase_mean = np.mean(phase_var)
        phase_std = np.std(phase_var)
        
        for i, pv in enumerate(phase_var):
            if abs(pv - phase_mean) > threshold_std * phase_std:
                anomalies.append({
                    'type': 'phase_variance',
                    'cycle': i,
                    'value': float(pv),
                    'expected_mean': float(phase_mean),
                    'deviation': float(abs(pv - phase_mean) / phase_std)
                })
        
        # Exergy anomalies
        exergy = np.array(history['exergy'])
        exergy_mean = np.mean(exergy)
        exergy_std = np.std(exergy)
        
        for i, ex in enumerate(exergy):
            if abs(ex - exergy_mean) > threshold_std * exergy_std:
                anomalies.append({
                    'type': 'exergy',
                    'cycle': i,
                    'value': float(ex),
                    'expected_mean': float(exergy_mean),
                    'deviation': float(abs(ex - exergy_mean) / exergy_std)
                })
        
        return anomalies
    
    def compare_traces(self, 
                      results_list: List[Dict],
                      labels: Optional[List[str]] = None) -> go.Figure:
        """
        Compare multiple recursion traces
        
        Args:
            results_list: List of results dictionaries
            labels: Optional list of labels for each trace
        
        Returns:
            Plotly figure comparing traces
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Phase Variance', 'Exergy', 'Decay Rate', 'All Metrics'),
            vertical_spacing=0.12
        )
        
        if labels is None:
            labels = [f"Trace {i+1}" for i in range(len(results_list))]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (results, label) in enumerate(zip(results_list, labels)):
            history = results['history']
            cycles = list(range(len(history['phase_variance'])))
            color = colors[idx % len(colors)]
            
            # Phase variance
            fig.add_trace(go.Scatter(
                x=cycles,
                y=history['phase_variance'],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                legendgroup=label
            ), row=1, col=1)
            
            # Exergy
            fig.add_trace(go.Scatter(
                x=list(range(len(history['exergy']))),
                y=history['exergy'],
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup=label
            ), row=1, col=2)
            
            # Decay rate
            fig.add_trace(go.Scatter(
                x=cycles,
                y=history['decay_rate'],
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup=label
            ), row=2, col=1)
        
        fig.update_layout(
            title_text="Recursion Trace Comparison",
            height=800,
            width=1200
        )
        
        return fig


class ExergyBudgetCalculator:
    """
    Calculate and manage exergy budgets
    """
    
    def __init__(self, engine):
        """
        Initialize exergy budget calculator with Polyrifringence Engine
        """
        self.engine = engine
    
    def calculate_total_exergy(self, 
                              E_initial: torch.Tensor,
                              num_cycles: int) -> Dict:
        """
        Calculate total exergy budget over cycles
        
        Args:
            E_initial: Initial polarization state
            num_cycles: Number of cycles to simulate
        
        Returns:
            Exergy budget dictionary
        """
        from .core_engine import ZeroPointExergy
        
        zpex = ZeroPointExergy()
        E_ground = torch.zeros_like(E_initial)
        
        total_exergy = 0
        exergy_per_cycle = []
        
        # Simulate cycles
        E = E_initial.clone()
        for cycle in range(num_cycles):
            cycle_exergy = float(torch.sum(zpex.calculate_zpex(E, E_ground)))
            exergy_per_cycle.append(cycle_exergy)
            total_exergy += cycle_exergy
            
            # Simple decay simulation
            E = E * 0.9
        
        budget = {
            'total_exergy': total_exergy,
            'exergy_per_cycle': exergy_per_cycle,
            'average_exergy': np.mean(exergy_per_cycle),
            'exergy_variance': np.var(exergy_per_cycle),
            'num_cycles': num_cycles,
            'exergy_ground_state': 0.0
        }
        
        return budget
    
    def allocate_exergy(self,
                       total_exergy: float,
                       allocations: Dict[str, float]) -> Dict:
        """
        Allocate exergy budget across components
        
        Args:
            total_exergy: Total available exergy
            allocations: Dict of component names and desired fractions
        
        Returns:
            Allocation dictionary with actual allocated exergy
        """
        # Normalize allocations
        total_fraction = sum(allocations.values())
        normalized_allocations = {
            k: v / total_fraction
            for k, v in allocations.items()
        }
        
        # Allocate exergy
        allocated = {
            k: total_exergy * fraction
            for k, fraction in normalized_allocations.items()
        }
        
        return allocated
    
    def optimize_allocation(self,
                           requirements: Dict[str, float],
                           total_exergy: float) -> Dict:
        """
        Optimize exergy allocation to maximize utility
        
        Args:
            requirements: Dict of component names and minimum requirements
            total_exergy: Total available exergy
        
        Returns:
            Optimized allocation dictionary
        """
        # Simple heuristic: allocate minimum requirements first,
        # then distribute remaining proportionally
        
        allocated = {}
        remaining = total_exergy
        remaining_components = list(requirements.keys())
        
        # Allocate minimum requirements
        for component, min_req in requirements.items():
            allocated[component] = min(remaining, min_req)
            remaining -= allocated[component]
        
        # Distribute remaining proportionally
        if remaining > 0 and remaining_components:
            per_component = remaining / len(remaining_components)
            for component in remaining_components:
                allocated[component] += per_component
        
        return allocated
    
    

    def exergy_efficiency(self, E_useful: torch.Tensor, E_total: torch.Tensor) -> float:
        """
        Compute exergy efficiency η = |E_useful| / |E_total|.

        This is a diagnostic convenience used by the UI. It does NOT modify the
        recursion or any canonical operator definitions.

        Args:
            E_useful: Tensor representing the usable component.
            E_total: Tensor representing the total component.

        Returns:
            Efficiency as a float in [0, 1] (clipped).
        """
        # Local import to avoid circular dependencies.
        from .core_engine import ZeroPointExergy

        try:
            eta = ZeroPointExergy.exergy_efficiency(E_useful, E_total)
            # Ensure plain float for UI.
            if isinstance(eta, torch.Tensor):
                eta = float(eta.detach().cpu().item())
            eta = float(eta)
        except Exception:
            # Fallback: robust scalar computation
            denom = float(torch.norm(E_total).detach().cpu().item()) if torch.norm(E_total) != 0 else 0.0
            num = float(torch.norm(E_useful).detach().cpu().item())
            eta = (num / denom) if denom > 0 else 0.0

        # Clip to [0, 1] for reporting
        if eta < 0.0:
            eta = 0.0
        if eta > 1.0:
            eta = 1.0
        return eta

def exergy_efficiency_analysis(self,
                                   results: Dict) -> Dict:
        """
        Analyze exergy efficiency over recursion
        
        Args:
            results: Results from engine.run_recursion()
        
        Returns:
            Efficiency analysis dictionary
        """
        from .core_engine import ZeroPointExergy
        
        history = results['history']
        exergy = history['exergy']
        
        if len(exergy) < 2:
            return {'error': 'Insufficient data for efficiency analysis'}
        
        # Calculate efficiency metrics
        initial_exergy = exergy[0]
        final_exergy = exergy[-1]
        total_exergy_input = initial_exergy * len(exergy)
        useful_exergy = sum(exergy)
        
        efficiency_metrics = {
            'exergy_preservation_ratio': final_exergy / initial_exergy if initial_exergy > 0 else 0,
            'overall_efficiency': useful_exergy / total_exergy_input if total_exergy_input > 0 else 0,
            'exergy_loss': initial_exergy - final_exergy,
            'exergy_loss_rate': (initial_exergy - final_exergy) / len(exergy),
            'peak_exergy': max(exergy),
            'peak_cycle': exergy.index(max(exergy)),
            'minimum_exergy': min(exergy),
            'exergy_variance': np.var(exergy),
            'exergy_range': max(exergy) - min(exergy)
        }
        
        return efficiency_metrics


class PerformanceBenchmark:
    """
    Benchmark engine performance
    """
    
    def __init__(self, engine):
        """
        Initialize benchmark with Polyrifringence Engine
        """
        self.engine = engine
        self.benchmark_results = []
    
    def benchmark_single_recursion(self, 
                                   num_runs: int = 100) -> Dict:
        """
        Benchmark single recursion performance
        
        Args:
            num_runs: Number of benchmark runs
        
        Returns:
            Benchmark statistics
        """
        import time
        
        times = []
        
        for _ in range(num_runs):
            E_initial = torch.randn(2, dtype=torch.complex128, device=self.engine.device)
            E_initial = E_initial / torch.norm(E_initial)
            
            theta_seq = [np.random.uniform(0, np.pi) for _ in range(7)]
            lambda_seq = [500e-9] * 7
            
            start_time = time.time()
            results = self.engine.run_recursion(E_initial, theta_seq, lambda_seq)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        stats = {
            'num_runs': num_runs,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'total_time': sum(times),
            'runs_per_second': num_runs / sum(times) if sum(times) > 0 else 0
        }
        
        self.benchmark_results.append(stats)
        return stats
    
    def benchmark_device_comparison(self, 
                                   num_runs: int = 50) -> Dict:
        """
        Benchmark CPU vs GPU performance
        
        Args:
            num_runs: Number of runs per device
        
        Returns:
            Comparison statistics
        """
        import time
        
        results = {}
        
        for device in ['cpu', 'cuda']:
            if device == 'cuda' and not torch.cuda.is_available():
                results[device] = {'error': 'CUDA not available'}
                continue
            
            print(f"Benchmarking {device}...")
            
            # Temporarily change device
            original_device = self.engine.device
            self.engine.device = torch.device(device)
            
            times = []
            for _ in range(num_runs):
                E_initial = torch.randn(2, dtype=torch.complex128, device=self.engine.device)
                E_initial = E_initial / torch.norm(E_initial)
                
                theta_seq = [np.random.uniform(0, np.pi) for _ in range(7)]
                lambda_seq = [500e-9] * 7
                
                start_time = time.time()
                self.engine.run_recursion(E_initial, theta_seq, lambda_seq)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            results[device] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'runs_per_second': num_runs / sum(times)
            }
            
            # Restore device
            self.engine.device = original_device
        
        # Calculate speedup
        if 'cpu' in results and 'cuda' in results:
            cpu_time = results['cpu'].get('mean_time')
            gpu_time = results['cuda'].get('mean_time')
            
            if cpu_time and gpu_time and gpu_time > 0:
                results['speedup'] = cpu_time / gpu_time
        
        return results
    
    def generate_benchmark_report(self) -> str:
        """
        Generate benchmark report
        """
        if not self.benchmark_results:
            return "No benchmark results available"
        
        report = []
        report.append("=" * 60)
        report.append("POLYRIFRINGENCE ENGINE BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")
        
        for idx, result in enumerate(self.benchmark_results):
            report.append(f"Benchmark Run {idx + 1}")
            report.append("-" * 40)
            
            for key, value in result.items():
                if isinstance(value, float):
                    report.append(f"{key}: {value:.6f}")
                else:
                    report.append(f"{key}: {value}")
            
            report.append("")
        
        return "\
".join(report)


def main():
    """
    Main execution for testing analysis toolkit
    """
    from .core_engine import PolyrifringenceEngine, EngineConfig
    
    # Initialize engine
    config = EngineConfig()
    engine = PolyrifringenceEngine(config)
    
    # Test parameter optimizer
    print("Testing Parameter Optimizer...")
    optimizer = ParameterOptimizer(engine)
    
    theta_seq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lambda_seq = [500e-9] * len(theta_seq)
    
    result = optimizer.grid_search_delta_omega(theta_seq, lambda_seq, (0.1, 0.2, 5))
    print(f"Best ΔΩ: {result.best_params['delta_omega']:.6f}")
    
    # Test trace analyzer
    print("\
Testing Trace Analyzer...")
    E_initial = torch.tensor([1.0, 0.5], dtype=torch.complex128)
    E_initial = E_initial / torch.norm(E_initial)
    
    results = engine.run_recursion(E_initial, theta_seq, lambda_seq)
    
    analyzer = TraceAnalyzer(engine)
    analysis = analyzer.analyze_recursion_trace(results)
    print(f"Convergence rate: {analysis['convergence_rate']:.6f}")
    print(f"Monotonically converging: {analysis['monotonically_converging']}")
    
    # Test exergy budget calculator
    print("\
Testing Exergy Budget Calculator...")
    budget_calc = ExergyBudgetCalculator(engine)
    efficiency = budget_calc.exergy_efficiency_analysis(results)
    print(f"Exergy preservation ratio: {efficiency['exergy_preservation_ratio']:.4f}")
    print(f"Overall efficiency: {efficiency['overall_efficiency']:.4f}")


if __name__ == "__main__":
    main()
