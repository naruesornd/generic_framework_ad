"""
Root Cause Analysis (RCA) module
Contains core functions, dashboards, multivariate analysis, and export utilities.
"""

from .core import run_automated_rca, run_automated_rca_sads
from .dashboard import plot_rca_dashboard
from .multivariate import plot_gradient_relationship, plot_dynamic_multivariate_anomalies
from .advanced import run_fast_system_backtest, plot_derivative_rca_dashboard, plot_advanced_dual_axis_dashboard
from .export import export_rca_to_excel

__all__ = [
    'run_automated_rca',
    'run_automated_rca_sads',
    'interactive_rca_dashboard',
    'plot_rca_dashboard',
    'plot_gradient_relationship',
    'plot_dynamic_multivariate_anomalies',
    'run_fast_system_backtest',
    'plot_derivative_rca_dashboard',
    'plot_advanced_dual_axis_dashboard',
    'export_rca_to_excel',
    'run_sads_rca',
    'plot_sads_rca_summary',
]
