"""
Infrastructure Reliability & Simulation Library
===============================================

This library provides the core components for:
1. Hierarchical Bayesian Inference (Poisson-Gamma)
2. SimPy Process Logic for Infrastructure Assets

Usage:
    import des_reliability.probability
    # Use functions like des_reliability.probability.estimate_failure_rates_pymc()
"""

import simpy
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import logging
from typing import Dict, List, Tuple

# Configure logger
logger = logging.getLogger("InfraLib")

def setup_logging(level=logging.INFO):
    """Configures the logger for console output."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(name)s] - %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )
    # Suppress verbose PyMC logs
    logging.getLogger('pymc').setLevel(logging.WARNING)
    logging.getLogger('pytensor').setLevel(logging.WARNING)
    logging.getLogger('jax').setLevel(logging.WARNING)

# -------------------------------------------------------------------------
# Statistical Modeling (Bayesian)
# -------------------------------------------------------------------------

def estimate_failure_rates_pymc(df_logs: pd.DataFrame, n_assets: int, exposure_years: float = 5.0) -> np.ndarray:
    """
    Performs Hierarchical Bayesian Inference to estimate failure rates.
    Uses Poisson-Gamma Conjugate model for shrinkage estimation.
    
    Backend: JAX (NumPyro) for accelerated sampling.
    """
    logger.info("Starting Bayesian parameter estimation (MCMC using JAX/NumPyro)...")
    
    # Filter and aggregate data
    if 'event_type' in df_logs.columns:
        failures = df_logs[df_logs['event_type'] == 'failure']
    else:
        failures = df_logs
        
    # Group by 'asset_id' consistent with the data module
    stats = failures.groupby('asset_id').size()
    stats = stats.reindex(range(n_assets), fill_value=0)
    observed_counts = stats.values
    
    # Exposure time
    exposure = np.full(n_assets, exposure_years)

    # Import JAX sampler explicitly
    import pymc.sampling.jax as pmjax

    with pm.Model() as model:
        # Hyperpriors (Population Level)
        global_alpha = pm.Exponential("global_alpha", lam=1.0)
        global_beta = pm.Exponential("global_beta", lam=1.0)
        
        # Priors (Individual Level)
        lambdas = pm.Gamma("lambdas", alpha=global_alpha, beta=global_beta, shape=n_assets)
        
        # Likelihood
        mu = lambdas * exposure
        y_obs = pm.Poisson("y_obs", mu=mu, observed=observed_counts)
        
        # Inference using JAX (NumPyro NUTS)
        trace = pmjax.sample_numpyro_nuts(
            draws=1000, 
            tune=1000, 
            chains=2, 
            target_accept=0.95, 
            progressbar=False
        )
        
    # Extract posterior means
    posterior_means = az.summary(trace, var_names=["lambdas"])['mean'].values
    
    logger.info("Parameter estimation completed.")
    return posterior_means

# -------------------------------------------------------------------------
# Discrete Event Simulation Logic (Classes)
# -------------------------------------------------------------------------

class MaintainableAsset:
    """
    Generic SimPy Process representing the lifecycle of a maintainable asset
    (e.g., Lock, Bridge, Pump Station) subject to competing risks.
    """
    def __init__(self, env: simpy.Environment, asset_id: int, failure_rate: float, 
                 repair_resource: simpy.Resource, component_weights: Dict[str, float]):
        self.env = env
        self.asset_id = asset_id
        self.failure_rate = failure_rate
        self.repair_resource = repair_resource
        
        # Specific definition of components is now injected, making the class generic
        self.components = component_weights
        
        # Start
        self.process = env.process(self.lifecycle())

    def competing_risks_ttf(self) -> Tuple[float, str]:
        """Generates Time-To-Failure based on weakest link (Series System)."""
        delays = {}
        for comp, weight in self.components.items():
            if weight > 0:
                comp_rate = self.failure_rate * weight
                delays[comp] = np.random.exponential(1.0 / comp_rate)
        
        if not delays:
            return float('inf'), "None"

        winner = min(delays, key=delays.get)
        return delays[winner], winner

    def lifecycle(self):
        while True:
            # Operational Phase
            ttf, cause = self.competing_risks_ttf()
            yield self.env.timeout(ttf)
            
            # Failure Event
            logger.debug(f"T={self.env.now:.2f}: Asset {self.asset_id} FAILED due to {cause}.")
            
            # Repair Phase
            start_wait = self.env.now
            with self.repair_resource.request() as req:
                yield req
                wait_time = self.env.now - start_wait
                
                # Repair time (Generic Log-normal assumption, approx 2 days)
                repair_time = np.random.lognormal(-1, 0.5) / 365.0
                yield self.env.timeout(repair_time)
                
            logger.debug(f"T={self.env.now:.2f}: Asset {self.asset_id} REPAIRED (Wait: {wait_time:.4f}yr).")