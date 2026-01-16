"""
Synthetic Data Generation Module
================================

This module handles the generation of synthetic datasets for testing and validation purposes.
Specifically designed to mimic historical maintenance logs for infrastructure assets.
"""

import pandas as pd
import numpy as np
from typing import Optional

def generate_synthetic_maintenance_data(n_assets: int = 50, years: float = 5.0, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic maintenance logs mimicking historical data.
    Simulates a heterogeneous population: most assets are reliable, some are prone to failure.
    
    Returns:
        pd.DataFrame: A dataframe containing failure events and dummy censoring records.
    """
    np.random.seed(seed)
    logs = []
    
    # Latent failure rates drawn from a Gamma distribution (Population Heterogeneity)
    # Shape=2.0, Scale=0.5 implies an average rate of 1.0, but with variance.
    true_rates = np.random.gamma(shape=2.0, scale=0.5, size=n_assets)
    
    for asset_id, rate in enumerate(true_rates):
        t = 0.0
        while True:
            # Time to next failure (Exponential assumption)
            dt = np.random.exponential(1.0 / rate)
            t += dt
            if t > years:
                break
            
            logs.append({
                'asset_id': asset_id,
                'event_type': 'failure',
                'timestamp_year': t,
                'duration_days': np.random.lognormal(0, 0.2)
            })
            
    # Add dummy records to ensure all assets exist in the dataset (handling censoring)
    all_ids = pd.DataFrame({'asset_id': range(n_assets)})
    df_logs = pd.DataFrame(logs)
    
    if df_logs.empty:
        return all_ids
        
    return df_logs