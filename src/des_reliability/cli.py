"""
Simulation Execution Script
===========================

This script demonstrates how to use the 'probability' library
to run a full infrastructure maintenance simulation.

Workflow:
1. Generate synthetic maintenance data (using data module).
2. Learn failure rates using Bayesian Inference (probability module).
3. Configure and Run the Discrete Event Simulation (probability module).
"""

import simpy
import numpy as np
import logging

# --- Imports ---
import des_reliability.probability
import des_reliability.data

# Setup
des_reliability.probability.setup_logging()
logger = logging.getLogger("SimulationMain")

# Configuration
N_ASSETS = 50
OBSERVATION_YEARS = 5.0
SIMULATION_HORIZON = 2.0
N_REPAIR_CREWS = 3

def main():
    # ---------------------------------------------------------
    # Data Preparation
    # ---------------------------------------------------------
    logger.info("--- RETRIEVING DATA ---")
    
    # Using the new separate module for synthetic data generation
    # Renamed to match the generic 'asset' terminology
    df_logs = des_reliability.data.generate_synthetic_maintenance_data(
        n_assets=N_ASSETS, 
        years=OBSERVATION_YEARS
    )
    
    total_failures = len(df_logs[df_logs['event_type'] == 'failure'])
    logger.info(f"Loaded {total_failures} failure events for {N_ASSETS} assets.")

    # ---------------------------------------------------------
    # Learning (Bayesian Inference)
    # ---------------------------------------------------------
    logger.info("--- LEARNING PARAMETERS ---")
    
    # This is the heavy lifting: turning scarce data into reliable rates
    estimated_rates = des_reliability.probability.estimate_failure_rates_pymc(
        df_logs, 
        n_assets=N_ASSETS, 
        exposure_years=OBSERVATION_YEARS
    )
    
    # Show example of shrinkage
    # Let's compare the first asset's raw rate vs Bayesian rate
    asset_0_events = len(df_logs[(df_logs['asset_id'] == 0) & (df_logs['event_type'] == 'failure')])
    raw_rate = asset_0_events / OBSERVATION_YEARS
    bayes_rate = estimated_rates[0]
    
    logger.info(f"Asset 0 Analysis: Raw Rate={raw_rate:.2f}/yr | Bayesian Rate={bayes_rate:.2f}/yr")

    # ---------------------------------------------------------
    # Discrete Event Simulation
    # ---------------------------------------------------------
    logger.info("--- RUNNING SIMULATION ---")
    
    env = simpy.Environment()
    
    # Shared Resource: Maintenance Crews
    repair_crews = simpy.Resource(env, capacity=N_REPAIR_CREWS)
    
    # Initialize all asset processes using the LEARNED parameters
    for i in range(N_ASSETS):
        des_reliability.probability.MaintainableAsset(
            env=env,
            asset_id=i,
            failure_rate=estimated_rates[i],
            repair_resource=repair_crews,
            component_weights={'Pump': 0.6, 'LockGate': 0.3, 'ControlSystem': 0.1}
        )
        
    # Run
    env.run(until=SIMULATION_HORIZON)
    logger.info("Simulation completed successfully.")

if __name__ == "__main__":
    main()