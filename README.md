# **Infrastructure Reliability & Maintenance Simulation**

A specialized Python framework that combines **Hierarchical Bayesian Inference** with **Discrete Event Simulation (DES)** to model infrastructure reliability (e.g., locks, bridges) under conditions of scarce data.

## **Overview**

This project addresses a common challenge in civil engineering: estimating failure rates for assets with limited historical failure data. Instead of relying on simple averages or guesses, this tool uses:

1. **Hierarchical Bayesian Modeling (HBM):** Borrows statistical strength from the entire population of assets to estimate individual failure rates (Poisson-Gamma shrinkage).  
2. **Discrete Event Simulation (DES):** Simulates the operational lifecycle, competing failure risks (e.g., pumps vs. gates), and resource contention (repair crews) over time.

## **Project Structure**

.  
├── des\_reliability/  
│   ├── \_\_init\_\_.py  
│   ├── data.py           \# Generates synthetic maintenance logs  
│   └── probability.py    \# PyMC models (JAX/NumPyro) & SimPy Asset logic  
├── simulation\_example.py \# Main execution script  
└── README.md

## **Prerequisites & Installation**

This project relies on the **JAX** backend for PyMC to ensure fast sampling.

### **Using uv (Recommended)**

uv sync 

### **Using pip**

pip install -e .

*Note: You may need to install a C++ compiler (g++) if not using the JAX backend, though the current code is configured for JAX/NumPyro.*

## **Usage**

Run the main simulation script to execute the full workflow:  
uv run src/des_reliability/cli.py

### **What happens during execution?**

1. **Data Generation:** Creates synthetic historical logs for N assets (default: 50\) over a 5-year observation period.  
2. **Bayesian Learning:** Runs a MCMC sampler (NUTS via NumPyro) to estimate the latent failure rate ($\\lambda$) for each asset. You will see shrinkage effects where data-poor assets are pulled towards the population mean.  
3. **Simulation:** Launches a SimPy environment where assets operate and fail according to competing risks (Pump, Gate, Control System). Maintenance crews repair assets based on availability.

## **Methodology**

### **1\. The "Scarce Data" Solution**

We utilize a **Poisson-Gamma Conjugate** model structure:

* **Likelihood:** $Y\_i \\sim \\text{Poisson}(\\lambda\_i \\cdot t)$  
* **Prior:** $\\lambda\_i \\sim \\text{Gamma}(\\alpha, \\beta)$  
* **Hyperprior:** $\\alpha, \\beta$ are learned from the global population.

### **2\. Competing Risks**

Assets do not fail as a monolith. The simulation models a series system where the first component to fail triggers the asset downtime:

* **Pump:** 60% weight  
* **Lock Gate:** 30% weight  
* **Control System:** 10% weight