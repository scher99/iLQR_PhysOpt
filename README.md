# iLQR PhysOpt

A small Python prototype for **trajectory optimization with iLQR** where the plant is not a pure analytic dynamics model.

The system uses:

- a **kinematic bicycle model**
- a **lower-level optimization** step to project the next state onto a feasible set
- **IFT-based local gradients** of that projected plant for use inside iLQR
- obstacle avoidance with circular obstacles and optional wall obstacles

## Overview

This project explores a two-level formulation:

1. **Nominal dynamics**
   - A kinematic bicycle model predicts the next state from the current state and control.

2. **Physics / feasibility layer**
   - The nominal next state is adjusted by solving a constrained optimization problem.
   - This projection enforces obstacle avoidance and physical feasibility.

3. **Gradient extraction**
   - The planner computes local plant Jacobians using an **Implicit Function Theorem (IFT)** style KKT solve.
   - These Jacobians are then used in the iLQR backward pass.

## Features

- iLQR trajectory optimization
- control saturation for speed and steering
- obstacle-aware projected dynamics
- IFT-inspired linearization of the projected plant
- plotting of:
  - straight-line initial guess
  - planned trajectory
  - noisy closed-loop rollout

## Repository Structure

- `main.py` — main script, plant model, planner, and plotting

## Requirements

Install the Python dependencies:

````bash
pip install numpy scipy matplotlib

## Run

From the repository root:

````bash
python3 [main.py](http://_vscodecontentref_/0)
````

A plot window will open showing:

- the initial straight-line path
- the optimized iLQR plan
- a noisy simulated execution
- the goal and obstacles

## Model Summary
State
The vehicle state is:

- x — position in world x
- y — position in world y
- theta — heading angle

Control
The control input is:

- v — forward velocity
- delta — steering angle

Lower-Level Physics
For each step:

- requested controls are clipped to physical bounds
- the bicycle model predicts a nominal next state
- a constrained optimization projects that state away from obstacles

## Obstacles
The current script supports:

* circle obstacles
* wall obstacles via distance-style constraints