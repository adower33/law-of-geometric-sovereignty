# The Law of Geometric Sovereignty (LGS)

This repository contains the numerical verification engine and the official manuscript for the **Law of Geometric Sovereignty**, a single-parameter unified field theory of mass and force.

## Overview

The LGS framework models the vacuum as a recursive manifold governed by discrete scale invariance, anchored to the Golden Ratio ($\varphi$). It demonstrates that physical particle masses emerge as discrete topological resonances (oscillons) within the vacuum field.

This repository provides the computational tools to solve the non-linear, time-dependent radial field equations and extract the resolution-converged core energy invariant:

\[
\langle I_E^{\mathrm{core}} \rangle = 95.06 \pm 0.09
\]

This invariant serves as the single geometric anchor used in the dimensional reconstruction of the Standard Model mass spectrum.

---

## Repository Contents

* `LGS_Paper.pdf`  
  The full manuscript detailing the Lagrangian formulation, Euler–Lagrange derivation, dimensional reconstruction, and falsifiable high-energy predictions (including a 327.4 GeV dark matter candidate).

* `lgs_plateau_oscillon_sweep.py`  
  The high-resolution finite-difference solver used to integrate the nonlinear radial field equation and extract the plateau oscillon energy invariant.

---

## Reproducibility & Code Execution

The numerical invariant reported in Section 5 of the manuscript can be reproduced exactly using the command below.

python lgs_plateau_oscillon_sweep.py --Amin 0.758824 --Amax 0.758824 --nA 1 --sigmin 9.058824 --sigmax 9.058824 --nsig 1 --dt 0.008 --dr 0.07 --tmax 3500 --plot_best

### Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy scipy matplotlib

**Expected Output:**

The script will integrate the dimensionless PDE using a Sommerfeld outgoing-wave boundary condition and apply a strict plateau criterion. 

Because this script runs the refined high-resolution grid ($\Delta\rho=0.07$), the terminal output will read:
`Core Energy Invariant: 94.971...`

*Note: In the official manuscript, this high-resolution run is averaged with the baseline run to produce the resolution-converged quoting of 95.06 ± 0.09.*
