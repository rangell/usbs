# Spectral Bundle Method with Sketching

Code for TODO: link Fast and Scalable Semidefinite Programming with Warm-Starting

## Overview

SpecBM is an optimization algorithm for solving large semidefinite programs.

## Setting up
This repo requires Python 3.9+. To install Python dependencies, the first package that needs to be installed is [JAX](https://github.com/google/jax) 0.4.13+. Follow the instructions [here](https://jax.readthedocs.io/en/latest/installation.html) to install the version of JAX needed for the desired hardware (CPU/GPU/TPU).

After JAX is installed, the remaining packages can be installed using the following command:
```bash
pip install equinox numpy scipy scikit-learn numba GitPython IPython mat73  
```

## Data
All of the data can be downloaded [here](https://drive.google.com/uc?id=12spD7qZ_6NFVAlTlqxrykGh3VomRHPYy&export=download). The max-cut data was aggregated from [Gset](https://www.cise.ufl.edu/research/sparse/matrices/Gset/) and [DIMACS10](https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/index.html).
The QAP data was aggregated from [QAPLIB](https://qaplib.mgi.polymtl.ca/) and [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/).

After downloading the data and moving it to `$PROJECT_ROOT`, the following command will extract the data into the expected location:
```bash
tar xzvf data.tgz
```

## Examples

The following are example commands to execute for each of the three problem types presented in the paper.

### Max Cut
```bash
PYTHONPATH="." python -u scripts/warm_start_maxcut_specbm.py --data_path=data/maxcut/Gset/G1.mat --max_iters=5000 --max_time=360 --trace_factor=2.0 --rho=0.01 --beta=0.25 --k_curr=10 --k_past=1 --sketch_dim=10 --obj_gap_eps=1e-07 --infeas_gap_eps=1e-07 --max_infeas_eps=1e-07 --subprob_max_iters=100 --subprob_eps=1e-15 --lanczos_max_restarts=10 --warm_start_strategy="none" 
```

### QAP
```bash
PYTHONPATH="." python -u scripts/warm_start_qap_specbm.py --data_path=data/qap/qapdata/chr12a.dat --max_iters=5000 --max_time=360 --trace_factor=2.0 --rho=0.005 --beta=0.25 --k_curr=2 --k_past=0 --obj_gap_eps=1e-07 --infeas_gap_eps=1e-07 --max_infeas_eps=1e-07 --subprob_max_iters=100 --subprob_eps=1e-7 --lanczos_max_restarts=10 --warm_start_strategy="none"
```

### Interactive Entity Resolution with $\exists$-constraints
```bash
PYTHONPATH="." python -u scripts/warm_start_ecc.py --seed=0 --debug --output_dir=test_out --data_path=data/ecc/merged_pubmed_processed.pkl --max_rounds=100 --max_iters=100000 --trace_factor=2.0 --k_curr=3 --k_past=0 --rho=0.01 --beta=0.25 --sketch_dim=-1 --subprob_max_iters=30 --subprob_eps=1e-7 --lanczos_max_restarts=10 --obj_gap_eps=0.1 --infeas_gap_eps=0.1 --max_infeas_eps=0.1
```

## Citing

If you use SpecBM in your work, please cite the following paper:  
```bibtex
TODO: add citation
```
## Questions / Feedback

If you have any questions, comments, or feedback on our work, please reach out at
[rangell@cs.umass.edu](mailto:rangell@cs.umass.edu)! (or open a GitHub issue)

## Licence
SpecBM is MIT licensed. See the [LICENSE](LICENSE) file for details.