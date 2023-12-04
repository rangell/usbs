# Spectral Bundle Method with Sketching



### Installation
This repo requires Python 3.9+. To install Python dependencies, the first package that needs to be installed is [JAX](https://github.com/google/jax) 0.4.13+. Follow the instructions [here](https://jax.readthedocs.io/en/latest/installation.html) to install the version of JAX needed for the desired hardware (CPU/GPU/TPU).

After JAX is installed, the remaining packages can be installed using the following command:
```bash
$ pip install equinox numpy scipy scikit-learn numba GitPython IPython mat73  
```

### Data
All of the data can be accessed [here](https://drive.google.com/uc?id=12spD7qZ_6NFVAlTlqxrykGh3VomRHPYy&export=download). The max-cut data was aggregated from [Gset](https://www.cise.ufl.edu/research/sparse/matrices/Gset/) and [DIMACS10](https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/index.html).
The QAP data was aggregated from [QAPLIB](https://qaplib.mgi.polymtl.ca/) and [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/).

From the root of the repo, the following command will extract the data:
```bash
$ tar xzvf data.tgz
```

### Examples

##### Max Cut
```bash
$ PYTHONPATH="." python scripts/warm_start_maxcut_specbm.py --data_path=data/maxcut/Gset/G1.mat --max_iters=5000 --max_time=360 --trace_factor=2.0 --rho=0.01 --beta=0.25 --k_curr=5 --k_past=0 --sketch_dim=10 --obj_gap_eps=1e-07 --infeas_gap_eps=1e-07 --max_infeas_eps=1e-07 --subprob_max_iters=100 --subprob_eps=1e-15 --lanczos_max_restarts=10 --warm_start_strategy="none" 
```

##### QAP
```bash
$ PYTHONPATH="." python scripts/warm_start_qap_specbm.py --data_path=data/qap/qapdata/chr12a.dat --max_iters=5000 --max_time=360 --trace_factor=2.0 --rho=0.01 --beta=0.25 --k_curr=5 --k_past=0 --obj_gap_eps=1e-07 --infeas_gap_eps=1e-07 --max_infeas_eps=1e-07 --subprob_max_iters=100 --subprob_eps=1e-15 --lanczos_max_restarts=100 --warm_start_strategy="none"
```

### Paper Citation