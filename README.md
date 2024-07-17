# Unified Spectral Bundling with Sketching

Code for the paper:

> [Fast, Scalable, Warm-Start Semidefinite Programming with Spectral Bundling and Sketching](https://arxiv.org/abs/2312.11801)\
> Rico Angell and Andrew McCallum\
> _arXiv:2312.11801_

## Overview

USBS is an optimization algorithm for solving large semidefinite programs.

## Setting up
This repo was developed with Python 3.12.4. To get started we recommend creating a fresh virutal environment using virtualenv, conda, or mamba. The main package that needs to be installed is [JAX](https://github.com/google/jax) (this repo was tested with version 0.4.31.dev20240701). Follow the instructions [here](https://jax.readthedocs.io/en/latest/installation.html) to install the version of JAX needed for the desired hardware (CPU/GPU/TPU). Although it is likely unnecessary for most users, we download and build JAX from source using the instructions [here](https://jax.readthedocs.io/en/latest/developer.html#building-from-source).

We build JAX from source in order to run USBS on a single NVIDIA A100 GPU which uses different versions of CUDA and CUDNN than the pre-built versions of JAX available [here](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html). We see a massive performance improvement from using a GPU, and thus, were inclined to build JAX from source. In the case that you have access to a GPU and want to get the most out of USBS, we provide some tips for building JAX from source for GPU use to augment the official build instructions. As a note, we built JAX on a Linux machine running Ubuntu 20.04.6 LTS. You should be able to ignore the following tips if there exists a pre-built version of JAX that suites your available hardware.

1) Before building, we add the file paths to both CUDA and CUDNN to the enviornment variable `LD_LIBRARY_PATH` using the following commands:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuda/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cudnn/
```

2) We found success building `jaxlib` using the `clang` compiler instead of `gcc`. It seems that this has something to due with builing XLA (see [this](https://openxla.org/xla/build_from_source)). Assuming `clang` is already installed on your system you can use it to build `jaxlib` using `clang` by adding the `--use_clang` flag to the end of the `python build/build.py ... ` command.

3) We found that after `jax` and `jaxlib` are built and installed into the virtual environment we needed to install the `bitsandbytes` package using the following command:
```bash
pip install -U bitsandbytes
```

After JAX is installed, the remaining packages can be installed using the following command:
```bash
pip install numpy scipy scikit-learn numba GitPython IPython mat73  
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

If you use USBS in your work, please cite the following paper:  
```bibtex
@misc{angell2023fast,
      title={Fast, Scalable, Warm-Start Semidefinite Programming with Spectral Bundling and Sketching}, 
      author={Rico Angell and Andrew McCallum},
      year={2023},
      eprint={2312.11801},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```
## Questions / Feedback

If you have any questions, comments, or feedback on our work, please reach out at
[rangell@cs.umass.edu](mailto:rangell@cs.umass.edu)! (or open a GitHub issue)

## Licence
USBS is MIT licensed. See the [LICENSE](LICENSE) file for details.
