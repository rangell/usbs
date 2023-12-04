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

From the root of the repo, the following command with extract the data:
```bash
$ tar xzvf data.tgz
```

### Examples

### Paper Citation