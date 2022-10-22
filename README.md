# sn_line_velocities
A Python script to fit equivalent width and absorption velocities in SN spectra with multiple Gaussian profiles.

## Package prerequisites
- numpy
- pymc>=4
- corner
- emcee (will be deprecated)

We recommend using Anaconda (or Miniforge) to install Python on your local machine, which allows for packages to be installed using its conda utility. Once you have conda installed, the packages required can be installed into a new conda environment as follows:

```shell
conda create -c conda-forge -n MY_ENV "pymc>=4" arviz corner emcee
conda activate MY_ENV
```

In the previous version, the posteriors are sampled using `emcee`, which will be deprecated soon. The default sampler is the No-U-Turn sampler embedded in `pymc`.

## Get Started
See the notebook `broad_absorption.ipynb`.

