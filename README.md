# LatticeVision

This repository contains all of the accompanying code for the paper: 

*LatticeVision: Image to Image Networks for Modeling Non-Stationary Spatial Data*

**Authors**: Antony Sikorski, Michael Ivanitskiy, Nathan Lenssen, Douglas Nychka, Daniel McKenzie

The paper is currently available on [arXiv](https://arxiv.org/abs/2505.09803).

---

# Getting Started

Prior to running this code, one will need to download both `R` and `Python`, clone this repository, and install all necessary dependencies. 

The `R` programming language may be downloaded [here](https://cran.r-project.org/bin/windows/base/). We strongly recommend downloading [`RStudio`](https://posit.co/download/rstudio-desktop/) to open and work with any of the `R` scripts (training data and synthetic field generation). 

The `Python` programming language may be downloaded [here](https://www.python.org/downloads/). We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) as our package manager. 

You can install the necessary dependencies and create a virtual environment using [`uv`](https://docs.astral.sh/uv/) by running `uv sync` in your terminal.

If you do not wish to use `uv`, you can create a virtual environment however you wish, and run `pip install -r requirements.txt`. Both of these will install the package `latticevision`, which contains our primary source code. 

## R (data generation)



## Python (Model training and inference)

You can install the necessary dependencies and create a virtual environment using [`uv`](https://docs.astral.sh/uv/) by running `uv sync`.

If you do not wish to use `uv`, you can create a virtual environment however you wish, and run `pip install -r requirements.txt`

---

## Citation

Please use the following BibTeX to cite this work: 

```{bibtex}
@article{sikorski2025latticevision,
  title={LatticeVision: Image to Image Networks for Modeling Non-Stationary Spatial Data},
  author={Sikorski, Antony and Ivanitskiy, Michael and Lenssen, Nathan and Nychka, Douglas and McKenzie, Daniel},
  journal={arXiv preprint arXiv:2505.09803},
  year={2025}
}
```

