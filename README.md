# LatticeVision (UNDER CONSTRUCTION)

This repository contains all of the accompanying code for the paper: 

*LatticeVision: Image to Image Networks for Modeling Non-Stationary Spatial Data*

**Authors**: Antony Sikorski, Michael Ivanitskiy, Nathan Lenssen, Douglas Nychka, Daniel McKenzie

The paper is currently available on [arXiv](https://arxiv.org/abs/2505.09803).

---

## Installation

Prior to running this code, one will need to download both `R` and `Python`, clone this repository, and install all necessary dependencies. 

- **R:** The `R` programming language may be downloaded [here](https://cran.r-project.org/bin/windows/base/). We strongly recommend downloading [`RStudio`](https://posit.co/download/rstudio-desktop/) to open and work with any of the `R` scripts (training data and synthetic field generation). 

- **Python:** The `Python` programming language may be downloaded [here](https://www.python.org/downloads/). We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) as our package manager. 

- **Cloning this repo:** This repository can be cloned by running `git clone https://github.com/antonyxsik/LatticeVision.git` in your terminal. 

- **Dependencies:** You can install the necessary dependencies and create a virtual environment using [`uv`](https://docs.astral.sh/uv/) by running `uv sync`. If you do not wish to use `uv`, you can create a virtual environment however you wish, and run `pip install -r requirements.txt`. Both install the `latticevision` package, which contains our primary source code. 

## Quick Start 

- Download sample data and model wghts from google drive. all scripts currently point towards sample data.
- run make test to run all of the tests.
- a few additional make commands.
- describe the notebooks and how they provide a good starting point.


## Reproducing our results. 

### Data Generation
- In order to reproduce our results, one must first generate the data.

### Train Models and Test on Synthetic Data
- Run the scripts, these also will give you all of the simulated data results.


### Climate Application
- do the exploring_cesm notebook, then run the R script which does the climate things. 


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

