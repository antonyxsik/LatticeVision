# LatticeVision

This repository contains all of the accompanying code for the paper: 

*LatticeVision: Image to Image Networks for Modeling Non-Stationary Spatial Data*

**Authors**: Antony Sikorski, Michael Ivanitskiy, Nathan Lenssen, Douglas Nychka, Daniel McKenzie

The paper is currently available on [arXiv](https://arxiv.org/abs/2505.09803).

## Abstract

> In many scientific and industrial applications, we are given a handful of instances (a 'small ensemble') of a spatially distributed quantity (a 'field') but would like to acquire many more. For example, a large ensemble of global temperature sensitivity fields from a climate model can help farmers, insurers, and governments plan appropriately. When acquiring more data is prohibitively expensive -- as is the case with climate models -- statistical emulation offers an efficient alternative for simulating synthetic yet realistic fields. However, parameter inference using maximum likelihood estimation (MLE) is computationally prohibitive, especially for large, non-stationary fields. Thus, many recent works train neural networks to estimate parameters given spatial fields as input, sidestepping MLE completely. In this work we focus on a popular class of parametric, spatially autoregressive (SAR) models. We make a simple yet impactful observation; because the SAR parameters can be arranged on a regular grid, both inputs (spatial fields) and outputs (model parameters) can be viewed as images. Using this insight, we demonstrate that image-to-image (I2I) networks enable faster and more accurate parameter estimation for a class of non-stationary SAR models with unprecedented complexity.

---

# Installation

## R (data generation)



## Python (Model training and inference)

You can install the necessary dependencies and create a virtual environment using [`uv`](https://docs.astral.sh/uv/) by running `uv sync`.

If you do not wish to use `uv`, you can create a virtual environment however you wish, and run `pip install -r requirements.txt`

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

