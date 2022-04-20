# Soundscapy
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MitchellAcoustics/Soundscapy/main?labpath=examples%2FHowToAnalyseAndRepresentSoundscapes.ipynb) 

A python library for analysing and visualising soundscape assessments. 

**Disclaimer:** This module is still heavily in development, and might break what you're working on. It will also likely require a decent amount of troubleshooting at this stage. I promise bug fixes and cleaning up is coming!

## Installation

For this under-development version, I suggest installing from source in the following way. 

Create a suitable conda environment by downloading just the environment.yml file:
```
conda env create -f environment.yml
conda activate soundscapy-dev
```
Then, install soundscapy from the github source:
```
pip install git+https://github.com/MitchellAcoustics/Soundscapy@main
```

## Examples

An example notebook which reproduces the figures used in our recent paper "How to analyse and represent quantitative soundscape data" is provided in the `examples` folder.

## Citation

If you are using Soundscapy in your research, please help our scientific visibility by citing our work! Please include a citation to our accompanying paper:

Mitchell, A., Aletta, F., & Kang, J. (2022). How to analyse and represent quantitative soundscape data. *JASA Express Letters, 2*, 37201. [https://doi.org/10.1121/10.0009794](https://doi.org/10.1121/10.0009794)


<!---
Bibtex:
```
@Article{Mitchell2022How,
  author         = {Mitchell, Andrew and Aletta, Francesco and Kang, Jian},
  journal        = {JASA Express Letters},
  title          = {How to analyse and represent quantitative soundscape data},
  year           = {2022},
  number         = {3},
  pages          = {037201},
  volume         = {2},
  doi            = {10.1121/10.0009794},
  eprint         = {https://doi.org/10.1121/10.0009794},
}

```
--->
