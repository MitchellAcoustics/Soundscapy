# Soundscapy
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MitchellAcoustics/Soundscapy/main?labpath=examples%2FHowToAnalyseAndRepresentSoundscapes.ipynb) 

A python library for analysing and visualising soundscape assessments. 

**Disclaimer:** This module is still heavily in development, and might break what you're working on.

## Installation

For this under-development version, I suggest installing from source in the following way. 

Create a suitable conda environment by downloading just the environment.yml file:
```
conda env create -f environment.yml
conda activate soundscapy-dev
```
Then, install soundscapy from the github source:
```
pip install git+git://github.com/MitchellAcoustics/Soundscapy@main
```

## Examples

An example notebook which reproduces the figures used in our recent paper "How to analyse and represent quantitative soundscape data" is provided in the `examples` folder.

## Citation

If you are using Soundscapy in your research, please help our scientific visibility by citing our work! Please include a citation to our accompanying paper:

Mitchell, A., Aletta, F., & Kang, J. (Accepted 2022). How to analyse and represent quantitative soundscape data. *JASA Express Letters*

<!---
Bibtex:
```
@Article{Mitchell2022analyse,
  author    = {Andrew Mitchell and Francesco Aletta and Jian Kang},
  journal   = {JASA Express Letters},
  title     = {How to analyse and represent quantitative data},
  year      = {2022},
  publisher = {Acoustical Society of America ({ASA})},
}
```
--->
