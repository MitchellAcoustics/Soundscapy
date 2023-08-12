<img src="https://raw.githubusercontent.com/MitchellAcoustics/Soundscapy/main/docs/logo/LightLogo.png" width="300">

# Soundscapy

[![Python package](https://github.com/MitchellAcoustics/Soundscapy/actions/workflows/python-package.yml/badge.svg)](https://github.com/MitchellAcoustics/Soundscapy/actions/workflows/python-package.yml)  [![PyPI version](https://badge.fury.io/py/soundscapy.svg)](https://badge.fury.io/py/soundscapy)  [![Documentation Status](https://readthedocs.org/projects/soundscapy/badge/?version=latest)](https://soundscapy.readthedocs.io/en/latest/?badge=latest)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MitchellAcoustics/Soundscapy/main?labpath=docs%2Ftutorials%2FHowToAnalyseAndRepresentSoundscapes.ipynb)

A python library for analysing and visualising soundscape assessments. 

**Disclaimer:** This module is still heavily in development, and might break what you're working on. It will also likely require a decent amount of troubleshooting at this stage. I promise bug fixes and cleaning up is coming!

## Installation

The package is still under development, but can be installed with pip:

```
pip install soundscapy
```

## Examples

We are currently working on writing more comprehensive examples and documentation, please bear with us in the meantime. 

An example notebook which reproduces the figures used in our recent paper "How to analyse and represent quantitative soundscape data" is provided in the `examples` folder.

## Acknowledgements

The newly added Binaural analysis functionality relies directly on three acoustic analysis libraries: 
 * [python-acoustics](https://github.com/python-acoustics/python-acoustics) for the standard environmental and building acoustics metrics, 
 * [scikit-maad](https://github.com/scikit-maad/scikit-maad) for the bioacoustics and ecological soundscape metrics, and
 * [MoSQITo](https://github.com/Eomys/MoSQITo) for the psychoacoustics metrics. We thank each of these packages for their great work in making advanced acoustic analysis more accessible.

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

## Development Plans

As noted, this package is in heavy development to make it more useable, more stable, and to add features and improvements. At this stage it is mostly limited to doing basic quality checks of soundscape survey data and creating the soundscape distribution plots. Some planned improvements are:

 - [x] Simplify the plotting options
 - [x] Possibly improve how the plots and data are handled - a more OOP approach would be good.
 - [ ] Add appropriate tests and documentation.
 - [ ] Bug fixes, ~~particularly around setting color palettes.~~

Large planned feature additions are:

 - [ ] Add better methods for cleaning datasets, including robust outlier exclusion and imputation.
 - [x] Add handling of .wav files.
 - [x] Integrate environmental acoustic and psychoacoustic batch processing. This will involve using existing packages (e.g. MoSQito, python-acoustics) to do the metric calculations, but adding useful functionality for processing many files at once, tieing them to a specific survey response, and implementing a configuration file for maintaining consistent analysis settings.
 - [ ] Integrate the predictive modelling results from the SSID team's research to enable a single pipelined from recording -> psychoacoustics -> predicted soundscape perception (this is very much a pie-in-the-sky future plan, which may not be possible).

### Contributing

If you would like to contribute or if you have any bugs you have found while using `Soundscapy', please feel free to get in touch or submit an issue or pull request!
