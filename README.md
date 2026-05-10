![Soundscapy Logo](https://raw.githubusercontent.com/MitchellAcoustics/Soundscapy/main/docs/img/LightLogo.png) <!-- markdownlint-disable-line MD041 -->

# Soundscapy

[![PyPI version](https://badge.fury.io/py/soundscapy.svg)](https://badge.fury.io/py/soundscapy)
[![Tests](https://github.com/MitchellAcoustics/Soundscapy/actions/workflows/test.yml/badge.svg)](https://github.com/MitchellAcoustics/Soundscapy/actions/workflows/test.yml)
![License](https://img.shields.io/github/license/MitchellAcoustics/Soundscapy)

A python library for analysing and visualising soundscape assessments.

## Installation

Soundscapy can be installed with pip:

```bash
pip install soundscapy
```

> **Upgrading from 0.7.x?** `v0.8.2` is the long-overdue stable release that
> consolidates the entire `v0.8.0rc*` and `v0.8.2.dev*` pre-release line.
> Several APIs have changed (most notably the new `ISOPlot` plotting API).
> See the [migration guide](https://drandrewmitchell.com/Soundscapy/migration-0.7-to-0.8/)
> before upgrading.

### Optional Dependencies

Soundscapy splits its functionality into optional modules to reduce the number of dependencies required for basic functionality. By default, Soundscapy includes the survey data processing and plotting functionality.

If you would like to use the binaural audio processing and psychoacoustics functionality, you will need to install the optional `audio` dependency:

```bash
pip install "soundscapy[audio]"
```

If you would like to use the R-backed SPI and SATP functionality, install the optional `r` dependency:

```bash
pip install "soundscapy[r]"
```

R-backed features also require a local R installation and the external `sn` R package:

```bash
R -q -e "install.packages('sn')"
```

CircE is bundled with Soundscapy as embedded R scripts, so you do not need to install CircE separately from GitHub.

To install all optional dependencies, use the following command:

```bash
pip install "soundscapy[all]"
```

## Examples

We are currently working on writing more comprehensive examples and documentation, please bear with us in the meantime.

Tutorials for using Soundscapy can be found in the [documentation](https://soundscapy.readthedocs.io/en/latest/).

## Acknowledgements

The newly added Binaural analysis functionality relies directly on three acoustic analysis libraries:

- [Acoustic Toolbox](https://github.com/Universite-Gustave-Eiffel/acoustic-toolbox) for the standard environmental and building acoustics metrics,
- [scikit-maad](https://github.com/scikit-maad/scikit-maad) for the bioacoustics and ecological soundscape metrics, and
- [MoSQITo](https://github.com/Eomys/MoSQITo) for the psychoacoustics metrics. We thank each of these packages for their great work in making advanced acoustic analysis more accessible.

## Citation

If you are using Soundscapy in your research, please help our scientific visibility by citing our work! Please include a citation to our accompanying paper:

Mitchell, A., Aletta, F., & Kang, J. (2022). How to analyse and represent quantitative soundscape data. _JASA Express Letters, 2_, 37201. [https://doi.org/10.1121/10.0009794](https://doi.org/10.1121/10.0009794)

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

Soundscapy is under active development as we add features and refine the API. The package now covers survey processing, the new layered `ISOPlot` plotting API, binaural and psychoacoustic audio analysis, and R-backed circumplex SEM (SATP) and Soundscape Perception Indices (SPI) modules. Some planned improvements are:

- [x] Simplify the plotting options
- [x] Possibly improve how the plots and data are handled - a more OOP approach would be good.
- [x] Add appropriate tests and documentation.
- [ ] Bug fixes, ~~particularly around setting color palettes.~~

Large planned feature additions are:

- [ ] Add better methods for cleaning datasets, including robust outlier exclusion and imputation.
- [x] Add handling of .wav files.
- [x] Integrate environmental acoustic and psychoacoustic batch processing. This will involve using existing packages (e.g. MoSQito, python-acoustics) to do the metric calculations, but adding useful functionality for processing any files at once, tieing them to a specific survey response, and implementing a configuration file for maintaining consistent analysis settings.
- [ ] Integrate the predictive modelling results from the SSID team's research to enable a single pipelined from recording -> psychoacoustics -> predicted soundscape perception (this is very much a pie-in-the-sky future plan, which may not be possible).

### Contributing

If you would like to contribute or if you have any bugs you have found while using `Soundscapy', please feel free to get in touch or submit an issue or pull request!

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
