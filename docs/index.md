<!-- markdownlint-disable MD041 -->

![Image title](img/LightLogoSmall.png#only-light)
![Image title](img/DarkLogoSmall.png#only-dark)

# Welcome to Soundscapy

[![PyPI version](https://badge.fury.io/py/soundscapy.svg)](https://badge.fury.io/py/soundscapy)
[![Documentation Status](https://readthedocs.org/projects/soundscapy/badge/?version=latest)](https://soundscapy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

_Soundscapy_ is a Python library for analysing and visualising soundscape assessments following the ISO 12913 standard. It is used by researchers and practitioners to process survey data, create publication-quality visualisations of soundscape perception, analyse binaural recordings, and compute Soundscape Perception Indices.

## What can you do with it?

**Process survey data** — Load, validate, and transform PAQ questionnaire responses into ISO coordinates (`ISOPleasant`, `ISOEventful`) with a single function call. Supports custom value ranges, language-specific angle adjustments, and the ISD, ARAUS, and SATP datasets.

**Visualise soundscapes** — Create scatter plots, bivariate density plots, Likert plots, and radar plots of soundscape perception distributions. The `ISOPlot` class supports multi-panel and multi-layer figures suitable for publication.

**Analyse audio recordings** — Compute psychoacoustic metrics (loudness, sharpness, roughness, fluctuation strength) and environmental indices from binaural WAV files using the `soundscapy[audio]` extension.

**Compute Soundscape Perception Indices** — Fit multi-dimensional skewed normal distributions to perception data and score locations against target distributions using `soundscapy[r]`. Run CircE structural equation models via the SATP module.

## Installation

```bash
pip install soundscapy
```

For audio analysis:

```bash
pip install "soundscapy[audio]"
```

For SPI and SATP (requires a local R installation):

```bash
pip install "soundscapy[r]"
R -q -e "install.packages('sn')"
```

!!! note
    CircE is bundled with _Soundscapy_ — no separate GitHub install needed. Only the `sn` R package is required externally.

To install everything at once:

```bash
pip install "soundscapy[all]"
R -q -e "install.packages('sn')"
```

## Where to start

New to soundscape analysis? Read [About Soundscape Analysis](background.md) for a concise overview of PAQ attributes, the ISO circumplex model, and what ISOPleasant/ISOEventful mean.

Ready to write code? Go to the [Quick Start](tutorials/rendered/QuickStart.md).

## Contributing

We welcome contributions from the community. Please get in touch or submit an issue on [GitHub](https://github.com/MitchellAcoustics/Soundscapy/).

## Citing Soundscapy

!!! note

    If you use _Soundscapy_ in your research, please include a citation to our accompanying paper:

        Mitchell, A., Aletta, F., & Kang, J. (2022). How to analyse and represent quantitative soundscape data. _JASA Express Letters, 2_, 37201. [https://doi.org/10.1121/10.0009794](https://doi.org/10.1121/10.0009794) <!-- markdownlint-disable MD046 -->

## License

This project is licensed under the BSD 3-Clause License. See the [license page](license.md) for details.
