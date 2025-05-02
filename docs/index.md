<!-- markdownlint-disable MD041 -->

![Image title](img/LightLogoSmall.png#only-light)
![Image title](img/DarkLogoSmall.png#only-dark)

# Welcome to Soundscapy

[![PyPI version](https://badge.fury.io/py/soundscapy.svg)](https://badge.fury.io/py/soundscapy)
[![Documentation Status](https://readthedocs.org/projects/soundscapy/badge/?version=latest)](https://soundscapy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

_Soundscapy_ is a Python library for analysing and visualising soundscape assessments. This package was designed to (1) load and process soundscape assessment data, (2) visualise the data, and (3) enable psychoacoustic analysis of soundscape recordings.

!!! note
This project is still under development. We're working hard to make it as good as possible, but there may be bugs or missing features. If you find any issues, please let us know by submitting an issue on Github.

## Getting Started

To get started with _Soundscapy_, you'll need to install it first. You can do this by running the following command:

```bash
pip install soundscapy
```

### Optional Dependencies

_Soundscapy_ splits its functionality into optional modules to reduce the number of dependencies required for basic functionality. By default, _Soundscapy_ includes the survey data processing and plotting functionality. If you would like to use the binaural audio processing and psychoacoustics functionality, you will need to install the optional `audio` dependency:

```bash
pip install "soundscapy[audio]"
```

## Documentation

This documentation is designed to help you understand and use _Soundscapy_ effectively. It's divided into several sections:

- **Tutorials**: Practical examples showing how to use our project in real-world scenarios.
- **API Reference**: Detailed information about our project's API.
- **News**: Change log and updates.

## Contributing

We welcome contributions from the community. If you're interested in contributing, please get in touch or submit an issue on Github.

## Citing Soundscapy

!!! note
If you use _Soundscapy_ in your research, please include a citation to our accompanying paper:

    Mitchell, A., Aletta, F., & Kang, J. (2022). How to analyse and represent quantitative soundscape data. _JASA Express Letters, 2_, 37201. [https://doi.org/10.1121/10.0009794](https://doi.org/10.1121/10.0009794) <!-- markdownlint-disable MD046 -->

## License

This project is licensed under the BSD 3-Clause License. For more information, please see the `license.md` file.

## Project layout

```plaintext
mkdocs.yml     # The configuration file.
docs/
    index.md   # The documentation homepage.
    about.md   # The about page.
    license.md # The license page.
    tutorials/ # Tutorial pages.
        Introduction to SSM Analysis.ipynb
    ...        # Other markdown pages, images and other files.
src/soundscapy/
```
