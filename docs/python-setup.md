# Python Setup Guide for Soundscapy

This guide provides step-by-step instructions for setting up Python and installing Soundscapy from scratch, designed for users who may not be familiar with Python or virtual environments.

## 1. Install Python

### Windows

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest Python installer for Windows.
2. Run the installer. **Important:** Check the box **"Add python.exe to PATH"** at the bottom of the installer window before clicking "Install Now".
3. Verify the installation by opening Command Prompt or PowerShell and running:
   ```bash
   python --version
   ```
   You should see something like `Python 3.12.x`.

### macOS

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download the macOS installer.
2. Run the `.pkg` file and follow the installation prompts.
3. Verify the installation:
   ```bash
   python3 --version
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

## 2. Create a Virtual Environment

A virtual environment keeps Soundscapy and its dependencies isolated from your other Python projects.

### Windows

```bash
# Create a project folder
mkdir soundscapy-project
cd soundscapy-project

# Create the virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate
```

### macOS / Linux

```bash
# Create a project folder
mkdir soundscapy-project
cd soundscapy-project

# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

After activation, you should see `(.venv)` at the beginning of your terminal prompt.

## 3. Install Soundscapy

With the virtual environment activated, install Soundscapy using pip:

```bash
# Basic installation (survey data processing and visualisation)
pip install soundscapy

# With audio analysis support (psychoacoustic metrics)
pip install "soundscapy[audio]"

# With SPI and SATP support (requires R)
pip install "soundscapy[r]"
R -q -e "install.packages('sn')"

# Everything at once
pip install "soundscapy[all]"
R -q -e "install.packages('sn')"
```

!!! note
    The `[all]` option installs every optional dependency. If you only need survey processing and visualisation, the basic `pip install soundscapy` is sufficient.

## 4. Verify the Installation

Test that Soundscapy is correctly installed:

```python
python -c "import soundscapy; print(soundscapy.__version__)"
```

You should see the version number printed (e.g., `0.8.4`).

## 5. Using Soundscapy with R (Optional)

Some Soundscapy features require R (SPI computation and CircE structural equation models).

### Install R

- **Windows**: Download from [r-project.org](https://cran.r-project.org/bin/windows/base/)
- **macOS**: Download from [r-project.org](https://cran.r-project.org/bin/macosx/)
- **Linux**: `sudo apt install r-base`

### Install the `sn` R Package

```bash
R -q -e "install.packages('sn')"
```

### Verify R Integration

```python
python -c "from soundscapy.satp import CircE; print('R integration OK')"
```

## 6. Common Issues

### `python` command not found

- **Windows**: Re-run the Python installer and ensure "Add to PATH" is checked. You may need to restart your terminal or computer.
- **macOS/Linux**: Try `python3` instead of `python`.

### Permission denied errors during pip install

Make sure your virtual environment is activated (you should see `(.venv)` in your prompt). If you're not using a virtual environment, try:
```bash
pip install --user soundscapy
```

### R package installation fails

Ensure R is installed and `R` is in your system PATH. You can test this by running `R --version` in your terminal.

## 7. Next Steps

- Follow the [Quick Start tutorial](https://drandrewmitchell.com/Soundscapy/tutorials/QuickStart.html)
- Read the [API reference](reference/index.md)
- Learn about [soundscape analysis concepts](background.md)
- [Contribute to Soundscapy](CONTRIBUTING.md)
