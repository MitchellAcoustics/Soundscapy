# API Reference

Curated API reference for Soundscapy's public modules.

| Module | What it does | Requires |
| --- | --- | --- |
| [soundscapy](soundscapy.md) | Top-level re-exports and logging setup | base |
| [Surveys](surveys.md) | PAQ validation, ISO coordinate calculation, ipsatization | base |
| [Plotting](plotting/index.md) | `ISOPlot`, `scatter`, `density`, Likert and radar plots | base |
| [Databases](databases.md) | Load and filter ISD and SATP datasets | base |
| [Audio](audio/index.md) | Binaural audio analysis and psychoacoustic metrics | `soundscapy[audio]` |
| [SPI](spi.md) | Soundscape Perception Indices via Multi-dimensional Skewed Normal | `soundscapy[r]` |
| [SATP](satp.md) | Circumplex SEM validation via CircE (`fit_circe`) | `soundscapy[r]` |
