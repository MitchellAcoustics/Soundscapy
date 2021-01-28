# Soundscapy

Your friendly little soundscape helper.

## Branches

All of the initial research code was written using functions. I'm planning to convert it to object-oriented programming to massively simplify it. In order to make something available right now, the `functional` branch:

1. is based on (way too many) functions
2. should actually work (heh, functional)

As of writing (2021-01-21), you should only use the `functional` branch.

## Database

The SSID data collection and database structure are based on the SSID Protocol laid out in Mitchell et al. (2020).

The SSID Database follows the same general structure shown in the `test_DB`. Labels for various aspects of the SSID database (i.e. LocationIDs from London, acoustic parameters, etc.) are defined in `ssid.parameters`

```markdown
|-- test_DB
    |-- directoryList.md
    |-- OFF_LocationA1_FULL_2020-12-31
    |   |-- OFF_LocationA1_BIN_2020-12-31
    |       |-- LocationA1_HDF
    |       |-- LocationA1_SpectrumData
    |       |   |-- FFT_Average
    |       |   |   |-- 11-42_1.1_n Octave Spectrum (FFT).csv
    |       |   |   |-- 11-42_1.1_n Octave Spectrum (FFT).csv.hadx
    |       |   |   |-- 11-50_2.1_n Octave Spectrum (FFT).csv
    |       |   |   |-- 11-50_2.1_n Octave Spectrum (FFT).csv.hadx
    |       |   |   |-- 11-52_3.1_n Octave Spectrum (FFT).csv
    |       |   |   |-- 11-52_3.1_n Octave Spectrum (FFT).csv.hadx
    |       |   |   |-- 12-02_4.1_n Octave Spectrum (FFT).csv
    |       |   |   |-- 12-02_4.1_n Octave Spectrum (FFT).csv.hadx
    |       |   |-- FFT_Peak
    |       |-- LocationA1_TimeSeries
    |       |   |-- FluctuationStrength_TS
    |       |   |-- Impulsiveness_TS
    |       |   |-- LevelA_TS
    |       |   |-- LevelC_TS
    |       |   |-- LevelZ_TS
    |       |   |-- Loudness_TS
    |       |   |-- RelativeApproach2D_TS
    |       |   |-- Roughness_TS
    |       |   |-- Sharpness_TS
    |       |   |-- SIL_TS
    |       |   |-- THD_TS
    |       |   |-- TonalityFrequency_TS
    |       |   |-- Tonality_TS
    |       |-- LocationA1_WAV
    |           |-- LA101.wav
    |           |-- LA101.wav.hadx
    |           |-- LA102.wav
    |           |-- LA102.wav.hadx
    |           |-- LA103.wav
    |           |-- LA103.wav.hadx
    |           |-- LA104.wav
    |           |-- LA104.wav.hadx
    |-- OFF_LocationA2_FULL_2021-01-01
    |   |-- OFF_LocationA2_BIN_2021-01-01
    |       |-- LocationA2_HDF
    |       |-- LocationA2_SpectrumData
    |       |   |-- FFT_Average
    |       |   |-- FFT_Peak
    |       |-- LocationA2_TimeSeries
    |       |   |-- FluctuationStrength_TS
    |       |   |-- Impulsiveness_TS
    |       |   |-- LevelA_TS
    |       |   |-- LevelC_TS
    |       |   |-- LevelZ_TS
    |       |   |-- Loudness_TS
    |       |   |-- RelativeApproach2D_TS
    |       |   |-- Roughness_TS
    |       |   |-- Sharpness_TS
    |       |   |-- SIL_TS
    |       |   |-- THD_TS
    |       |   |-- TonalityFrequency_TS
    |       |   |-- Tonality_TS
    |       |-- LocationA2_WAV
    |           |-- LA201.wav
    |           |-- LA201.wav.hadx
    |           |-- LA202.wav
    |           |-- LA202.wav.hadx
    |           |-- LA203.wav
    |           |-- LA203.wav.hadx
    |-- OFF_LocationB1_FULL_2021-01-13
        |-- OFF_LocationB1_BIN_2021-01-13
            |-- LocationB1_WAV
            |   |-- LB101.wav
            |   |-- LB101.wav.hadx
            |   |-- LB102.wav
            |   |-- LB102.wav.hadx
            |   |-- LB103.wav
            |   |-- LB103.wav.hadx
            |   |-- LB104.wav
            |   |-- LB104.wav.hadx
            |-- LocationB2_HDF
            |-- LocationB2_SpectrumData
            |   |-- FFT_Average
            |   |-- FFT_Peak
            |-- LocationB2_TimeSeries
                |-- FluctuationStrength_TS
                |-- Impulsiveness_TS
                |-- LevelA_TS
                |-- LevelC_TS
                |-- LevelZ_TS
                |-- Loudness_TS
                |-- RelativeApproach2D_TS
                |-- Roughness_TS
                |-- Sharpness_TS
                |-- SIL_TS
                |-- THD_TS
                |-- TonalityFrequency_TS
                |-- Tonality_TS

```
