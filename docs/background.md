# About Soundscape Analysis

## What is a soundscape?

A soundscape is not simply the sound level at a location — it is the acoustic environment as perceived and experienced by a person in context. Two places with identical noise levels can feel completely different: a busy park might feel vibrant and pleasant while a road junction with the same level feels chaotic and stressful. Soundscape research, formalised in [ISO 12913](https://www.iso.org/standard/52161.html), aims to capture this perceptual reality rather than reducing the environment to a single physical measurement.

The standard defines a structured questionnaire approach (Method A) in which participants rate a soundscape against eight **Perceived Affective Quality (PAQ)** attributes:

| Attribute | Opposite |
|-----------|----------|
| Pleasant | Annoying |
| Vibrant | Monotonous |
| Eventful | Uneventful |
| Chaotic | Calm |

Each attribute is rated on a five-point Likert scale (1 = Strongly disagree, 5 = Strongly agree). The eight items are independent in the questionnaire but form a theoretically motivated structure — the **circumplex model** — in which they are arranged at 45° intervals around two orthogonal axes.

## The ISO 12913 circumplex model

The circumplex model maps the eight PAQ ratings onto two summary dimensions:

- **ISOPleasant** — the pleasant–unpleasant axis (x-axis in Soundscapy plots)
- **ISOEventful** — the eventful–uneventful axis (y-axis)

These are calculated by projecting the PAQ ratings trigonometrically onto the two axes:

$$
\text{ISOPleasant} = \frac{(\text{pleasant} - \text{annoying}) + \cos 45° \cdot [(\text{calm} - \text{chaotic}) + (\text{vibrant} - \text{monotonous})]}{4 + \sqrt{32}}
$$

$$
\text{ISOEventful} = \frac{(\text{eventful} - \text{uneventful}) + \cos 45° \cdot [(\text{chaotic} - \text{calm}) + (\text{vibrant} - \text{monotonous})]}{4 + \sqrt{32}}
$$

The denominator scales the result to the range \([-1, +1]\). A point at (+1, 0) represents a maximally pleasant, neutral-eventfulness soundscape; a point at (−1, +0.5) represents an unpleasant, moderately eventful one (like a busy road).

Soundscapy computes these coordinates with a single call:

```python
import soundscapy as sspy

data = sspy.isd.load()
data = sspy.add_iso_coords(data)
# → adds "ISOPleasant" and "ISOEventful" columns
```

For a deeper treatment of the formula and why a distributional representation of responses is preferred over single-point summaries, see [Analysing and Representing Soundscapes](tutorials/rendered/HowToAnalyseAndRepresentSoundscapes.md).

## Why distributions, not means

A single mean ISOPleasant value conceals important information: whether a location is perceived similarly by everyone, or whether responses are split between those who find it pleasant and those who do not. Soundscapy is built around the insight that soundscape assessment data should be treated as a **distribution** of responses, not a single point.

This shapes everything in the library — from the density plot visualisations (which show the full bivariate distribution) to the Soundscape Perception Index (which quantifies similarity between distributions).

## What Soundscapy provides

Soundscapy is organised into capability tiers based on dependencies:

| Capability | Install | Key entry points |
|------------|---------|-----------------|
| Survey data processing | `pip install soundscapy` | `add_iso_coords()`, `ipsatize()`, `isd.load()` |
| Visualisation | `pip install soundscapy` | `ISOPlot`, `scatter()`, `density()`, `paq_likert()` |
| Psychoacoustic audio analysis | `pip install "soundscapy[audio]"` | `Binaural`, `AudioAnalysis` |
| Soundscape Perception Index | `pip install "soundscapy[r]"` + R + `sn` | `spi.MultiSkewNorm` |
| CircE structural equation model | `pip install "soundscapy[r]"` + R + `sn` | `satp.fit_circe()` |

## Where to go next

- **Run code immediately** → [Quick Start](tutorials/rendered/QuickStart.md)
- **Step-by-step introduction** → [Getting Started](tutorials/rendered/0_QuickStart_for_Beginners.md)
- **The distributional analysis approach** → [Analysing and Representing Soundscapes](tutorials/rendered/HowToAnalyseAndRepresentSoundscapes.md)
- **Full API documentation** → [API Reference](reference/index.md)
