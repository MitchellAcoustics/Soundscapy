# Refactored Plotting Module

This directory contains a refactored implementation of the plotting functionality in soundscapy. The refactoring focuses on:

1. Using composition instead of inheritance
2. Simplifying the relationships between components
3. Consolidating parameter models
4. Improving type safety

## Architecture

The new architecture consists of the following components:

### Core Components

- **ISOPlot**: The main entry point for creating plots. Uses composition to delegate functionality to specialized managers.
- **PlotContext**: Manages data, state, and parameters for a plot or subplot. The central component that owns parameter models.
- **Layer**: Base class for visualization layers. Layers know how to render themselves onto a PlotContext's axes.

### Managers

- **LayerManager**: Manages the creation and rendering of visualization layers.
- **StyleManager**: Manages the styling of plots.
- **SubplotManager**: Manages the creation and configuration of subplots.

### Parameter Models

- **BaseParams**: Base model for all parameter types.
- **AxisParams**: Parameters for axis configuration.
- **SeabornParams**: Base parameters for seaborn plotting functions.
- **ScatterParams**: Parameters for scatter plot functions.
- **DensityParams**: Parameters for density plot functions.
- **SimpleDensityParams**: Parameters for simple density plots.
- **SPISeabornParams**: Base parameters for SPI plotting functions.
- **SPISimpleDensityParams**: Parameters for SPI simple density plots.
- **StyleParams**: Configuration options for styling circumplex plots.
- **SubplotsParams**: Parameters for subplot configuration.

### Layer Types

- **ScatterLayer**: Layer for rendering scatter plots.
- **DensityLayer**: Layer for rendering density plots.
- **SimpleDensityLayer**: Layer for rendering simple density plots.
- **SPISimpleLayer**: Layer for rendering SPI simple density plots.

### Protocols

- **RenderableLayer**: Protocol defining what a renderable layer must implement.
- **ParameterProvider**: Protocol defining how parameters are provided.
- **ParamModel**: Protocol defining the interface for parameter models.
- **PlotContext**: Protocol defining the interface for plot contexts.

## Usage

The new implementation maintains the same public API as the original, so existing code should continue to work with minimal changes:

```python
from soundscapy.plotting.new import ISOPlot

# Create a plot
plot = ISOPlot(data=data, hue="LocationID")

# Add layers
plot.create_subplots()
plot.add_scatter()
plot.add_density()
plot.apply_styling()

# Show the plot
plot.show()
```

## Benefits of the New Architecture

1. **Clearer Separation of Concerns**: Each component has a well-defined responsibility.
2. **Reduced Coupling**: Components are less tightly coupled, making the code more maintainable.
3. **Improved Type Safety**: Better use of type hints and protocols for structural typing.
4. **More Flexible Composition**: Easier to extend with new layer types and customize behavior.
5. **Reduced Duplication**: Single source of truth for parameters.
6. **Simplified Testing**: Components can be tested in isolation.

## Implementation Notes

- The refactored code is in a separate directory to avoid breaking existing code.
- The parameter models use Pydantic for validation, maintaining the type safety of the original implementation.
- The layer system has been simplified, with a focus on using parameters from the context.
- The ISOPlot class uses composition instead of inheritance, delegating functionality to specialized managers.