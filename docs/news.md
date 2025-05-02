# News

## [2024-08-15] Significant Enhancements to Soundscapy's Plotting Module

I am pleased to announce a comprehensive update to Soundscapy's plotting module, introducing enhanced flexibility, improved performance, and more extensive customization options for soundscape visualizations. This update represents a substantial improvement in our toolkit's capabilities.

### Modular and Extensible Architecture

The plotting module has undergone a complete restructuring, resulting in a more modular and maintainable codebase. This new architecture not only facilitates easier maintenance but also establishes a robust foundation for future enhancements and extensions.

### Multi-Backend Support

A key feature of this update is the introduction of multi-backend support. While Seaborn remains our primary plotting engine, we have integrated experimental support for Plotly. This addition enables the creation of interactive plots suitable for web-based applications, alongside our traditional static plots, providing users with increased flexibility in their visualization choices.

### Enhanced Customization

We have introduced a new `CircumplexPlot` class that serves as the central mechanism for creating and customizing plots. Complementing this, we've developed a `StyleOptions` class that offers granular control over visualization aesthetics. These additions allow for precise adjustments to plot elements, such as z-order modification and kernel density estimation bandwidth tuning.

### Streamlined API with Dual Interfaces

While we've significantly expanded the capabilities of our plotting module, we've maintained a focus on user accessibility. We now offer two primary interfaces for plot creation:

- Function-based Interface: The `scatter_plot()` and `density_plot()` functions remain available and have been optimized to leverage the new CircumplexPlot class internally. These functions offer a straightforward method for creating standard plots with minimal code.
- Class-based Interface: For users requiring more advanced customization, the `CircumplexPlot` class provides direct access to a wide array of plotting options and methods.

This dual approach ensures that both newcomers and advanced users can efficiently create the visualizations they need.

### Multiple Plot Creation

We've introduced a new `create_circumplex_subplots()` function, designed to simplify the process of creating multiple related plots. This function is particularly useful for comparing soundscapes across different locations or time periods, allowing for easy creation of grid-based visualizations.

### Future Developments

Our development roadmap includes several exciting features:

- Implementation of joint plots for the Seaborn backend
- Further improvements to the Plotly backend, including support for additional plot types and customization options
- Ongoing performance optimizations

### Upgrading and Breaking Changes

It's important to note that this update introduces breaking changes that will require modifications to existing code. The primary areas affected are:

- Import statements: The module structure has changed, necessitating updates to import statements. For example:

  ```python
  from soundscapy.plotting import scatter_plot, density_plot, Backend
  ```

- Function names and parameters: Some function names and their parameters have been modified for consistency and clarity. Please refer to the updated documentation for specific changes.
- Class-based interface: If you were previously using lower-level plotting functions, you may need to transition to the new CircumplexPlot class for advanced customizations.

We strongly recommend reviewing the updated documentation thoroughly when upgrading to this new version. While these changes may require some code adjustments, we believe the improved functionality and flexibility justify the effort.

We are confident that these improvements to the Soundscapy plotting module will significantly enhance your ability to create insightful and visually appealing soundscape visualizations. We look forward to seeing the innovative ways in which our user community will leverage these new capabilities.
