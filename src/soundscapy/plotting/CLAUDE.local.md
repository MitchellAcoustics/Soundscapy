# Soundscapy Plotting Module Development Guide

- Maintain consistency with the general design principles of Soundscapy given in `CLAUDE.md` and `design.md`

## Plotting Module Design

- User-facing API should have explicit parameters (avoid **kwargs where possible)
- Use element-based architecture inspired by grammar of graphics
- Support combinations of plot elements (scatter, density, marginals)
- Support consistent grouping via 'hue' parameter across all elements
- Use explicit parameters named for clarity (e.g., scatter_alpha, density_fill)
- Maintain backend-independent interface with consistent return types
- High-level API functions should be simple with good defaults
- Use protocols for backend interfaces rather than inheritance
- Keep styling separate from data visualization
- Provide convenience functions for common use cases