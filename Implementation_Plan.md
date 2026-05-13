# Real-Valued Functions Applet

## Key Features
- Plot Surface Plots, Level Curves and Contour Plots for $f(x,y)$
  - Ability to plot multiple functions simultaneously
  - Use PyTorch for differentiation
  - Ability to slice along a given axes or unit vector (Optional)
  - Include sliders for x and y variables
- Plot Level Surfaces for $f(x,y,z)$
    - Ability to slice along a chosen axis
- Compute Directional Derivatives
- Plot Gradient Vectors
- Include information on absolute/ local extrema or saddle points
- Include info on functions value on hover

## Supported Functions
- $\text{sqrt}(x)$
- $\ln(x)$
- $\log(x,y)$
- $e**x$
- $\sin(x), \, \cos(x), \, \tan(x)$
- $\arcsin(x), \, \arccos(x), \, \arctan(x)$
- $\sinh(x), \, \cosh(x), \, \tanh(x)$

## Execptions
- Handle division by zero as discontinuities
- Treat Complex solutions as outside of the valid domain
- Treat negtive logarithms as outside of the valid domain