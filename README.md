# Real-Valued Functions Explorer

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![GUI](https://img.shields.io/badge/GUI-Tkinter-lightgrey)
![Math](https://img.shields.io/badge/Math-Matplotlib%20%7C%20SciPy-orange)
A powerful, interactive Python desktop application for visualizing and analyzing mathematical functions in both two and three dimensions. Built with `tkinter` and `matplotlib`, this tool serves as an interactive sandbox for students, educators, and mathematics enthusiasts exploring multivariable calculus, geometry, and real-valued functions.

## Features

- **f(x,y) Visualizer:** Render 3D surface plots, level curves, and contour lines for multivariable functions. Supports implicit curves, precise cross-section slicing along the x and y axes, vector-directional slices, and real-time hover value readouts.
- **f(x,y,z) Level Surfaces:** Visualize 3D level surfaces (isosurfaces) like $x^2 + y^2 + z^2 = c$ using the Marching Cubes algorithm. Slice along arbitrary arbitrary unit vectors $\hat{u}$ to explore cross-sections.
- **Calculus Analysis:** Perform live calculus computations. Visualizes the gradient field (quiver plot), directional derivatives, and automatically detects and classifies critical points (minima, maxima, and saddle points) using a squared gradient norm minimization solver.
- **Domain & Expression Parsing:** Smart input parsing allows evaluating mathematical expressions (e.g., `pi`, `e`, `sqrt(2)`) directly into domain boundaries and directional vectors.

## Dependencies

Ensure you have Python installed along with the following packages:
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-image` (Required for 3D marching cubes rendering in the Level Surfaces tab)

You can install them via pip:
```bash
pip install numpy scipy matplotlib scikit-image
```

## Running the Application

To launch the visualizer, execute `main.py` via Python:
```bash
python main.py
```

## Project Architecture
This application was designed and architected by Japn07 (Project Architect) and developed with the assistance of AI pair programmers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
