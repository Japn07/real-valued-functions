"""
main.py  –  Entry point for the Real-Valued Functions Applet.

Launches a Tkinter window with three tabs:
  1. f(x,y) Surface / Contour Visualizer
  2. f(x,y,z) Level Surfaces
  3. Calculus Analysis (Gradient, Directional Derivative, Extrema)
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

from tabs.surface_tab import SurfaceTab
from tabs.level_surface_tab import LevelSurfaceTab
from tabs.analysis_tab import AnalysisTab


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def main():
    # Tell Windows this is a distinct app so the taskbar icon works properly
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('my_math_viz_app')
    except Exception:
        pass

    root = tk.Tk()
    root.title("Real-Valued Functions Applet")
    root.geometry("1400x900")

    try:
        root.iconbitmap(default=resource_path('logo.ico'))
    except Exception:
        pass

    # Modernized styling
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Segoe UI', 11, 'bold'), padding=[10, 5])
    except Exception:
        pass

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Tab 1 — f(x,y)
    surface_tab = SurfaceTab(notebook)
    notebook.add(surface_tab, text='1. f(x,y) Visualizer')

    # Tab 2 — f(x,y,z) level surfaces
    level_tab = LevelSurfaceTab(notebook)
    notebook.add(level_tab, text='2. Level Surfaces f(x,y,z)')

    # Tab 3 — Calculus analysis
    analysis_tab = AnalysisTab(notebook)
    notebook.add(analysis_tab, text='3. Calculus Analysis')

    root.mainloop()


if __name__ == '__main__':
    main()
