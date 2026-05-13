"""
build.py  –  PyInstaller build script for the Real-Valued Functions Applet.

Run:  python build.py
"""
import PyInstaller.__main__

print("Starting compilation of Real-Valued Functions Applet...")

PyInstaller.__main__.run([
    'main.py',
    '--name=Real_Valued_Functions_Applet',
    '--windowed',
    '--onefile',
    '--noconfirm',
    '--icon=logo.ico',
    '--add-data=logo.ico;.',
    '--hidden-import=numpy',
    '--hidden-import=matplotlib',
    '--hidden-import=scipy',
    '--hidden-import=skimage',
])

print("\nBuild Complete! Executable is in the 'dist/' folder.")
