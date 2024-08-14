Lets you see a 2d Riemannian manifold as if you were an ant in it. Input functions and conditions as SymPy expressions. Walls are drawn at finite boundaries. It always draws as if the ant's eye is half as tall as the wall (so wall height also changes how far away things look). If it starts with the ant in an invalid position it won't work (for example if the default starting position (0,0) is outside/on the boundary of the image of the chart it starts "in."

Running with no arguments launches the gui. The -i flag runs it in the console instead. Running with a file as the argument launches the file if it exists and runs it in console mode and saves it to the file if it doesn't exist.

Windows users:
The version of Py-OpenGl that pip installs from PyPI requires you to already have GLUT installed. If you don't want to go through installing it yourself there are two options: install from a wheel at https://drive.google.com/drive/folders/1mz7faVsrp0e6IKCQh8MyZh-BcCqEGPwx or run showmeamanifold.py with option "-w" which will try to throw the binary in this directory in the right place in your Py-OpenGL install. If it still doesn't work after the "-w" run, you may have to change the 64 to 32 or vc14 to vc9 or something in the filename.

Requires Py-OpenGL, NumPy, SymPy, SciPy, PyQt5
