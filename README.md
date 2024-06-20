Lets you see a 2d Riemannian manifold as if you were an ant in it. Input functions and conditions as SymPy expressions. Walls are drawn at finite boundaries. It always draws as if the ant's eye is half as tall as the wall (so wall height also changes how far away things look). If it starts with the ant in an invalid position it won't work (for example if the default starting position (0,0) is outside/on the boundary of the image of the chart it starts "in."

Running with no arguments launches the gui. The -i flag runs it in the console instead. Running with a file as the argument launches the file if it exists and runs it in console mode and saves it to the file if it doesn't exist.

Requires Py-OpenGl, NumPy, SymPy, SciPy, PyQt5
