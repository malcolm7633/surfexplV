Lets you see a 2d manifold as if you were an ant in it. Can only handle manifolds with one patch right now. Input functions and conditions as SymPy expressions. Walls are drawn at finite boundaries.

Requires Py-OpenGl, NumPy, SymPy, SciPy.

The tweakable options are up top:

n: number of pixels horizontally and vertically
r: number of rays of light to trace out
fov: field of view
wallh: height of walls drawn at boundaries
turningincrement: how fast you turn when you press arrow keys
movingincrement: how fast you move with WASD
pos: initial (x1,y1)
eyes: initial direction you look (dx1(eyes),dx2(eyes))
perpv: any vector independent from eyes
