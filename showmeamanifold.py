from OpenGL.GLUT import *
from OpenGL.GL import *
import numpy as np
import sympy as sym
import scipy.integrate
from numpy import sqrt

n = 800
r = 40
fov = 1
wallh = .2
turningincrement = np.pi/16
movingincrement = .05

global pos
global eyes
global perpv
pos = np.array([0,0]).T
eyes = np.array([0,1]).T
perpv = np.array([1,0]).T

def whatmelook():
    calcs = []
    for theta in thetas:
        vec = np.cos(theta)*eyes+np.sin(theta)*perpv
        integrator = scipy.integrate.LSODA(fun=model, t0=0, y0=[*pos,*vec],t_bound=5,max_step=.05,jac=npJac)
        while integrator.status == "running":
            integrator.step()
            if not regioncheck(integrator.y[0],integrator.y[1]):
                calcs.append((integrator.t,1))
                break
        if not integrator.status == "running":
            calcs.append((0,0))
    return calcs
def turning(key,x,y):
    global eyes
    global perpv
    if key == GLUT_KEY_RIGHT:
        eyes = np.cos(turningincrement)*eyes+np.sin(turningincrement)*perpv
        perpv = perpv-gp(perpv,eyes)/gp(eyes,eyes)*eyes
        eyes = eyes/np.sqrt(gp(eyes,eyes))
        perpv = perpv/np.sqrt(gp(perpv,perpv))
    elif key == GLUT_KEY_LEFT:
        eyes = np.cos(turningincrement)*eyes-np.sin(turningincrement)*perpv
        perpv = perpv-gp(perpv,eyes)/gp(eyes,eyes)*eyes
        eyes = eyes/np.sqrt(gp(eyes,eyes))
        perpv = perpv/np.sqrt(gp(perpv,perpv))
def movin(key,x,y):
    global pos
    global eyes
    global perpv
    vec = np.array([0,0]).T
    if key == b'w':
        vec = eyes
        integrator = scipy.integrate.LSODA(fun=model, t0=0, y0=[*pos,*vec],t_bound=movingincrement,max_step=.05,jac=npJac)
        while integrator.status == "running":
            integrator.step()
        pos = np.array([integrator.y[0],integrator.y[1]]).T
        eyes = np.array([integrator.y[2],integrator.y[3]]).T
        perpv = perpv-gp(perpv,eyes)/gp(eyes,eyes)*eyes
    elif key == b's':
        vec = -eyes
        integrator = scipy.integrate.LSODA(fun=model, t0=0, y0=[*pos,*vec],t_bound=movingincrement,max_step=.05,jac=npJac)
        while integrator.status == "running":
            integrator.step()
        pos = np.array([integrator.y[0],integrator.y[1]]).T
        eyes = -np.array([integrator.y[2],integrator.y[3]]).T
        perpv = perpv-gp(perpv,eyes)/gp(eyes,eyes)*eyes
    elif key == b'd':
        vec = perpv
        integrator = scipy.integrate.LSODA(fun=model, t0=0, y0=[*pos,*vec],t_bound=movingincrement,max_step=.05,jac=npJac)
        while integrator.status == "running":
            integrator.step()
        pos = np.array([integrator.y[0],integrator.y[1]]).T
        perpv = np.array([integrator.y[2],integrator.y[3]]).T
        eyes = eyes-gp(perpv,eyes)/gp(perpv,perpv)*perpv
    elif key == b'a':
        vec = -perpv
        integrator = scipy.integrate.LSODA(fun=model, t0=0, y0=[*pos,*vec],t_bound=movingincrement,max_step=.05,jac=npJac)
        while integrator.status == "running":
            integrator.step()
        pos = np.array([integrator.y[0],integrator.y[1]]).T
        perpv = -np.array([integrator.y[2],integrator.y[3]]).T
        eyes = eyes-gp(perpv,eyes)/gp(perpv,perpv)*perpv

    eyes = eyes/np.sqrt(gp(eyes,eyes))
    perpv = perpv/np.sqrt(gp(perpv,perpv))

def render():
    calcs = whatmelook()
    vertices = np.array([])
    for i in range(r):
        if calcs[i][1] == 1:
            j = 2*wallh/fov/calcs[i][0]
            vertices = np.append(vertices,[xbounds[i],j,xbounds[i],-j,xbounds[i+1],j,xbounds[i+1],j,xbounds[i],-j,xbounds[i+1],-j])
    vertices = vertices.astype(np.float32)
    glBufferData(GL_ARRAY_BUFFER,vertices,GL_STREAM_DRAW)
    glClear(GL_COLOR_BUFFER_BIT,GL_DEPTH_BUFFER_BIT)
    
    glBindBuffer(GL_ARRAY_BUFFER,vbo2)
    glUseProgram(shaderProgram2)
    glBegin(GL_TRIANGLE_STRIP)
    glVertex3f(-1.,0.,0.)
    glVertex3f(1.,0.,0.)
    glVertex3f(-1.,-1.,0.)
    glVertex3f(1.,-1.,0.)
    glEnd()
    
    glBindBuffer(GL_ARRAY_BUFFER,vbo)
    glUseProgram(shaderProgram)
    glDrawArrays(GL_TRIANGLES,0,vertices.size)
    glutSwapBuffers()

if __name__=="__main__":
    
    g11str = input("g_11: ")
    g12str = input("g_12: ")
    g21str = input("g_21: ")
    g22str = input("g_22: ")

    x1, x2, a, b = sym.symbols("x1 x2 a b")
    regioncheck = sym.lambdify([x1,x2],sym.sympify(input("Containment condition: ")))
    x = [x1,x2]
    g = sym.Matrix([[g11str, g12str],[g21str,g22str]])
    gup = g.inv()

    G = sym.Array([[[1/2*(gup[k,0]*(sym.diff(g[j,0],x[i])+sym.diff(g[i,0],x[j])-sym.diff(g[i,j],x[0]))+gup[k,1]*(sym.diff(g[j,1],x[i])+sym.diff(g[i,1],x[j])-sym.diff(g[i,j],x[1]))) for k in range(2)] for j in range(2)] for i in range(2)])
    
    def dJacsgen(i,j):
        indies = [x1, x2, a, b]
        return sym.diff(-(G[0][0][i]*a**2+2*G[0][1][i]*a*b+G[1][1][i]*b**2),indies[j])
    Jac = sym.Matrix([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    dJacs = sym.Matrix(2,4,dJacsgen)
    Jac[2:4,:]=dJacs

    def model(t,w):
        return np.array([w[2],w[3],npmod2(*w),npmod3(*w)])

    npg = sym.lambdify([x1,x2],g,"numpy")
    npJach = sym.lambdify([x1,x2,a,b],Jac,"numpy")
    def npJac(t,y):
        return npJach(*y)
    npmod2 = sym.lambdify([x1,x2,a,b],-(G[0][0][0]*a**2+2*G[0][1][0]*a*b+G[1][1][0]*b**2),"numpy")
    npmod3 = sym.lambdify([x1,x2,a,b],-(G[0][0][1]*a**2+2*G[0][1][1]*a*b+G[1][1][1]*b**2),"numpy")
    def gp(v,w):
        return v.T@npg(*pos)@w
    perpv = perpv-gp(perpv,eyes)/gp(eyes,eyes)*eyes
    perpv = perpv/np.sqrt(gp(perpv,perpv))

    thetas = np.array([np.arctan(fov*(k/(r-1)-1/2)) for k in range(r)])
    xbounds = np.array([2*(m-1/2)/(r-1)-1 for m in range(r+1)])

    glutInit()
    glutInitDisplayMode(GLUT_DEPTH|GLUT_DOUBLE|GLUT_RGBA)
    glutInitWindowSize(n,n)
    glutCreateWindow("surfexplV")
    glutDisplayFunc(render)
    glutIdleFunc(render)
    glutSpecialFunc(turning)
    glutKeyboardFunc(movin)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER,vbo)

    vertexSource = R"""
    #version 150
    
    in vec2 position;

    void main()
    {
        gl_Position = vec4(position.x,position.y,0.,1.);
    }"""
    
    fragmentSource = R"""
    #version 150

    out vec4 outColor;

    void main()
    {
        outColor = vec4(0.5,.25,0.,1.);
    }"""

    vertexShader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertexShader,vertexSource)
    glCompileShader(vertexShader)
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragmentShader,fragmentSource)
    glCompileShader(fragmentShader)

    shaderProgram = glCreateProgram()
    glAttachShader(shaderProgram,vertexShader)
    glAttachShader(shaderProgram,fragmentShader)
    glLinkProgram(shaderProgram)
    glUseProgram(shaderProgram)
    
    posAttrib = glGetAttribLocation(shaderProgram,"position")
    glVertexAttribPointer(posAttrib,2,GL_FLOAT,GL_FALSE,0,None)
    glEnableVertexAttribArray(posAttrib)
    
    vbo2 = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER,vbo2)

    fragmentSource2 = R"""
    #version 150

    out vec4 outColor;

    void main()
    {
        outColor = vec4(0.,1.,0.,1.);
    }"""
    fragmentShader2 = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragmentShader2,fragmentSource2)
    glCompileShader(fragmentShader2)

    shaderProgram2 = glCreateProgram()
    glAttachShader(shaderProgram2,fragmentShader2)
    glLinkProgram(shaderProgram2)

    glutMainLoop()
