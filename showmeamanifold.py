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
tbound = 7

global pos
global eyes
global perpv
pos = [[0,np.array([0,0]).T]]
eyes = [np.array([0,1]).T]
perpv = [np.array([1,0]).T]

class patch:
    def __init__(self,i,g11str,g12str,g22str,regioncheck,compats):
        self.regioncheck = lambda y:(1 if regioncheck(y[0],y[1]) else -1)
        
        x = [x1,x2]
        self.regionchecks = []
        self.cocs = []
        self.cobs = []
        for j in compats:
            if j[0]==False:
                self.regionchecks.append(callFalse)
                self.cocs.append(wampwamp)
                self.cobs.append(wampwamp)
            else:
                cregcheckbool = sym.lambdify([x1,x2],j[0],'numpy')
                def cregcheck(x):
                    return 1 if cregcheckbool(x[0],x[1]) else -1
                ejoeii = (sym.sympify(j[1]),sym.sympify(j[2]))
                ejoeiimap = sym.lambdify([x1,x2],sym.Matrix(ejoeii).T,'numpy')
                def coc(pos):
                    return ejoeiimap(pos[0],pos[1])
                cob = sym.lambdify([x1,x2],sym.Matrix(2,2,lambda k,l: sym.diff(ejoeii[k],x[l])),'numpy')
                def cobmap(pos,vec):
                    return cob(pos[0],pos[1])@vec
                self.regionchecks.append(cregcheck)
                self.cocs.append(coc)
                self.cobs.append(cobmap)
        def cregcheck(x):
            return 1 if regioncheck(x[0],x[1]) else -1
        self.regionchecks[i]=cregcheck
        
        g21str = g12str
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
        def gp(v,w,p):
            return v.T@npg(*p)@w

        self.model = model
        self.npJac = npJac
        self.gp = gp
        

def whatmelook():
    calcs = []
    for theta in thetas:
        psimin = []
        for p in range(len(pos)):
            vec = np.cos(theta)*eyes[p]+np.sin(theta)*perpv[p]
            psimin.append([*pos[p],vec])
        integrator = scipy.integrate.LSODA(fun=patches[psimin[0][0]].model, t0=0, y0=[*psimin[0][1],*psimin[0][2]],t_bound=tbound,max_step=.05,jac=patches[psimin[0][0]].npJac)
        while len(psimin) > 0 and integrator.status == "running":
            integrator.step()
            pin = psimin[0][0]
            if patches[pin].regionchecks[pin]((integrator.y[0],integrator.y[1]))==-1:
                psimin.pop(0)
                if len(psimin)>0:
                    integrator = scipy.integrate.LSODA(fun=patches[psimin[0][0]].model, t0=integrator.t_old, y0=[*psimin[0][1],*psimin[0][2]],t_bound=tbound,max_step=.05,jac=patches[psimin[0][0]].npJac)
                continue
            psimin = [psimin[0]]
            for p in range(patchnum):
                if not p == pin:
                    if patches[pin].regionchecks[p]((integrator.y[0],integrator.y[1])) == 1:
                        psimin.append([p,patches[pin].cocs[p]((integrator.y[0],integrator.y[1]))[0],patches[pin].cobs[p]((integrator.y[0],integrator.y[1]),np.array([integrator.y[2],integrator.y[3]]).T)])
        if not integrator.status == "running":
            calcs.append((0,0))
        else:
            interp = integrator.dense_output()
            def dense(t):
                return patches[pin].regionchecks[pin](interp(t))
            calcs.append((scipy.optimize.bisect(dense,interp.t_min,interp.t_max,xtol=1e-7),1))
    return calcs
def turning(key,x,y):
    global eyes
    global perpv
    if key == GLUT_KEY_RIGHT:
        for i in range(len(pos)):
            cperpv = perpv[i]
            ceyes = eyes[i]
            cgp = patches[pos[i][0]].gp
    
            ceyes = np.cos(turningincrement)*ceyes+np.sin(turningincrement)*cperpv
            cperpv = cperpv-cgp(cperpv,ceyes,pos[i][1])/cgp(ceyes,ceyes,pos[i][1])*ceyes
            ceyes = ceyes/np.sqrt(cgp(ceyes,ceyes,pos[i][1]))
            cperpv = cperpv/np.sqrt(cgp(cperpv,cperpv,pos[i][1]))
         
            perpv[i] = cperpv
            eyes[i] = ceyes

    elif key == GLUT_KEY_LEFT:
        for i in range(len(pos)):
            cperpv = perpv[i]
            ceyes = eyes[i]
            cgp = patches[i].gp
    
            ceyes = np.cos(turningincrement)*ceyes-np.sin(turningincrement)*cperpv
            cperpv = cperpv-cgp(cperpv,ceyes,pos[i][1])/cgp(ceyes,ceyes,pos[i][1])*ceyes
            ceyes = ceyes/np.sqrt(cgp(ceyes,ceyes,pos[i][1]))
            cperpv = cperpv/np.sqrt(cgp(cperpv,cperpv,pos[i][1]))
         
            perpv[i] = cperpv
            eyes[i] = ceyes

def movin(key,x,y):
    global pos
    global eyes
    global perpv
    while True:
        if key == b'w':
            vec = eyes[0]
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=movingincrement,max_step=.05,jac=patches[pos[0][0]].npJac)
            while integrator.status == "running":
                integrator.step()
            if patches[pos[0][0]].regionchecks[pos[0][0]]((integrator.y[0],integrator.y[1])) == -1:
                if len(pos) == 1:
                    break
                pos.pop(0)
                eyes.pop(0)
                perpv.pop(0)
            else:
                pos[0][1] = np.array([integrator.y[0],integrator.y[1]]).T
                eyes[0] = np.array([integrator.y[2],integrator.y[3]]).T
                perpv[0] = perpv[0]-patches[pos[0][0]].gp(perpv[0],eyes[0],pos[0][1])/patches[pos[0][0]].gp(eyes[0],eyes[0],pos[0][1])*eyes[0]
                break
        elif key == b's':
            vec = -eyes[0]
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=movingincrement,max_step=.05,jac=patches[pos[0][0]].npJac)
            while integrator.status == "running":
                integrator.step()
            if patches[pos[0][0]].regionchecks[pos[0][0]]((integrator.y[0],integrator.y[1])) == -1:
                if len(pos) == 1:
                    break
                pos.pop(0)
                eyes.pop(0)
                perpv.pop(0)
            else:
                pos[0][1] = np.array([integrator.y[0],integrator.y[1]]).T
                eyes[0] = -np.array([integrator.y[2],integrator.y[3]]).T
                perpv[0] = perpv[0]-patches[pos[0][0]].gp(perpv[0],eyes[0],pos[0][1])/patches[pos[0][0]].gp(eyes[0],eyes[0],pos[0][1])*eyes[0]
                break
        elif key == b'd':
            vec = perpv[0]
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=movingincrement,max_step=.05,jac=patches[pos[0][0]].npJac)
            while integrator.status == "running":
                integrator.step()
            if patches[pos[0][0]].regionchecks[pos[0][0]]((integrator.y[0],integrator.y[1])) == -1:
                if len(pos) == 1:
                    break
                pos.pop(0)
                eyes.pop(0)
                perpv.pop(0)
            else:
                pos[0][1] = np.array([integrator.y[0],integrator.y[1]]).T
                perpv[0] = np.array([integrator.y[2],integrator.y[3]]).T
                eyes[0] = eyes[0]-patches[pos[0][0]].gp(perpv[0],eyes[0],pos[0][1])/patches[pos[0][0]].gp(perpv[0],perpv[0],pos[0][1])*perpv[0]
                break
        elif key == b'a':
            vec = -perpv[0]
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=movingincrement,max_step=.05,jac=patches[pos[0][0]].npJac)
            while integrator.status == "running":
                integrator.step()
            if patches[pos[0][0]].regionchecks[pos[0][0]]((integrator.y[0],integrator.y[1])) == -1:
                if len(pos) == 1:
                    break
                pos.pop(0)
                eyes.pop(0)
                perpv.pop(0)
            else:
                pos[0][1] = np.array([integrator.y[0],integrator.y[1]]).T
                perpv[0] = -np.array([integrator.y[2],integrator.y[3]]).T
                eyes[0] = eyes[0]-patches[pos[0][0]].gp(perpv[0],eyes[0],pos[0][1])/patches[pos[0][0]].gp(perpv[0],perpv[0],pos[0][1])*perpv[0]
                break
    
    pos = [pos[0]]
    eyes = [eyes[0]]
    perpv = [perpv[0]]
    for p in range(patchnum):
        if not p == pos[0][0]:
            if patches[pos[0][0]].regionchecks[p]((integrator.y[0],integrator.y[1])) == 1:
                pos.append([p,patches[pos[0][0]].cocs[p]((integrator.y[0],integrator.y[1]))[0]])
                eyes.append(patches[pos[0][0]].cobs[p]((integrator.y[0],integrator.y[1]),eyes[0]))
                perpv.append(patches[pos[0][0]].cobs[p]((integrator.y[0],integrator.y[1]),perpv[0]))

    for i in range(len(pos)):
        cperpv = perpv[i]
        ceyes = eyes[i]
        cgp = patches[pos[i][0]].gp

        cperpv = cperpv/np.sqrt(cgp(cperpv,cperpv,pos[i][1]))
        ceyes = ceyes/np.sqrt(cgp(ceyes,ceyes,pos[i][1]))
        
        perpv[i] = cperpv
        eyes[i] = ceyes

def render():
    calcs = whatmelook()
    vertices = np.array([])
    for i in range(r):
        if calcs[i][1] == 1:
            j = 2*wallh/fov/calcs[i][0]/coss[i]
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

def wampwamp(*x):
    print("wampwamp :( this function should never be called")
    raise KeyboardInterrupt

def callFalse(*x):
    return False

if __name__=="__main__":
    
    patchnum = int(input("number of patches: "))
    patches = []
    for i in range(patchnum):
        print("patch "+str(i+1)+":")
        g11str = input("g_11: ")
        g12str = input("g_12: ")
        g22str = input("g_22: ")

        x1, x2, a, b = sym.symbols("x1 x2 a b")
        regioncheck = sym.lambdify([x1,x2],sym.sympify(input("Containment condition: ")))
        compats = []
        for j in range(patchnum):
            if i == j:
                compats.append((False,wampwamp,wampwamp))
            else:
                overlap = input("overlap with patch "+str(j+1)+": ")
                if overlap == False:
                    compats.append((False,wampwamp,wampwamp))
                else:
                    compats.append((overlap,input("y1 on patch "+str(j+1)+": "),input("y2 on patch "+str(j+1)+": ")))

        patches.append(patch(i,g11str,g12str,g22str,regioncheck,compats))
    
    for i in range(len(pos)):
        cperpv = perpv[i]
        ceyes = eyes[i]
        cgp = patches[pos[i][0]].gp

        cperpv = cperpv-cgp(cperpv,ceyes,pos[i][1])/cgp(ceyes,ceyes,pos[i][1])*ceyes
        cperpv = cperpv/np.sqrt(cgp(cperpv,cperpv,pos[i][1]))
        
        perpv[i] = cperpv

    thetas = np.array([np.arctan(fov*(k/(r-1)-1/2)) for k in range(r)])
    coss = np.cos(thetas)
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
