from OpenGL.GLUT import *
from OpenGL.GL import *
import numpy as np
import sympy as sym
import scipy.integrate
from numpy import sqrt
import sys
import os
import shelve
from functools import partial

settings = {"width": 1067, "height": 800, "r": 40, "fov": 1, "wallh": .4, "turningincrement": np.pi/16, "movingincrement": .05, "tbound": 7}
pos = [[0,np.array([0.,0.]).T]]
eyes = [np.array([0,1]).T]
perpv = [np.array([1,0]).T]

x1, x2, a, b = sym.symbols("x1 x2 a b")

class patch:
    def __init__(self,i,g11str,g12str,g22str,regioncheckstr,compats):
        def regcheckwrapper(cregcheckbool):
            def cregcheck(x):
                return 1 if cregcheckbool(x[0],x[1]) else -1
            return cregcheck
        def condom(ejoeiimap):
            def coc(pos):
                return ejoeiimap(pos[0],pos[1])
            return coc
        def cobwrapper(cob):
            def cobmap(pos,vec):
                return cob(pos[0],pos[1])@vec
            return cobmap

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
                ejoeii = (sym.sympify(j[1]),sym.sympify(j[2]))
                ejoeiimap = sym.lambdify([x1,x2],sym.Matrix(ejoeii).T,'numpy')
                cob = sym.lambdify([x1,x2],sym.Matrix(2,2,lambda k,l: sym.diff(ejoeii[k],x[l])),'numpy')
                self.regionchecks.append(regcheckwrapper(cregcheckbool))
                self.cocs.append(condom(ejoeiimap))
                self.cobs.append(cobwrapper(cob))
        regioncheck = sym.lambdify([x1,x2],sym.sympify(regioncheckstr))
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
        
class patchdata:
    def __init__(self,i,g11str,g12str,g22str,regioncheckstr,compats):
        self.i = i
        self.g11str = g11str
        self.g12str = g12str
        self.g22str = g22str
        self.regioncheckstr = regioncheckstr
        self.compats = compats

def whatmelook():
    calcs = []
    for theta in thetas:
        psimin = []
        for p in range(len(pos)):
            vec = np.cos(theta)*eyes[p]+np.sin(theta)*perpv[p]
            psimin.append([*pos[p],vec])
        integrator = scipy.integrate.LSODA(fun=patches[psimin[0][0]].model, t0=0, y0=[*psimin[0][1],*psimin[0][2]],t_bound=settings["tbound"],max_step=.05,jac=patches[psimin[0][0]].npJac)
        while len(psimin) > 0 and integrator.status == "running":
            integrator.step()
            pin = psimin[0][0]
            if patches[pin].regionchecks[pin]((integrator.y[0],integrator.y[1]))==-1:
                psimin.pop(0)
                if len(psimin)>0:
                    integrator = scipy.integrate.LSODA(fun=patches[psimin[0][0]].model, t0=integrator.t_old, y0=[*psimin[0][1],*psimin[0][2]],t_bound=settings["tbound"],max_step=.05,jac=patches[psimin[0][0]].npJac)
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
    if key == GLUT_KEY_RIGHT:
        for i in range(len(pos)):
            cperpv = perpv[i]
            ceyes = eyes[i]
            cgp = patches[pos[i][0]].gp
    
            ceyes = np.cos(settings["turningincrement"])*ceyes+np.sin(settings["turningincrement"])*cperpv
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
    
            ceyes = np.cos(settings["turningincrement"])*ceyes-np.sin(settings["turningincrement"])*cperpv
            cperpv = cperpv-cgp(cperpv,ceyes,pos[i][1])/cgp(ceyes,ceyes,pos[i][1])*ceyes
            ceyes = ceyes/np.sqrt(cgp(ceyes,ceyes,pos[i][1]))
            cperpv = cperpv/np.sqrt(cgp(cperpv,cperpv,pos[i][1]))
         
            perpv[i] = cperpv
            eyes[i] = ceyes

def movin(key,x,y):
    while True:
        if key == b'w':
            vec = eyes[0]
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=settings["movingincrement"],max_step=.05,jac=patches[pos[0][0]].npJac)
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
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=settings["movingincrement"],max_step=.05,jac=patches[pos[0][0]].npJac)
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
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=settings["movingincrement"],max_step=.05,jac=patches[pos[0][0]].npJac)
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
            integrator = scipy.integrate.LSODA(fun=patches[pos[0][0]].model, t0=0, y0=[*pos[0][1],*vec],t_bound=settings["movingincrement"],max_step=.05,jac=patches[pos[0][0]].npJac)
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
        else:
            break
    temppos = pos[0]
    tempeyes = eyes[0]
    tempperpv = perpv[0]
    pos.clear()
    pos.append(temppos)
    eyes.clear()
    eyes.append(tempeyes)
    perpv.clear()
    perpv.append(tempperpv)
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
    for i in range(settings["r"]):
        if calcs[i][1] == 1:
            j = settings["wallh"]/settings["fov"]/calcs[i][0]*settings["width"]/settings["height"]/coss[i]
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

def textpatches():
    patchnum = int(input("number of patches: "))
    patches = []
    patchdatas = []
    for i in range(patchnum):
        print("patch "+str(i+1)+":")
        g11str = input("g_11: ")
        g12str = input("g_12: ")
        g22str = input("g_22: ")

        regioncheckstr = input("Containment condition: ")
        compats = []
        for j in range(patchnum):
            if i == j:
                compats.append([False,wampwamp,wampwamp])
            else:
                overlap = input("overlap with patch "+str(j+1)+": ")
                if overlap == "False":
                    compats.append([False,wampwamp,wampwamp])
                else:
                    compats.append([overlap,input("y1 on patch "+str(j+1)+": "),input("y2 on patch "+str(j+1)+": ")])

        patches.append(patch(i,g11str,g12str,g22str,regioncheckstr,compats))
        patchdatas.append(patchdata(i,g11str,g12str,g22str,regioncheckstr,compats))
    return patches, patchdatas

def writepatches(path,patchdatas):
    shelf = shelve.open(path)
    shelf["patchdatas"] = patchdatas
    shelf.close()

def loadpatches(path):
    shelf = shelve.open(path)
    patchdatas = shelf["patchdatas"]
    shelf.close()
    patches = []
    for p in patchdatas:
        patches.append(patch(p.i,p.g11str,p.g12str,p.g22str,p.regioncheckstr,p.compats))
    return patches, patchdatas

def guipatches():
    import PyQt5
    from PyQt5 import QtWidgets as qt
    from PyQt5.QtCore import Qt as Qt
    
    def updatepatchwids(mf):
        if mf in loadedpatches.keys():
            patchdatas = loadedpatches[mf]
        else:
            patches, patchdatas = loadpatches(mf)
            loadedpatches[mf]=patchdatas
        for i in range(len(patchwidslist)):
            patchwidslist[i].deleteLater()
        patchwidslist.clear()
        nonlocal patchplus
        patchplus.deleteLater()
        for patchdata in patchdatas:
            patchwid = patchWidget(patchdata)
            patchwids.addWidget(patchwid)
            patchwidslist.append(patchwid)

        patchplus = qt.QPushButton("+")
        patchplus.clicked.connect(addpatch)
        patchwids.addWidget(patchplus)
        nonlocal cmf
        cmf = mf

    def addpatch():
        loadedpatches[cmf].append(patchdata(0,'','','','',[]))
        for pat in loadedpatches[cmf]:
            pat.compats.append([False,'',''])
            loadedpatches[cmf][-1].compats.append([False,'',''])
        loadedpatches[cmf][-1].compats.pop()
        loadedpatches[cmf][-1].i = len(loadedpatches[cmf][-1].compats)-1
        updatepatchwids(cmf)
    
    def settingfromform(key,element,typ=float):
        settings[key] = typ(element.text())

    def startpatchfromform(element):
        pos[0][0] = int(element.text())

    def startposfromform(i,element):
        pos[0][1][i] = float(element.text())

    def mfreset():
        loadedpatches.pop(cmf)
        updatepatchwids(cmf)

    def mfsave():
        writepatches(cmf,loadedpatches[cmf])

    def filesupdate():
        for file in fileslist:
            file.deleteLater()
        nonlocal pb
        pb.deleteLater()
        fileslist.clear()
        for mf in mfs:
            file = qt.QPushButton(mf)
            file.clicked.connect(partial(updatepatchwids,mf))
            fileslist.append(file)
            files.addWidget(file)
        pb = fileplusbuttonWidget()
        files.addWidget(pb)

    def newfile():
        nonlocal newmfnum
        newmfnum += 1
        newname, ok = qt.QInputDialog.getText(window,"Name new manifold","Manifold name",text="New manifold "+str(newmfnum))
        if not ok:
            newmfnum -= 1
            return
        if not newname[-3:]==".mf":
            newname = newname + ".mf"
        mfs.append(newname)
        patchedatas = [patchdata(0,'','','','',[[False,'','']])]
        loadedpatches[newname] = patchedatas
        filesupdate()
    class mapWidget(qt.QFormLayout):
        def __init__(self,patchdata,pnum):
            super().__init__()

            def compatfromentry(patchdata,i,entry):
                patchdata.compats[pnum][i] = entry.text()

            entry = qt.QLineEdit(str(patchdata.compats[pnum][0]))
            entry.editingFinished.connect(partial(compatfromentry,patchdata,0,entry))
            self.addRow("Overlap condition:",entry)
            entry = qt.QLineEdit(str(patchdata.compats[pnum][1]))
            entry.editingFinished.connect(partial(compatfromentry,patchdata,1,entry))
            self.addRow("y1:",entry)
            entry = qt.QLineEdit(str(patchdata.compats[pnum][2]))
            entry.editingFinished.connect(partial(compatfromentry,patchdata,2,entry))
            self.addRow("y2:",entry)
            self.setFormAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            
    class fileplusbuttonWidget(qt.QPushButton):
        def __init__(self):
            super().__init__("+")
            self.clicked.connect(newfile)

    class patchWidget(qt.QWidget):
        def __init__(self,patchdata):
            super().__init__()

            def updatemap():
                nonlocal mapwid
                for i in range(2,-1,-1):
                    mapwid.removeRow(i)
                mapwid.deleteLater()
                mapwid = mapWidget(patchdata,int(pselector.currentText())-1)
                layout.addLayout(mapwid)

            def setfromentry(patchdata,attr,entry):
                setattr(patchdata,attr,entry.text())

            layout = qt.QHBoxLayout()
            metric = qt.QFormLayout()
            entry = qt.QLineEdit(patchdata.g11str)
            entry.editingFinished.connect(partial(setfromentry,patchdata,"g11str",entry))
            metric.addRow("g11",entry)
            entry = qt.QLineEdit(patchdata.g12str)
            entry.editingFinished.connect(partial(setfromentry,patchdata,"g12str",entry))
            metric.addRow("g12",entry)
            entry = qt.QLineEdit(patchdata.g22str)
            entry.editingFinished.connect(partial(setfromentry,patchdata,"g22str",entry))
            metric.addRow("g22",entry)
            entry = qt.QLineEdit(patchdata.regioncheckstr)
            entry.editingFinished.connect(partial(setfromentry,patchdata,"regioncheckstr",entry))
            metric.addRow("Containment condition:",entry)
            metric.setFormAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            layout.addLayout(metric)
            if len(patchdata.compats) > 1:
                layout.addWidget(qt.QLabel("Map to patch"))
                pselector = qt.QComboBox()
                items = [*range(len(patchdata.compats))]
                items.remove(patchdata.i)
                items = [str(item+1) for item in items]
                pselector.addItems(items)
                pselector.currentTextChanged.connect(updatemap)
                layout.addWidget(pselector)
                layout.addWidget(qt.QLabel(":"))
                mapwid = mapWidget(patchdata,int(pselector.currentText())-1)
                layout.addLayout(mapwid)

            self.setLayout(layout)

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "."

    files = os.listdir(path)
    mfs = []
    cmf = ":D"
    newmfnum = 0
    loadedpatches={}
    fileslist = []
    patchwidslist = []
    for file in files:
        if file[-3:]==".mf":
            mfs.append(file)

    app = qt.QApplication([sys.argv])
    window = qt.QWidget()
    layout = qt.QHBoxLayout()
    files = qt.QVBoxLayout()
    patchwids = qt.QVBoxLayout()
    pb = fileplusbuttonWidget()
    patchplus = qt.QPushButton("+")

    if len(mfs)==0:
        newfile()
    filesupdate()
    updatepatchwids(mfs[0])

    layout.addLayout(files)
    layout.addLayout(patchwids)
    layoutwoptionsnbuttons = qt.QVBoxLayout()
    layoutwoptionsnbuttons.addLayout(layout)
    
    optionsnbuttons = qt.QWidget()
    optionswindow = qt.QWidget()
    opLayout = qt.QFormLayout()
    cop = qt.QHBoxLayout()
    width = qt.QLineEdit(str(settings["width"]))
    width.editingFinished.connect(partial(settingfromform,"width",width,int))
    cop.addWidget(width)
    height = qt.QLineEdit(str(settings["height"]))
    height.editingFinished.connect(partial(settingfromform,"height",height,int))
    cop.addWidget(height)
    opLayout.addRow("Window width and height (px):",cop)
    r = qt.QLineEdit(str(settings["r"]))
    r.editingFinished.connect(partial(settingfromform,"r",r,int))
    opLayout.addRow("Rays to shoot out per frame",r)
    fov = qt.QLineEdit(str(settings["fov"]))
    fov.editingFinished.connect(partial(settingfromform,"fov",fov))
    opLayout.addRow("Fov",fov)
    wallh = qt.QLineEdit(str(settings["wallh"]))
    wallh.editingFinished.connect(partial(settingfromform,"wallh",wallh))
    opLayout.addRow("Wall height",wallh)
    turningincrement = qt.QLineEdit(str(settings["turningincrement"]))
    turningincrement.editingFinished.connect(partial(settingfromform,"turningincrement",turningincrement))
    opLayout.addRow("Turning increment (rad)",turningincrement)
    movingincrement = qt.QLineEdit(str(settings["movingincrement"]))
    movingincrement.editingFinished.connect(partial(settingfromform,"movingincrement",movingincrement))
    opLayout.addRow("Moving increment",movingincrement)
    tbound = qt.QLineEdit(str(settings["tbound"]))
    tbound.editingFinished.connect(partial(settingfromform,"tbound",tbound))
    opLayout.addRow("Path length to trace for light",tbound)
    cop = qt.QHBoxLayout()
    sp = qt.QLineEdit(str(pos[0][0]))
    sp.editingFinished.connect(partial(startpatchfromform,sp))
    cop.addWidget(sp)
    positionwid = qt.QLineEdit(str(pos[0][1][0]))
    positionwid.editingFinished.connect(partial(startposfromform,0,positionwid))
    cop.addWidget(positionwid)
    positionwid = qt.QLineEdit(str(pos[0][1][1]))
    positionwid.editingFinished.connect(partial(startposfromform,1,positionwid))
    cop.addWidget(positionwid)

    opLayout.addRow("Starting position (patch number, x1, x2)",cop)
    opLayoutwok = qt.QVBoxLayout()
    opLayoutwok.addLayout(opLayout)
    okbutt = qt.QPushButton("Ok")
    okbutt.clicked.connect(optionswindow.hide)
    opLayoutwok.addWidget(okbutt)
    optionswindow.setLayout(opLayoutwok)
    optionswindow.setWindowTitle("Options")

    optionsnbuttons = qt.QHBoxLayout()
    option = qt.QPushButton("Options")
    option.clicked.connect(optionswindow.show)
    optionsnbuttons.addWidget(option)
    butt = qt.QPushButton("Reset")
    butt.clicked.connect(mfreset)
    optionsnbuttons.addWidget(butt)
    butt = qt.QPushButton("Save")
    butt.clicked.connect(mfsave)
    optionsnbuttons.addWidget(butt)
    butt = qt.QPushButton("Run")
    butt.clicked.connect(partial(app.exit,1))
    optionsnbuttons.addWidget(butt)
    layoutwoptionsnbuttons.addLayout(optionsnbuttons)
    window.setLayout(layoutwoptionsnbuttons)
    window.setWindowTitle("surfexplV")
    window.show()
    code = app.exec()
    if code == 1:
        patches = []
        for p in loadedpatches[cmf]:
            patches.append(patch(p.i,p.g11str,p.g12str,p.g22str,p.regioncheckstr,p.compats))
        return patches
    else:
        sys.exit()
if __name__=="__main__":
    
    if "-i" in sys.argv:
        patches, patchdatas = textpatches()
    elif len(sys.argv) > 1:
        if not os.path.isdir(sys.argv[1]):
            if os.path.exists(sys.argv[1]):
                patches, patchdatas = loadpatches(sys.argv[1])
            elif os.path.exists(sys.argv[1]+".mf"):
                patches, patchdatas = loadpatches(sys.argv[1]+".mf")
            else:
                patches,patchdatas = textpatches()
                if sys.argv[1][-3:]==".mf":
                    writepatches(sys.argv[1],patchdatas)
                    print("written to "+sys.argv[1])
                else:
                    writepatches(sys.argv[1]+".mf",patchdatas)
                    print("written to "+sys.argv[1]+".mf")
    else:
        patches = guipatches()
    patchnum = len(patches)

    
    for i in range(len(pos)):
        cperpv = perpv[i]
        ceyes = eyes[i]
        cgp = patches[pos[i][0]].gp

        cperpv = cperpv-cgp(cperpv,ceyes,pos[i][1])/cgp(ceyes,ceyes,pos[i][1])*ceyes
        cperpv = cperpv/np.sqrt(cgp(cperpv,cperpv,pos[i][1]))
        
        perpv[i] = cperpv

    thetas = np.array([np.arctan(settings["fov"]*(k/(settings["r"]-1)-1/2)) for k in range(settings["r"])])
    coss = np.cos(thetas)
    xbounds = np.array([2*(m-1/2)/(settings["r"]-1)-1 for m in range(settings["r"]+1)])

    glutInit()
    glutInitDisplayMode(GLUT_DEPTH|GLUT_DOUBLE|GLUT_RGBA)
    glutInitWindowSize(settings["width"],settings["height"])
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
