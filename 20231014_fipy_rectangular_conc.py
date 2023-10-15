import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fipy import (
    CellVariable,
    ConvectionTerm,
    Grid2D,
    FaceVariable,
    TransientTerm,
    DiffusionTerm,
    CentralDifferenceConvectionTerm,
    Viewer
)
import fipy as fp

from tqdm import tqdm

from fipy.viewers.matplotlibViewer.matplotlibStreamViewer import MatplotlibStreamViewer

# Geometry
Lx = 2   # meters
Ly = 1    # meters
nx = 51  # nodes
ny = 51

cellSize = 0.05
radius = 1.5

# Build the mesh:
mesh = Grid2D(Lx=Lx, Ly = Ly, nx=nx, ny=ny)

# Main variable and initial conditions
vx = CellVariable(name="x-velocity",
                  mesh=mesh,
                  value=0.,
                  hasOld=True)
vy = CellVariable(name="y-velocity",
                  mesh=mesh,
                  value=-0.,
                  hasOld=True)

v = FaceVariable(name='velocity',
                 mesh=mesh, rank = 1)

p = CellVariable(name = 'pressure',
                 mesh=mesh,
                 value=0.0,
                 hasOld=True)

phip = CellVariable(name="Protein concentration",
                  mesh=mesh,
                  value=0.,
                  hasOld=True)


# Boundary conditions
X, Y = mesh.faceCenters

vx.constrain(0, where=mesh.exteriorFaces)
vy.constrain(0, where=mesh.exteriorFaces)
# p.constrain(0, where=mesh.exteriorFaces)

# left
bottom = Ly*0.40
top = Ly*0.60
inlet = (Y < top) & (Y > bottom)
vx.constrain(1, where=mesh.facesLeft & inlet)
phip.constrain(0, where=mesh.facesLeft & inlet)



# right

outlet = (Y < top) & (Y > bottom)
p.constrain(0, where=mesh.facesRight & outlet)
# vx.constrain(1, where=mesh.facesRight & outlet)


#Equations
Re = 1e-2
Pe = 1e4
kf = 5.96e4/400e-9
kr = 2.48e-3

dt = 0.1

# Equation definition
eqvx = (TransientTerm(var=vx) == DiffusionTerm(coeff=1/Re,var=vx) - ConvectionTerm(coeff=v,var=vx) - (1 / Re) * p.grad[0])
eqvy = (TransientTerm(var=vy) == DiffusionTerm(coeff=1/Re,var=vy) - ConvectionTerm(coeff=v,var=vy) - (1 / Re) * p.grad[1])
eqp = (DiffusionTerm(coeff=1.) == -1 * Re * (v.divergence**2 - v.divergence / dt))
eqphip = (TransientTerm(var=phip) == DiffusionTerm(coeff=1/Pe,var=phip) - ConvectionTerm(coeff=v,var=phip))


steps = 1000
sweeps = 3
print('Total time: {} seconds'.format(dt*steps))
total_time = 0.0


# viewer1 = MatplotlibStreamViewer(v)
# viewer2 = Viewer(v)
# pviewer = Viewer(p)
# vxviewer = Viewer(vx)
# vyviewer = Viewer(vy)
phipviewer = Viewer(phip,datamin=0,datamax=4)
# phiaviewer = Viewer(phia)

for step in tqdm(range(steps)):
    vx.updateOld()
    vy.updateOld()
    p.updateOld()
    phip.updateOld()

    for sweep in range(sweeps):
        res_p = eqp.sweep(var=p, dt=dt)
        res0 = eqvx.sweep(var=vx, dt=dt)
        res1 = eqvy.sweep(var=vy, dt=dt)
        resphip = eqphip.sweep(var=phip, dt=dt)

        # print(f"step: {step}, sweep: {sweep}, res_p: {res_p}, res0: {res0}, res1: {res1}")

        v[0, :] = vx.faceValue
        v[1, :] = vy.faceValue

    if step == 10:
        phip.constrain(1, where=mesh.facesLeft & inlet)
    if step == 110:
        phip.constrain(0, where=mesh.exteriorFaces)
    # viewer1.plot()
    # viewer2.plot()
    # pviewer.plot()
    # vxviewer.plot()
    # vyviewer.plot()
    # phiaviewer.plot()
    if step%20 == 0:
        phipviewer.plot()
        plt.savefig(f"./figures/rectangular/peclet_{Pe}_step_{step}.png")


fp.dump.write(v,"velocity_rectangular.gz")
input('end')