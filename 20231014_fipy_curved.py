import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fipy import (
    CellVariable,
    ConvectionTerm,
    Gmsh2D,
    FaceVariable,
    TransientTerm,
    DiffusionTerm,
    CentralDifferenceConvectionTerm,
    Viewer
)
import fipy as fp
from tqdm import tqdm

from fipy.viewers.matplotlibViewer.matplotlibStreamViewer import MatplotlibStreamViewer


# Build the mesh:
mesh = Gmsh2D("""
Point(1) = {0.4, 0, 0, 0.025};
Point(2) = {1.6, 0, 0, 0.025};
Point(3) = {2, 0.4, 0, 0.025};
Point(5) = {1.6, 1, 0, 0.025};
Point(6) = {0.4, 1, 0, 0.025};
Point(7) = {0, 0.6, 0, 0.025};
Point(8) = {0, 0.4, 0, 0.025};
Point(9) = {2, 0.6, 0, 0.025};
Line(1) = {9, 3};
Line(2) = {7, 8};
Line(3) = {7, 6};
Line(4) = {5, 9};
Line(5) = {3, 2};
Line(6) = {1, 2};
Line(7) = {1, 8};
Line(8) = {6, 5};
Curve Loop(1) = {8, 4, 1, 5, -6, 7, -2, 3};
Plane Surface(1) = {1};
""") 

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
inlet = (Y < 0.6) & (Y > 0.4) & (X < 0.5)
vx.constrain(1, where=mesh.exteriorFaces & inlet)
phip.constrain(1, where=mesh.exteriorFaces & inlet)

# right

outlet = (Y < 0.6) & (Y > 0.4) & (X > 1.75)
p.constrain(0, where=mesh.exteriorFaces & outlet)



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
        phip.constrain(1, where=mesh.exteriorFaces & inlet)
    if step == 110:
        phip.constrain(0, where=mesh.exteriorFaces & inlet)
    # viewer1.plot()
    # viewer2.plot()
    # pviewer.plot()
    # vxviewer.plot()
    # vyviewer.plot()
    # phiaviewer.plot()
    phipviewer.plot()
    plt.savefig(f"./figures/curved/peclet_{Pe}_step_{step}.png")

input('end')