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

cellSize = 0.000000005

# Build the mesh:
mesh = Gmsh2D("curved3.geo") 

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

# Boundary conditions
X, Y = mesh.faceCenters

vx.constrain(0, where=mesh.exteriorFaces)
vy.constrain(0, where=mesh.exteriorFaces)
# p.constrain(0, where=mesh.exteriorFaces)

# left
inlet = (Y < 0.525) & (Y > 0.475) & (X < 0.5)
vx.constrain(1, where=mesh.exteriorFaces & inlet)


# right

outlet = (Y < 0.575) & (Y > 0.475) & (X > 1.75)
p.constrain(0, where=mesh.exteriorFaces & outlet)



#Equations
nu = 100
rho = 0.1

dt = 0.01

# Equation definition
eqvx = (TransientTerm(var=vx) == DiffusionTerm(coeff=nu,var=vx) - ConvectionTerm(coeff=v,var=vx) - (1 / rho) * p.grad[0])
eqvy = (TransientTerm(var=vy) == DiffusionTerm(coeff=nu,var=vy) - ConvectionTerm(coeff=v,var=vy) - (1 / rho) * p.grad[1])
eqp = (DiffusionTerm(coeff=1.) == -1 * rho * (v.divergence**2 - v.divergence / dt))



steps = 1000
sweeps = 2
print('Total time: {} seconds'.format(dt*steps))
total_time = 0.0

viewer1 = MatplotlibStreamViewer(v)
viewer2 = Viewer(v)
# pviewer = Viewer(p)
# vxviewer = Viewer(vx)
# vyviewer = Viewer(vy)

for step in tqdm(range(steps)):
    vx.updateOld()
    vy.updateOld()
    p.updateOld()

    for sweep in range(sweeps):
        res_p = eqp.sweep(var=p, dt=dt)
        res0 = eqvx.sweep(var=vx, dt=dt)
        res1 = eqvy.sweep(var=vy, dt=dt)

        # print(f"step: {step}, sweep: {sweep}, res_p: {res_p}, res0: {res0}, res1: {res1}")

        v[0, :] = vx.faceValue
        v[1, :] = vy.faceValue

    viewer1.plot()
    viewer2.plot()
    # pviewer.plot()
    # vxviewer.plot()
    # vyviewer.plot()

fp.dump.write(v,"velocity_curved.gz")
input('end')