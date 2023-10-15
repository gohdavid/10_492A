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
Lx = 880e-6   # meters
Ly = 440e-6    # meters
nx = 50  # nodes
ny = 50

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

phip = CellVariable(name="Phi-protein",
                  mesh=mesh,
                  value=0.,
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
bottom = Ly*0.45
top = Ly*0.55
inlet = (Y < top) & (Y > bottom)
vx.constrain(1e-4, where=mesh.facesLeft & inlet)
phip.constrain(0.4e-9, where=mesh.facesLeft)



# right

outlet = (Y < top) & (Y > bottom)
p.constrain(0, where=mesh.facesRight & outlet)



#Equations
nu = 1e-4
rho = 1e3
dt = 0.01
D = 1e-12

# Equation definition
eqvx = (TransientTerm(var=vx) == DiffusionTerm(coeff=nu,var=vx) - ConvectionTerm(coeff=v,var=vx) - (1 / rho) * p.grad[0])
eqvy = (TransientTerm(var=vy) == DiffusionTerm(coeff=nu,var=vy) - ConvectionTerm(coeff=v,var=vy) - (1 / rho) * p.grad[1])
eqp = (DiffusionTerm(coeff=1.) == -1 * rho * (v.divergence**2 - v.divergence / dt))
eqphip = (TransientTerm(var=phip) == DiffusionTerm(coeff=D,var=phip) - ConvectionTerm(coeff=v,var=phip))


steps = 100
sweeps = 10
print('Total time: {} seconds'.format(dt*steps))
total_time = 0.0



viewer1 = Viewer(v)
viewer2 = Viewer(phip)
# pviewer = Viewer(p)
vxviewer = Viewer(vx)
# vyviewer = Viewer(vy)

for step in tqdm(range(steps)):
    vx.updateOld()
    vy.updateOld()
    p.updateOld()
    phip.updateOld()

    for sweep in range(sweeps):
        res_p = eqp.sweep(var=p, dt=dt)
        res0 = eqvx.sweep(var=vx, dt=dt)
        res1 = eqvy.sweep(var=vy, dt=dt)
        res2 = eqphip.sweep(var=phip, dt=dt)

        # print(f"step: {step}, sweep: {sweep}, res_p: {res_p}, res0: {res0}, res1: {res1}")

        v[0, :] = vx.faceValue
        v[1, :] = vy.faceValue

    viewer1.plot()
    viewer2.plot()
    # pviewer.plot()
    vxviewer.plot()
    # vyviewer.plot()

input('end')