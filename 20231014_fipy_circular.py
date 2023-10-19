import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

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

from fipy.viewers.matplotlibViewer.matplotlibStreamViewer import MatplotlibStreamViewer
from fipy.viewers.matplotlibViewer.matplotlibVectorViewer import MatplotlibVectorViewer
# Geometry

cellSize = 0.02
radius = 0.5

# Build the mesh:
mesh = Gmsh2D('''
              cellSize = %(cellSize)g;
              radius = %(radius)g;
              Point(1) = {0, 0, 0, cellSize};
              Point(2) = {-radius, 0, 0, cellSize};
              Point(3) = {0, radius, 0, cellSize};
              Point(4) = {radius, 0, 0, cellSize};
              Point(5) = {0, -radius, 0, cellSize};
              Circle(6) = {2, 1, 3};
              Circle(7) = {3, 1, 4};
              Circle(8) = {4, 1, 5};
              Circle(9) = {5, 1, 2};
              Line Loop(10) = {6, 7, 8, 9};
              Plane Surface(11) = {10};
              ''' % locals()) 

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
# p.constrain(1, where=mesh.exteriorFaces)

# left
inlet = (Y < 0.1) & (Y > -0.1) & (X < -0.25)
vx.constrain(1, where=mesh.exteriorFaces & inlet)
phip.constrain(0, where=mesh.exteriorFaces & inlet)


# right

outlet = (Y < 0.1) & (Y > -0.1) & (X > 0.25)
p.constrain(0, where=mesh.exteriorFaces & outlet)



#Equation#Equations
Re = 1e-2
Pe = 1e2
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
# viewer2 = MatplotlibVectorViewer(v)
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
        phip.constrain(0,  where=mesh.exteriorFaces & inlet)
    # viewer2.plot()
    # if step%20 == 0:
    #     plt.savefig(f"./figures/circular/velocityfield_step_{step}.png")
    # vxviewer.plot()
    # if step%20 == 0:
        # plt.savefig(f"./figures/circular/vx_step_{step}.png")
    # vyviewer.plot()
    # if step%20 == 0:
        # plt.savefig(f"./figures/circular/vy_step_{step}.png")
    # pviewer.plot()
    phipviewer.plot()

    plt.savefig(f"./figures/circular/peclet_{Pe}_reynold_{Re}_step_{step}.png")
    # phiaviewer.plot()


fp.dump.write(v,"velocity_circular.gz")
input('end')