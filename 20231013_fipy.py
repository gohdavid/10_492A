cellSize = 0.05
radius = 1.
from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, FaceVariable, ConvectionTerm
from fipy.tools import numerix
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
phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = 0.) 
viewer = None
from fipy import input
if __name__ == '__main__':
    viewer = Viewer(vars=phi, datamin=-1, datamax=1.)
    viewer.plotMesh()
D = 1.
eq = TransientTerm() == DiffusionTerm(coeff=D) - ConvectionTerm(coeff=(-1 * FaceVariable(mesh=mesh, value=[1,0]) + 10 * FaceVariable(mesh=mesh, value=[0,1])))
X, Y = mesh.faceCenters
phi.constrain(0, mesh.facesLeft)
phi.constrain(1, mesh.facesRight)
timeStepDuration = 10 * 0.9 * cellSize**2 / (2 * D)
steps = 1000000
from builtins import range
for step in range(steps):
    eq.solve(var=phi,
             dt=timeStepDuration) 
    if viewer is not None:
        viewer.plot()
input("end")