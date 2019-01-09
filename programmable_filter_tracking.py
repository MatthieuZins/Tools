input = self.GetInput()
p0 = input.GetBlock(0)
p1 = input.GetBlock(1)


import numpy as np
from vtk.util import numpy_support
pts0 = numpy_support.vtk_to_numpy(p0.GetPoints().GetData()).copy()
pts1 = numpy_support.vtk_to_numpy(p1.GetPoints().GetData()).copy()


x_full = 20     # this is in m
y_half = 20     # this is in m
z_half = 1      # this is in m

scale_factor = 10

x_full *= scale_factor
y_half *= scale_factor
z_half *= scale_factor

x_range = x_full
y_range = y_half * 2
z_range = z_half * 2


grid0 = np.zeros((x_range, y_range, z_range))
pts0 = pts0[:, :3] * scale_factor
pts0 += [0, y_half, z_half]
pts0 = np.round(pts0).astype(int)
pts0 = pts0[pts0[:, 0] >= 0]
pts0 = pts0[pts0[:, 0] < x_full]
pts0 = pts0[pts0[:, 1] >= 0]
pts0 = pts0[pts0[:, 1] < y_range]
pts0 = pts0[pts0[:, 2] >= 0]
pts0 = pts0[pts0[:, 2] < z_range]

grid0[pts0[:, 0], pts0[:, 1], pts0[:, 2]] = 1


grid1 = np.zeros((x_range, y_range, z_range))
pts1 = pts1[:, :3] * scale_factor
pts1 += [0, y_half, z_half]
pts1 = np.round(pts1).astype(int)
pts1 = pts1[pts1[:, 0] >= 0]
pts1 = pts1[pts1[:, 0] < x_full]
pts1 = pts1[pts1[:, 1] >= 0]
pts1 = pts1[pts1[:, 1] < y_range]
pts1 = pts1[pts1[:, 2] >= 0]
pts1 = pts1[pts1[:, 2] < z_range]

grid1[pts1[:, 0], pts1[:, 1], pts1[:, 2]] = 1

diff = grid0 - grid1
xx, yy, zz = np.where(diff == 1)
pts = np.vstack((xx, yy, zz)).T
pts = np.ascontiguousarray(pts, dtype=np.float32)

pts -= [0, y_half, z_half]
pts /= scale_factor


from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=5, min_samples=30).fit_predict(pts)


lines = vtk.vtkCellArray()
bbs = []
idx = 0
for k in range(max(clustering) + 1):
    points = pts[clustering == k, :]
    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    bb = np.array([[xmin, ymin, zmin],
                   [xmax, ymin, zmin],
                   [xmax, ymax, zmin],
                   [xmin, ymax, zmin],
                   [xmin, ymin, zmax],
                   [xmax, ymin, zmax],
                   [xmax, ymax, zmax],
                   [xmin, ymax, zmax]])
    bbs.append(bb)
    lines.InsertNextCell(5)
    lines.InsertCellPoint(idx + 0)
    lines.InsertCellPoint(idx + 1)
    lines.InsertCellPoint(idx + 2)
    lines.InsertCellPoint(idx + 3)
    lines.InsertCellPoint(idx + 0)

    lines.InsertNextCell(5)
    lines.InsertCellPoint(idx + 4 + 0)
    lines.InsertCellPoint(idx + 4 + 1)
    lines.InsertCellPoint(idx + 4 + 2)
    lines.InsertCellPoint(idx + 4 + 3)
    lines.InsertCellPoint(idx + 4 + 0)
    idx += 8


bbs = np.vstack(bbs)




poly = vtk.vtkPolyData()
points = vtk.vtkPoints()
points.SetData(numpy_support.numpy_to_vtk(bbs))        
poly.SetPoints(points)
poly.SetLines(lines)


output = self.GetOutput()
output.DeepCopy(poly)