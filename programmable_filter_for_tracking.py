from vtk.util import numpy_support
import os
import numpy as np
import programmable_filter_tracking
reload(programmable_filter_tracking)



inp = self.GetInput()
p0 = inp.GetBlock(0)
p1 = inp.GetBlock(1)

t = int(inp.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()))


pts, lines, colors = programmable_filter_tracking.run_algo(p0, p1, t)


if pts is not None and lines is not None and colors is not None:
    # numpy to polydata
    poly = vtk.vtkPolyData()
    # points
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(pts, deep=True, array_type=vtk.VTK_FLOAT))       
    poly.SetPoints(points)
    
    # lines
    vtklines = vtk.vtkCellArray()
    for l in lines:
        vtklines.InsertNextCell(len(l))
        for p in l:
            vtklines.InsertCellPoint(p)
    poly.SetLines(vtklines)

    # colors
    color_array = numpy_support.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    color_array.SetName("Colors")
    poly.GetCellData().SetScalars(color_array)

    
    # copy to output
    output = self.GetOutput()
    output.DeepCopy(poly)