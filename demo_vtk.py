import vtk


def main():
    colors = vtk.vtkNamedColors()

    # Set the background color.
    colors.SetColor("BkgColor", [0.3, 0.2, 0.1, 1.0])

    input1 = vtk.vtkPolyData()
    input2 = vtk.vtkPolyData()

    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(5, 0, 0)
    sphereSource.Update()

    input1.ShallowCopy(sphereSource.GetOutput())

    coneSource = vtk.vtkConeSource()
    coneSource.Update()

    input2.ShallowCopy(coneSource.GetOutput())
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(2)
    multiblock.SetBlock(0, input1)
    multiblock.SetBlock(1, input2)



    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(multiblock.GetBlock(0))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actors to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("BkgColor"))

    # Render and interact
    renderWindowInteractor.Initialize()
    renderWindow.Render()
    renderer.GetActiveCamera().Zoom(0.9)
    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == "__main__":
    main()