#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:03:47 2019

@author: matthieu
"""

import vtk

filename = "/home/matthieu/Dev/Tools/tracking_results/17/ply/pc_0000000000.ply"



#Read and display for verication
reader = vtk.vtkPLYReader()
reader.SetFileName(filename)
reader.Update()

glyph = vtk.vtkVertexGlyphFilter()
glyph.SetInputConnection(reader.GetOutputPort())

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyph.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

renderer.AddActor(actor)
renderer.SetBackground(.3, .6, .3)   #Background color green

renderWindow.Render()
renderWindowInteractor.Start()