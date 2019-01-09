#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:46:03 2019

@author: matthieu
"""

import numpy


print("start")








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:13:12 2019

@author: matthieu
"""

import numpy as np



#
#grids = []
#for i, pc_file in enumerate(pc_list[:-1]):
#    grid = set()
#    print("pointcloud ", i)
#    # load point cloud
#    pts = np.fromfile(pc_file, np.float32, -1).reshape((-1, 4))
#    pts = pts[:, :3]
#    pts = pts[pts[:, 0] > 0]
#    pts = pts[pts[:, 2] > -0.9]
#    pts = np.round(pts * voxel_grid_scale).astype(int)
#    for p in pts:
#        s = "|".join(p.astype(str))
#        grid.add(s)
#    grids.append(grid)
#     
#    
#occupied = grids[0]
#for i in range(1, len(grids)):
#    diff = grids[i].difference(grids[i-1])
#    if not diff:
#        diff = set()
#    write(diff, "out2/diff_%04d.ply" % (i))
#    
    


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

from vtk.util import numpy_support

grids = []
for i in range(int(vv.app.scene.EndTime)+1):
    vv.app.scene.AnimationTime = i
    grid = np.zeros((x_range, y_range, z_range))

    #pts = np.fromfile(pc_file, np.float32, -1).reshape((-1, 4))
    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    
    pts = pts[:, :3] * scale_factor
    pts += [0, y_half, z_half]
    pts = np.round(pts).astype(int)
    pts = pts[pts[:, 0] >= 0]
    pts = pts[pts[:, 0] < x_full]
    pts = pts[pts[:, 1] >= 0]
    pts = pts[pts[:, 1] < y_range]
    pts = pts[pts[:, 2] >= 0]
    pts = pts[pts[:, 2] < z_range]

    grid[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    grids.append(grid)


#for i in range(1, len(grids)):
#    diff = np.abs(grids[i] - grids[i-1])
#    diff = grids[i]
#    x, y, z = np.where(diff)
#    x -= 20
#    y -= 20
#    z -= 2
#    
#    header = "ply\n" \
#         "format ascii 1.0\n" \
#         "comment VCGLIB generated\n" \
#         "element vertex " + str(len(x)) + "\n" \
#         "property float x\n" \
#         "property float y\n" \
#         "property float z\n" \
#         "element face 0\n" \
#         "property list uchar int vertex_indices\n" \
#         "end_header\n"
#
#    filename = "voxelgrid2/diff_%04d.ply" % (i)
#    with open(filename, "w") as fout:
#        fout.write(header)
#        for xx, yy, zz in zip(x, y, z):
#           fout.write(" ".join(map(str, [xx, yy, zz])) + "\n")

#%%
import copy

dyn_vox = [[]]
to_remove = []
for i in range(1, len(grids)):
    diff = grids[i] - grids[i-1]
    cur_dyn_vox = copy.copy(dyn_vox[i-1])
    for e in to_remove:
        cur_dyn_vox.remove(e)
    to_remove[:] = []
    xx, yy, zz = np.where(diff == 1)
    for x, y, z in zip(xx, yy, zz):
        cur_dyn_vox.append("%d|%d|%d" % (x, y, z))
    xx, yy, zz = np.where(diff == -1)
    for x, y, z in zip(xx, yy, zz):
        c = "%d|%d|%d" % (x, y, z)
        if c in dyn_vox[i-1]:
            to_remove.append(c)
    dyn_vox.append(cur_dyn_vox)
    


for i in range(1, len(dyn_vox)):
    header = "ply\n" \
         "format ascii 1.0\n" \
         "comment VCGLIB generated\n" \
         "element vertex " + str(len(dyn_vox[i])) + "\n" \
         "property float x\n" \
         "property float y\n" \
         "property float z\n" \
         "element face 0\n" \
         "property list uchar int vertex_indices\n" \
         "end_header\n"

    filename = "out2/diff_%04d.ply" % (i)
    with open(filename, "w") as fout:
        fout.write(header)
        for pt in dyn_vox[i]:
           fout.write(pt.replace("|", " ") + "\n")

#%%
from sklearn.cluster import DBSCAN

obj_ids = 0
class obj:
    def __init__(self, frame_id, pos):
        global obj_ids
        self.id = obj_ids
        obj_ids += 1
        self.poses = {frame_id : pos}
        self.color = np.random.randint(0, 255, 3)


    def add_observation(self, frame_id, pos):
        if frame_id in self.poses.keys():
            print("Error position already known at this frame")
        else:
            self.poses[frame_id] = pos

    
obj_list = []        
moving_pts = []
THRESHOLD_NEW_OBJECT = 1.9 * scale_factor

for i in range(1, len(dyn_vox)):
    pts = []
    for pt in dyn_vox[i]:
        pts.append(list(map(float, pt.split("|"))))
    pts = np.vstack(pts)
    moving_pts.append(pts)
    
    # cluster points with DBSCAN
    clustering = DBSCAN(eps=5, min_samples=30).fit_predict(pts)

#    colors = np.random.randint(0, 255, (max(clustering) + 1, 3))
#    colors = np.vstack((colors, [0, 0, 0]))  # black at then end so that -1 correspond to it

    # save grid clustered with colors
#    filename = "out2/diff_%04d.obj" % (i)
#    with open(filename, "w") as fout:
#        for i, p in enumerate(pts):
#            fout.write("v " + " ".join(p.astype(str)) + " " +
#                       " ".join(colors[clustering[i], :].astype(str)) + "\n")

    obj_prev_poses = []
    for o in obj_list:
        if i-1 in o.poses.keys():
            obj_prev_poses.append(o.poses[i-1])
        else:
            obj_prev_poses.append([999999, 999999, 999999])
    if len(obj_prev_poses) > 0:
        obj_prev_poses = np.vstack(obj_prev_poses)
        
    for k in range(max(clustering) + 1):
        centroid = np.mean(pts[clustering == k, :], axis=0)
        if len(obj_prev_poses) == 0:
            new = obj(i, centroid)
            obj_list.append(new)
        else:
            dist = np.sum(np.sqrt((obj_prev_poses - centroid)**2), axis=1)
            closest_i = np.argmin(dist)
            closest_d = np.min(dist)
            if closest_d < THRESHOLD_NEW_OBJECT:
                print("obj associated with dist ", closest_d *10)
                obj_list[closest_i].add_observation(i, centroid)
            else:
                new = obj(i, centroid)
                obj_list.append(new)


obj_to_skip = []
for i, o in enumerate(obj_list):
    frames = sorted(o.poses.keys())
    pa = o.poses[frames[0]]
    pb = o.poses[frames[-1]]
    dist = np.sum(np.sqrt((pa - pb)**2))
    print("dist ", dist)
    if dist < 2 * scale_factor:
        obj_to_skip.append(i)
        continue        
    with open("tracked/" + str(i) + "_obj.obj", "w") as fout:
        for f in sorted(o.poses.keys()):
            p = o.poses[f]
            fout.write("v " + " ".join(p.astype(str)) + " " +
                       " ".join(o.color.astype(str)) + "\n")



total_pts = []
total_colors = []
for i in range(1, len(dyn_vox)):
    pts = []
    colors = []
    for j, o in enumerate(obj_list):
        if j not in obj_to_skip:
            if i in o.poses.keys():
                pts.append(o.poses[i])
                colors.append(o.color)
    total_pts += pts
    total_colors += colors
total_pts = np.vstack(total_pts)
total_colors = np.vstack(total_colors)



    
