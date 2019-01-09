#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:13:12 2019

@author: matthieu
"""

import numpy as np
from detect_frustrums import get_rotation_around_axis
import glob
import os

pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_26_drive_0017_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_28_drive_0054_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_28_drive_0053_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_29_drive_0026_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_28_drive_0034_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_26_drive_0060_sync/velodyne_points/data/*.bin")

voxel_grid_scale = 5       # 1 means meters, 10 means dm, ...


def write(s, filename):
    # Write pointcloud with a color per class
    header = "ply\n" \
             "format ascii 1.0\n" \
             "comment VCGLIB generated\n" \
             "element vertex " + str(len(s)) + "\n" \
             "property float x\n" \
             "property float y\n" \
             "property float z\n" \
             "element face 0\n" \
             "property list uchar int vertex_indices\n" \
             "end_header\n"
    with open(filename, "w") as fout:
        fout.write(header)
        for p in s:
            pt = list(map(lambda x: float(x)/voxel_grid_scale, p.split("|")))
            fout.write(" ".join(map(str, pt)) + "\n")

    
    
#    with open(filename, "w") as fout:
#        for p in s:
#            pt = list(map(lambda x: float(x)/voxel_grid_scale, p.split("|")))
#            fout.write("v " + " ".join(map(str, pt)) + "\n")


pc_list = sorted(pc_list)
pc_names = []
for f in pc_list:
    pc_names.append(os.path.splitext(os.path.basename(f))[0])

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


grids = []
for i, pc_file in enumerate(pc_list[:-1]):
    grid = np.zeros((x_range, y_range, z_range))

    pts = np.fromfile(pc_file, np.float32, -1).reshape((-1, 4))
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
dyn_vox = [[]]
to_remove = []
for i in range(1, len(grids)):
    diff = grids[i] - grids[i-1]
    cur_dyn_vox = dyn_vox[i-1].copy()
    for e in to_remove:
        cur_dyn_vox.remove(e)
    to_remove.clear()
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
    current = []
    for j, o in enumerate(obj_list):
        if j not in obj_to_skip:
            if i in o.poses.keys():
                pts.append(o.poses[i])
                colors.append(o.color)
                current.append(j)
    print("nb = ", len(current))
    print(current)
    
    filename = "tracked2/tracks_%04d.ply" % (i)
    with open(filename, "w") as fout:
        header = "ply\n" \
         "format ascii 1.0\n" \
         "comment VCGLIB generated\n" \
         "element vertex " + str(len(pts) + len(total_pts)) + "\n" \
         "property float x\n" \
         "property float y\n" \
         "property float z\n" \
         "property uchar red\n" \
         "property uchar green\n" \
         "property uchar blue\n" \
         "element face 0\n" \
         "property list uchar int vertex_indices\n" \
         "end_header\n"
        fout.write(header)
        for p, c in zip(pts, colors):
            p = (p - [0, y_half, z_half]) / scale_factor
            fout.write(" ".join(p.astype(str)) + " " + " ".join(c.astype(str)) + "\n")
        for p, c in zip(total_pts, total_colors):
            p = (p - [0, y_half, z_half]) / scale_factor
            fout.write(" ".join(p.astype(str)) + " " + " ".join(c.astype(str)) + "\n")
    total_pts += pts
    total_colors += colors
    
#%%
### try to save polylines
#for i in range(1, len(dyn_vox)):
#    polylines = []
#    npts = 0
#    for j, o in enumerate(obj_list):
#        if j not in obj_to_skip:
#            frame_ids = sorted(o.poses.keys())
#            if i in frame_ids:
#                line = []
#                for fi in frame_ids:
#                    line.append(o.poses[fi])
#                    npts += 1
#                    if fi == i:
#                        break
#                polylines.append(line)
#    nfaces = sum([len(line) for line in polylines])
#    filename = "tracked2/tracks_%04d.ply" % (i)
#    with open(filename, "w") as fout:
#        header = "ply\n" \
#                 "format ascii 1.0\n" \
#                 "comment VCGLIB generated\n" \
#                 "element vertex " + str(npts * 2) + "\n" \
#                 "property float x\n" \
#                 "property float y\n" \
#                 "property float z\n" \
#                 "property uchar red\n" \
#                 "property uchar green\n" \
#                 "property uchar blue\n" \
#                 "element face " + str(npts-1) + "\n" \
#                 "property list uchar int vertex_indices\n" \
#                 "end_header\n"
#        fout.write(header)
#        faces = []
#        for line in polylines:
#            face = []
#            for p in line:
#                fout.write(" ".join(p.astype(str)) + " 255 255 255 " +  "\n")
#                fout.write(" ".join((p + [0.001, 0.001, 0.001]).astype(str)) + " 255 255 255 " +  "\n")
#                
#        for i in range(1, npts):
#            fout.write("3 " + str(2*(i-1)) + " " + str(2*i) + " " + str(2*i-1) + "\n")
#    
#%%
#for i, o in enumerate(obj_list):
#    if i not in obj_to_skip:
#        print(i, " => ", len(o.poses.keys()))
#        frames = sorted(o.poses.keys())
#        print(frames)
#        print(frames[0])
#        print(frames[-1])
#        print(o.poses[frames[0]])
#        print(o.poses[frames[-1]])
#        print(np.sum(np.sqrt((o.poses[frames[0]] - o.poses[frames[-1]])**2)))
#        if np.sum(np.sqrt((o.poses[frames[0]] - o.poses[frames[-1]])**2)) < 3 * scale_factor:
#            print("INFERIOR !!!!!!!!!!!!!!!!!!!")
            
            
#%%
kk = 0
ts = sorted(obj_list[kk].poses.keys())
pos = []
for t in ts:
    pos.append(obj_list[kk].poses[t])
pos = np.vstack(pos)
pos /= scale_factor
d = [0]
v = [0]
for i in range(1, pos.shape[0]):
    d.append(np.sqrt(np.sum((pos[i, :] - pos[i-1, :]) ** 2)))
    v.append(3600 * (d[-1] / 0.1) / 1000)
    
pos
v
            