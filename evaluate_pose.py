#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:00:07 2019

@author: matthieu
"""

import numpy as np

import matplotlib.pyplot as plt

gt_file = "/media/matthieu/DATA/KITTI/dataset/poses/03.txt"

gt_poses_data = np.loadtxt(gt_file)
gt_poses = [a.reshape((3, 4)) for a in gt_poses_data]
gt_positions = gt_poses_data[:, (3, 7, 11)]


result_file = "/home/matthieu/Dev/VeloView-kwinternal/results_03_bis.csv"


# T 04
T_lidar_to_cam4 = np.array([[-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
                           [-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
                           [9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
                           [0, 0, 0, 1]])

# T03
T_lidar_to_cam = np.array([[2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03],
                           [1.044940741659e-02, 1.056535364138e-02, -9.998895741176e-01, -7.510879138296e-02],
                           [9.999453885620e-01, 1.243653783865e-04, 1.045130299567e-02, -2.721327964059e-01],
                           [0, 0, 0, 1]])

# load transforms (3x4)
with open(result_file, "r") as fin:
    lines = fin.readlines()
res_positions = []
for i in range(1, len(lines)):      # skip the first line
    if len(lines[i]) > 0:
        temp = list(map(float, lines[i].split(",")))
        x = temp[-3]
        y = temp[-2]
        z = temp[-1]
    res_positions.append([x, y, z])
res_positions = np.vstack(res_positions)

res_positions = (T_lidar_to_cam @ np.hstack((res_positions, np.ones((res_positions.shape[0], 1)))).T).T
res_positions = res_positions[:, :3]
res_positions = np.vstack(([0, 0, 0], res_positions))


K = min(gt_positions.shape[0], res_positions.shape[0])


plt.scatter(gt_positions[:K, 0], gt_positions[:K, 2])
plt.scatter(res_positions[:K, 0], res_positions[:K, 2])
plt.ylim(0, 100)
plt.xlim(-100, 100)
plt.show()


plt.figure("test")
plt.plot(gt_positions[:K, 1])
plt.plot(res_positions[:K, 1])

steps = np.linspace(10, K-1, 6).astype(np.int)


# total length
total_dists = [0]
for i in range(1, K):
    total_dists.append(total_dists[-1] + np.sqrt(np.sum((gt_positions[i-1, :] - gt_positions[i, :])**2)))


errors = []
for i in range(K):
    errors.append(np.sqrt(np.sum((gt_positions[i, :] - res_positions[i, :])**2)))

plt.figure("errors")
plt.plot(errors)

for i in steps:
    r = np.sqrt(np.sum((gt_positions[i, :] - res_positions[i, :])**2))
    dist = total_dists[i]
    print("error after %.3fm is %.3fm => %.2f%%" % ( dist, r, 100*r/dist))


with open("gt_trajectory.obj", "w") as fout:
    for p in gt_positions:
        fout.write("v " + " ".join(p.astype(str)) + "\n")
