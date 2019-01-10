input = self.GetInput()
p0 = input.GetBlock(0)
p1 = input.GetBlock(1)

# transform polydata to numpy points
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
clustering = DBSCAN(eps=0.9, min_samples=30).fit_predict(pts)



import numpy as np
import os

def distance(pos1, pos2):
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))


class track:
    def __init__(self, color=None):
        print("Create new track with color : ", color)
        self.poses = {}     # contains frame_id : position
        self.dims = {}      # contains frame_id : dimension
        if color is not None:
            self.color = color
        else:
            self.color = np.random.randint(0, 255, 3)
        print(self.color)


    def add_observation(self, frame_id, position, dim):
        if frame_id in self.poses.keys():
            print("Warning: a position is already known for this object at this frame.")
        self.poses[frame_id] = position
        self.dims[frame_id] = dim

    def last_observation(self):
        ids = sorted(self.poses.keys())
        max_idx = max(ids)
        return max_idx, self.poses[max_idx], self.dims[max_idx]

    def __str__(self):
        ids = sorted(self.poses.keys())
        s = ""
        for i in ids:
            s += str(i) + " " + str(self.poses[i]) + " " + str(self.dims[i]) + "   "
        return s

    def total_movement(self):
        ids = sorted(self.poses.keys())
        total = 0
        for i in range(1, len(ids)):
            p0 = self.poses[ids[i-1]]
            p1 = self.poses[ids[i]]
            total += distance(p0, p1)
        return total

class tracks_manager:
    def __init__(self, treshold_tracks_association=2):
        self.tracks = []
        self.THRESHOLD_TRACKS_ASSOCIATION = treshold_tracks_association
        pass

    def add_observation(self, frame_id, position, dim):
        BIG_NUMBER = 999999999
        dist_to_prev = [BIG_NUMBER] * len(self.tracks)
        for i, tr in enumerate(self.tracks):
            last_frame, last_pos, last_dim = tr.last_observation()
            if last_frame == frame_id - 1 or last_frame == frame_id:
                dist_to_prev[i] = distance(position, last_pos)
        dist_min = BIG_NUMBER
        dist_min_i = 0
        if len(dist_to_prev) > 0:
            dist_min_i = np.argmin(dist_to_prev)
            dist_min = dist_to_prev[dist_min_i]
        if dist_min > self.THRESHOLD_TRACKS_ASSOCIATION:
            self.tracks.append(track())  # add a new track
            dist_min_i = len(self.tracks) - 1
        self.tracks[dist_min_i].add_observation(frame_id, position, dim)

    def get_moving_tracks_idx(self):
        list_of_moving_tracks = []
        for i, tr in enumerate(self.tracks):
            if len(tr.poses.keys()) > 2 and tr.total_movement() < 2:
                continue
            list_of_moving_tracks.append(i)
        return list_of_moving_tracks


    def __str__(self):
        s = ""
        for i, track in enumerate(self.tracks):
            s += "track " + str(i) + " " + str(track) + "\n"
        return s


def save(tm, filename):
    max_id = max([max(tr.poses.keys()) for tr in tm.tracks]) + 2
    # data layout:
    #   - each track is on two lines
    #   - first line for positions
    #   - second line for bb dims
    #   - the last col contains the colors
    data = np.zeros((len(tm.tracks) * 2, max_id + 1, 4))
    for i, tr in enumerate(tm.tracks):
        for k in sorted(tr.poses.keys()):
            data[2 * i, k, :3] = tr.poses[k]
            data[2 * i, k, 3] = 1
            data[2 * i + 1, k, :3] = tr.dims[k]
            data[2 * i + 1, k, 3] = 1
        data[2 * i, -1, :3] = tr.color
        data[2 * i + 1, -1, :3] = tr.color
    np.save(filename, data)

def load(filename):
    filename += ".npy"
    if os.path.exists(filename):
        data = np.load(filename)
        colors = data[::2, -1, :]
        positions = data[::2, :-1, :]
        dimensions = data[1::2, :-1, :]
        tm = tracks_manager()
        for i in range(positions.shape[0]):
            tr = track(colors[i, :3])
            for k in range(positions.shape[1]):
                if positions[i, k, 3] == 1:
                    tr.add_observation(k, positions[i, k, :3], dimensions[i, k, :3])
            tm.tracks.append(tr)
        return tm
    else:
        return tracks_manager()


t = int(input.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()))
if t > 0:
    tm = load("/home/matthieu/Dev/Tools/filesave.bin")
else:
    tm = tracks_manager()


print "time ", t
for k in range(max(clustering) + 1):
    center = np.mean(pts[clustering == k, :], axis=0)
    xmin, ymin, zmin = np.min(pts[clustering == k, :], axis=0)
    xmax, ymax, zmax = np.max(pts[clustering == k, :], axis=0)
    xsize = max(xmax - center[0], center[0] - xmin)
    ysize = max(ymax - center[1], center[1] - ymin)
    zsize = max(zmax - center[2], center[2] - zmin)
    print "new observation ", t
    tm.add_observation(t, center, [xsize, ysize, zsize]) ## frame id 


save(tm, "/home/matthieu/Dev/Tools/filesave.bin")

lines = vtk.vtkCellArray()
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")
bbs = []
idx = 0

moving_objects = tm.get_moving_tracks_idx()     # list of tracks considered moving
for i in moving_objects:
    track = tm.tracks[i]
    idf, pos, dim = track.last_observation()
    if idf != t:
        continue
    xmin, ymin, zmin = pos - dim
    xmax, ymax, zmax = pos + dim
    bb = np.array([[xmin, ymin, zmin],
                   [xmax, ymin, zmin],
                   [xmax, ymax, zmin],
                   [xmin, ymax, zmin],
                   [xmin, ymin, zmax],
                   [xmax, ymin, zmax],
                   [xmax, ymax, zmax],
                   [xmin, ymax, zmax]])
    bbs.append(bb)
    lines.InsertNextCell(16)
    lines.InsertCellPoint(idx + 0)
    lines.InsertCellPoint(idx + 1)
    lines.InsertCellPoint(idx + 2)
    lines.InsertCellPoint(idx + 3)
    lines.InsertCellPoint(idx + 0)
    lines.InsertCellPoint(idx + 4)
    lines.InsertCellPoint(idx + 5)
    lines.InsertCellPoint(idx + 1)
    lines.InsertCellPoint(idx + 5)
    lines.InsertCellPoint(idx + 6)
    lines.InsertCellPoint(idx + 2)
    lines.InsertCellPoint(idx + 6)
    lines.InsertCellPoint(idx + 7)
    lines.InsertCellPoint(idx + 3)
    lines.InsertCellPoint(idx + 7)
    lines.InsertCellPoint(idx + 4)

    idx += 8

    #colors.InsertNextTuple3(*track.color)
    colors.InsertNextTuple3(*track.color)

    # line
    positions = [track.poses[fid] for fid in sorted(track.poses.keys())]
    bbs.append(positions)

    lines.InsertNextCell(len(positions))
    for j in range(len(positions)):
        lines.InsertCellPoint(idx + j)
    idx += len(positions)
    colors.InsertNextTuple3(track.color[0], track.color[1], track.color[2])


if len(bbs) > 0:
    bbs = np.vstack(bbs)

    # transform to output poly
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(bbs))        
    poly.SetPoints(points)
    poly.SetLines(lines)
    poly.GetCellData().SetScalars(colors)

    output = self.GetOutput()
    output.DeepCopy(poly)