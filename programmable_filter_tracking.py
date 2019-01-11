def run_algo(p0, p1, t):

    # transform polydata to numpy points
    import numpy as np
    from paraview.vtk.util import numpy_support
    from paraview import vtk
    import os


    if t == 0 and os.path.exists("/home/matthieu/Dev/Tools/filesave.bin.npy"):
        print "remove filesave"
        os.remove("/home/matthieu/Dev/Tools/filesave.bin.npy")
        return None

    if p0 is None or p1 is None:
        return None

    pts0 = numpy_support.vtk_to_numpy(p0.GetPoints().GetData()).copy()
    pts1 = numpy_support.vtk_to_numpy(p1.GetPoints().GetData()).copy()


    x_bounds = [0, 30]    # [min, max[ in meters
    y_bounds = [-40, 40]
    z_bounds = [-1, 1]

    scale_factor = 10

    x_bounds[0] *= scale_factor
    x_bounds[1] *= scale_factor
    y_bounds[0] *= scale_factor
    y_bounds[1] *= scale_factor
    z_bounds[0] *= scale_factor
    z_bounds[1] *= scale_factor

    x_range = x_bounds[1] - x_bounds[0]
    y_range = y_bounds[1] - y_bounds[0]
    z_range = z_bounds[1] - z_bounds[0]


    grid0 = np.zeros((x_range, y_range, z_range))
    pts0 = pts0[:, :3] * scale_factor
    pts0 -= [x_bounds[0], y_bounds[0], z_bounds[0]]
    pts0 = np.floor(pts0).astype(int)
    pts0 = pts0[pts0[:, 0] >= 0]
    pts0 = pts0[pts0[:, 0] < x_range]
    pts0 = pts0[pts0[:, 1] >= 0]
    pts0 = pts0[pts0[:, 1] < y_range]
    pts0 = pts0[pts0[:, 2] >= 0]
    pts0 = pts0[pts0[:, 2] < z_range]

    grid0[pts0[:, 0], pts0[:, 1], pts0[:, 2]] = 1


    grid1 = np.zeros((x_range, y_range, z_range))
    pts1 = pts1[:, :3] * scale_factor
    pts1 -= [x_bounds[0], y_bounds[0], z_bounds[0]]
    pts1 = np.floor(pts1).astype(int)
    pts1 = pts1[pts1[:, 0] >= 0]
    pts1 = pts1[pts1[:, 0] < x_range]
    pts1 = pts1[pts1[:, 1] >= 0]
    pts1 = pts1[pts1[:, 1] < y_range]
    pts1 = pts1[pts1[:, 2] >= 0]
    pts1 = pts1[pts1[:, 2] < z_range]

    grid1[pts1[:, 0], pts1[:, 1], pts1[:, 2]] = 1

    diff = grid0 - grid1
    xx, yy, zz = np.where(diff == 1)
    pts = np.vstack((xx, yy, zz)).T
    pts = np.ascontiguousarray(pts, dtype=np.float32)

    pts += [x_bounds[0], y_bounds[0], z_bounds[0]]
    pts /= scale_factor


    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.9, min_samples=30).fit_predict(pts)



    import numpy as np


    THRESHOLD_TRACKS_ASSOCIATION = 3

    def distance(pos1, pos2):
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))


    class track:
        def __init__(self, color=None):
            self.poses = {}     # contains frame_id : position
            self.dims = {}      # contains frame_id : dimension
            if color is not None:
                self.color = color
            else:
                self.color = np.random.randint(0, 255, 3)


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
        def __init__(self):
            self.tracks = []

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
            if dist_min > THRESHOLD_TRACKS_ASSOCIATION:
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


    if t > 0:
        tm = load("/home/matthieu/Dev/Tools/filesave.bin")
    


    print "time ", t
    for k in range(max(clustering) + 1):
        center = np.mean(pts[clustering == k, :], axis=0)
        xmin, ymin, zmin = np.min(pts[clustering == k, :], axis=0)
        xmax, ymax, zmax = np.max(pts[clustering == k, :], axis=0)
        xsize = max(xmax - center[0], center[0] - xmin)
        ysize = max(ymax - center[1], center[1] - ymin)
        zsize = max(zmax - center[2], center[2] - zmin)
        #print "new observation ", t
        tm.add_observation(t, center, [xsize, ysize, zsize]) ## frame id 


    save(tm, "/home/matthieu/Dev/Tools/filesave.bin")

    lines = []
    colors = []
    bbs = []
    idx = 0

    #moving_objects = tm.get_moving_tracks_idx()     # list of tracks considered moving
    #for i in moving_objects:           # can be used to hide some small tracks
    for i in range(len(tm.tracks)):
        track = tm.tracks[i]
        idf, pos, dim = track.last_observation()
        if idf != t:
            continue
        xmin, ymin, zmin = pos - dim
        xmax, ymax, zmax = pos + dim
        # bounding box
        bb = np.array([[xmin, ymin, zmin],
                       [xmax, ymin, zmin],
                       [xmax, ymax, zmin],
                       [xmin, ymax, zmin],
                       [xmin, ymin, zmax],
                       [xmax, ymin, zmax],
                       [xmax, ymax, zmax],
                       [xmin, ymax, zmax]])
        bbs.append(bb)
        # bounding box edges
        lines.append([idx + k for k in [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]])
        idx += 8

        # past positions
        positions = [track.poses[fid] for fid in sorted(track.poses.keys())]
        bbs.append(positions)

        # past positions line
        lines.append([idx + j for j in range(len(positions))])
        idx += len(positions)

        # colors        
        colors.append(track.color)
        colors.append(track.color)

    if len(bbs) > 0:
        bbs = np.vstack(bbs)
        colors = np.vstack(colors).astype(np.uint8)
        return bbs, lines, colors
    else:
        return None,  None, None

