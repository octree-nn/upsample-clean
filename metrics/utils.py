import numpy as np
import point_cloud_utils as pcu


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    N, C = xyz.shape
    centroids = np.zeros((npoint, ), dtype=np.int32)
    distance = np.ones((N, )) * 1e10
    farthest = np.random.randint(0, N, (1, ))[0]

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.argmax(0)
    return centroids

def chamfer_distance(target_path, out_path):
    target = np.loadtxt(target_path)
    out = np.loadtxt(out_path) / 1.7
    dis = pcu.chamfer_distance(target, out, squared_distances=True)
    return dis

def hausdorff_distance(target_path, out_path):
    target = np.loadtxt(target_path)
    out = np.loadtxt(out_path) / 1.7
    dis = pcu.hausdorff_distance(target, out, squared_distances=True)
    return dis

def p2f(target_path, out_path):
    target_path = target_path[:-3]+'off'
    v, f = pcu.load_mesh_vf(target_path)
    out = np.loadtxt(out_path) / 1.7
    d, fi, bc = pcu.closest_points_on_mesh(out, v, f)
    d = np.mean(d)
    return d

