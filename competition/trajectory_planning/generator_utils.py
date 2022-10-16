import os
import math
import numpy as np
from competition.trajectory_planning.TrajGen.trajGen import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def point2line_dist(lp1, lp2, p):
    vec1 = lp1-p
    vec2 = lp2-p
    vec3 = lp2-lp1
    return np.linalg.norm(np.cross(vec1,vec2)) / np.linalg.norm(vec3)

def point2line_project(lp1, lp2, p):
    vec1 = p-lp1
    vec2 = lp2-lp1
    t = np.sum(vec1*vec2) / (np.linalg.norm(vec2)**2)
    return lp1 + t*vec2

def point2segment_dist(sp1, sp2, p):
    vec1 = p-sp1
    vec2 = sp2-sp1
    vec3 = p-sp2
    seg_length = np.linalg.norm(vec2)
    r = np.sum(vec1*vec2) / (seg_length*seg_length)
    if r <= 0:
        return np.linalg.norm(vec1)
    elif r >= 1:
        return np.linalg.norm(vec3)
    else:
        return np.linalg.norm(np.cross(vec1,vec3)) / seg_length


def get_vecs_angle(vec1, vec2):
    vec_len1 = np.linalg.norm(vec1)
    vec_len2 = np.linalg.norm(vec2)
    if vec_len1 == 0 or vec_len2 == 0:
        return 0.
    t = np.sum(vec1*vec2) / (vec_len1*vec_len2)
    t = max(min(t, 1.0), -1.0)
    return math.acos(t)


def get_distance_matrix(gates=[], start_point=[0.0,0.0,0.0], end_point=[5.0,5.0,5.0]):

    def gates_to_end_dist(num_of_gates):
        dist = np.zeros(num_of_gates+1)
        dist[0] = np.Inf
        if end_point is not None:
            for i in range(1, num_of_gates+1):
                dist[i] = np.linalg.norm(gates[i-1][0:3]-end_point)
        return dist

    def start_to_gates_dist(num_of_gates):
        dist = np.zeros(num_of_gates+1)
        dist[0] = np.Inf
        if start_point is not None:
            for i in range(1, num_of_gates+1):
                dist[i] = np.linalg.norm(gates[i-1][0:3]-start_point)
        return dist
    
    num_of_gates = len(gates)
    dist_matrix = np.zeros((num_of_gates+1, num_of_gates+1))
    dist_matrix[:][0] = gates_to_end_dist(num_of_gates)
    dist_matrix[0][:] = start_to_gates_dist(num_of_gates)

    gates = np.array(gates)
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    for i in range(1, num_of_gates+1):
        dist_matrix[0][i] = np.linalg.norm(gates[i-1][0:3]-start_point)
    for j in range(num_of_gates):
        for k in range(num_of_gates):
            dist_matrix[j+1][k+1] = np.linalg.norm(gates[j][0:3] - gates[k][0:3]) if j != k else 0
    
    return dist_matrix

def tsp_dp(dist_matrix):
    num_of_mid_point = dist_matrix.shape[0]-1

    dp_array = [[-1.0]*(2**num_of_mid_point) for i in range(dist_matrix.shape[0])]
    for i in range(dist_matrix.shape[0]):
        dp_array[i][0] = 0
    dp_array[0][0] = np.Inf

    path_table = np.ones((dist_matrix.shape[0], 2**num_of_mid_point)) * (-1)

    def set_to_num(set):
        num = 0
        for s in set:
            num = num + 2**(s-1)
        return num
    
    def _dp(start_point, set_to_go):
        if dp_array[start_point][set_to_num(set_to_go)] != -1.0:
            return dp_array[start_point][set_to_num(set_to_go)]
        else:
            min_len = np.Inf
            for s in set_to_go:
                start_p = s
                sub_set = [it for it in set_to_go if it is not s]
                temp_len = _dp(start_p, sub_set) + dist_matrix[start_point][s]
                if temp_len < min_len:
                    min_len = temp_len
                    path_table[start_point][set_to_num(set_to_go)] = s
            dp_array[start_point][set_to_num(set_to_go)] = min_len
            return min_len
    
    def track_back():
        track_array = []
        future_set = range(dist_matrix.shape[0])
        temp_point = 0
        while len(future_set) > 0:
            track_array.append(temp_point)
            future_set = [it for it in future_set if it != temp_point]
            temp_point = int(path_table[temp_point][set_to_num(future_set)])
        return track_array

    min_length = _dp(0, range(1, dist_matrix.shape[0]))
    trajectory = track_back()
    trajectory.append(num_of_mid_point+1)
    
    return trajectory, min_length
