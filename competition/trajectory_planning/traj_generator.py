import pybullet as p
from competition.trajectory_planning.generator_utils import *

class TrajGenerator:

    def __init__(self, params, gates, obstacles):

        print("Initializing trajectory generator...")

        self.obstacle_geo = params["obstacle_geo"]
        self.gate_geo = params["gate_geo"]
        self.gate_height = params["gate_height"]

        gates = np.array(gates)
        for i in range(gates.shape[0]):
            gates[i][2] = self.gate_height[int(gates[i][6])]
        self.gates = gates[0:6] if gates.ndim == 1 else gates[:,0:6]
        self.obstacles = np.array(obstacles)
        self.episode_len_sec = params["ctrl_time"]
        self.ctrl_freq = params["ctrl_freq"]
        self.ctrl_dt = 1.0 / self.ctrl_freq
        self.gate_sequence_fixed = params["gate_sequence_fixed"]
        self.start_pos = params["start_pos"]
        self.stop_pos = params["stop_pos"][0:3]
        self.accuracy = params["accuracy"]
        self.max_recursion_num = int(params["max_recursion_num"])

        self.uav_radius = params["uav_radius"]
        self.gate_collide_angle = params["gate_collide_angle"]
        self.gate_waypoint_safe_dist = params["gate_waypoint_safe_dist"]
        self.path_insert_point_dist_min = params["path_insert_point_dist_min"]

        self.traj_max_vel = params["traj_max_vel"]
        self.traj_gamma = params["traj_gamma"]

        self._gate_sequence_plan()
        self._find_init_path()
        self._path_correct()
        self._trajectory_plan()

    def _if_pos_collide_with_obstacles(self, pos, uav_radius=0.075, obstacle_geo=[1.0,0.075]):
        # obstacle_geo = [height, radius]
        # obstacle_pos = [x, y, 0]
        if self.uav_radius is not None:
            uav_radius = self.uav_radius
        if self.obstacle_geo is not None:
            obstacle_geo = self.obstacle_geo
        
        if pos[2] <= 0.03:
            return True
        
        for obs in self.obstacles:
            if pos[2] >= 0 and pos[2] <= obstacle_geo[0]:
                dist = np.linalg.norm(pos[0:2]-obs[0:2])
            else:
                dist = min(np.linalg.norm(pos-obs[:3]),
                            np.linalg.norm(pos-obs[:3]-np.array([0,0,obstacle_geo[0]])))
            
            if dist <= uav_radius+obstacle_geo[1]:
                return True
        return False
    
    def _if_pos_collide_with_gates(self, pos, uav_radius=0.075, gate_geo=[0.45, 0.05, 0.05]):
        # gate_geo = [edge_length, edge_thickness, support_radius]
        # gate_pos = [x, y, z, r, p, y]

        if self.uav_radius is not None:
            uav_radius = self.uav_radius
        if self.gate_geo is not None:
            gate_geo = self.gate_geo
        safe_angle = np.pi/4
        if self.gate_collide_angle is not None:
            safe_angle = self.gate_collide_angle
        
        detect_radius = (gate_geo[0]+2*gate_geo[1])/math.sqrt(2.0) + 0.02 # delta = 0.02m for robustness
        for gate in self.gates:
            if pos[2] < gate[2] - detect_radius:
                if np.linalg.norm(pos[0:2]-gate[0:2]) <= gate_geo[2] + uav_radius:
                    return True
            else:
                gate_to_pos_dir = pos - gate[0:3]
                if np.linalg.norm(gate_to_pos_dir) <= detect_radius + uav_radius:
                    r = gate[3]
                    p = gate[4]
                    y = gate[5]
                    gate_dir = np.array([-math.cos(p)*math.sin(y),
                                        math.cos(r)*math.cos(y)-math.sin(r)*math.sin(p)*math.sin(y),
                                        math.cos(y)*math.sin(r)+math.cos(r)*math.sin(p)*math.sin(y)])
                    angle = get_vecs_angle(gate_dir, gate_to_pos_dir)
                    if angle > safe_angle and angle < np.pi-safe_angle:
                        return True
        return False
    
    def _gate_sequence_plan(self):
        if self.gate_sequence_fixed:
            seq = range(0, len(self.gates)+2)
        else:
            dist_matrix = get_distance_matrix(gates=self.gates, start_point=self.start_pos, end_point=self.stop_pos)
            seq, _ = tsp_dp(dist_matrix)
            seq = np.array(seq)
        self.gate_sequence = seq

    def _find_init_path(self):
        gates = self.gates
        stop_gate = self.stop_pos
        stop_gate.extend([0, 0, 0])
        gates = np.concatenate((gates, [stop_gate])) if len(gates) != 0 else np.array([stop_gate])
        safe_angle = np.pi/4
        if self.gate_collide_angle is not None:
            safe_angle = self.gate_collide_angle
        safe_rad = self.gate_waypoint_safe_dist
        seq = self.gate_sequence
        modified_path = [self.start_pos]

        def _get_gate_dir_vec(gate):
            r = gate[3]
            p = gate[4]
            y = gate[5]
            return np.array([-math.cos(p)*math.sin(y),
                            math.cos(r)*math.cos(y)-math.sin(r)*math.sin(p)*math.sin(y),
                            math.cos(y)*math.sin(r)+math.cos(r)*math.sin(p)*math.sin(y)])

        for i in range(len(seq)-2):
            temp_gate = gates[seq[i]]
            next_gate = gates[seq[i+1]]
            gate_dir = _get_gate_dir_vec(temp_gate)
            last_point = np.array(modified_path[-1])
            front_vec = temp_gate[0:3] - last_point
            latter_vec = next_gate[0:3] - temp_gate[0:3]
            front_angle = get_vecs_angle(front_vec, gate_dir)
            latter_angle = get_vecs_angle(latter_vec, gate_dir)

            front_angle_flag = 0 # forward 1, backward -1
            if front_angle > safe_angle and front_angle < np.pi-safe_angle:
                if front_angle <= np.pi/2:
                    _pos = temp_gate[0:3] - safe_rad*gate_dir
                    modified_path.append(_pos.tolist())
                    front_angle_flag = 1
                else:
                    _pos = temp_gate[0:3] + safe_rad*gate_dir
                    modified_path.append(_pos.tolist())
                    front_angle_flag = -1
            else:
                if front_angle <= np.pi/2:
                    front_angle_flag = 1
                else:
                    front_angle_flag = -1

            modified_path.append(temp_gate[0:3].tolist())
            
            gate_parallel_angle_threshold = 0.17 # determine whether gate's angle is parallel to the desired path direction

            if latter_angle <= np.pi/2 - gate_parallel_angle_threshold and front_angle_flag == -1:
                _t_pos = temp_gate[0:3] - 0.05*gate_dir
                modified_path.append(_t_pos.tolist())
                _pos = temp_gate[0:3] + (safe_rad - 0.05)*gate_dir
                modified_path.append(_pos.tolist())
            elif latter_angle > np.pi/2 + gate_parallel_angle_threshold and front_angle_flag == 1:
                _t_pos = temp_gate[0:3] + 0.05*gate_dir
                modified_path.append(_t_pos.tolist())
                _pos = temp_gate[0:3] - (safe_rad - 0.05)*gate_dir
                modified_path.append(_pos.tolist())
            elif latter_angle > safe_angle and latter_angle < np.pi-safe_angle:
                if front_angle_flag == 1:
                    _pos = temp_gate[0:3] + safe_rad*gate_dir
                    modified_path.append(_pos.tolist())
                else:
                    _pos = temp_gate[0:3] - safe_rad*gate_dir
                    modified_path.append(_pos.tolist())

        modified_path.append(gates[-1][0:3].tolist())
        self.init_path = modified_path

        print("\033[0;33;40mInitial infeasible path found.\033[0m")

    def _path_correct(self):
        waypoints = np.array(self.init_path)

        def _find_viable_point(intp1, intp2, dir_num=32):
            world_vec = (intp2-intp1) / np.linalg.norm(intp2-intp1)
            if world_vec[0] == 0 and world_vec[2] == 0:
                roll = math.pi/2 if world_vec[1] == -1 else -math.pi/2
                pitch = 0.0
            else:
                pitch = math.atan2(world_vec[0], world_vec[2])
                cos_roll = world_vec[0]/math.sin(pitch)
                roll = math.atan2(-world_vec[1], cos_roll)
            
            rot_M = np.array([[math.cos(pitch), math.sin(pitch)*math.sin(roll), math.sin(pitch)*math.cos(roll)],
                            [0, math.cos(roll), -math.sin(roll)],
                            [-math.sin(pitch), math.cos(pitch)*math.sin(roll), math.cos(pitch)*math.cos(roll)]])
            
            angles = np.linspace(0.0, 2*math.pi, num=dir_num, endpoint=False)
            directions = np.array([[math.cos(a), math.sin(a), 0] for a in angles])
            mid_point = (intp1 + intp2) / 2
            trans_directions = np.array([rot_M.dot(dir.T) for dir in directions])

            step_len = self.accuracy
            max_step_num = 1000*dir_num
            step_count = 0
            while step_count < max_step_num:
                fit_dist = np.inf
                best_point = None
                for d in trans_directions:
                    temp_point = mid_point + step_len*d + 0.1 # 0.1 for robustness
                    dist_to_l = point2line_dist(intp1, intp2, temp_point)
                    if not self._if_pos_collide_with_obstacles(temp_point) and not self._if_pos_collide_with_gates(temp_point) and dist_to_l < fit_dist:
                        fit_dist = dist_to_l
                        best_point = temp_point
                if best_point is not None:
                    return best_point
                step_count = step_count + 1
                step_len = step_len + step_count*self.accuracy
            return None

        def _intersect(p1, p2, count=0):
            dist = np.linalg.norm(p2-p1)
            if dist < self.path_insert_point_dist_min or count >= self.max_recursion_num:
                return np.concatenate(([p1], [p2]))
            seg_num = int(dist/self.accuracy)
            seg_points = np.linspace(p1, p2, num=seg_num, endpoint=True)
            flag = False
            intersect_p1 = None
            intersect_p2 = None
            index = 0
            for p in seg_points:
                index = index + 1
                if self._if_pos_collide_with_obstacles(p) or self._if_pos_collide_with_gates(p):
                    intersect_p1 = p
                    flag = True
                    break
                
            if flag is True:
                for p in seg_points[index:]:
                    if not self._if_pos_collide_with_obstacles(p) or self._if_pos_collide_with_gates(p):
                        intersect_p2 = p
                
                mid_point = _find_viable_point(intersect_p1, intersect_p2)
                front_way = _intersect(p1, mid_point, count+1)
                after_way = _intersect(mid_point, p2, count+1)
                return np.concatenate((front_way, after_way[1:]))
            else:
                return np.concatenate(([p1], [p2]))

        path = [waypoints[0]]
        for i in range(len(waypoints)-1):
            start_p = waypoints[i]
            end_p = waypoints[i+1]
            segment = _intersect(start_p, end_p)
            path = np.concatenate((path, segment[1:]))

        self.raw_path = path
        self.path = path

        print("\033[0;32;40mInitial feasible path found!\033[0m")

    def _trajectory_plan(self, traj_max_vel=1.0, traj_gamma=150):
        waypoints = self.path
        if self.traj_max_vel is not None:
            traj_max_vel = self.traj_max_vel
        if self.traj_gamma is not None:
            traj_gamma = self.traj_gamma
        
        print("\033[7mConducting trajectory planning...\033[0m")

        generator = trajGenerator(np.array(waypoints), max_vel=traj_max_vel, gamma=traj_gamma)
        self.traj_generator = generator

        timestamp = np.ones(int(self.ctrl_freq*self.episode_len_sec)) * self.ctrl_dt
        timestamp[0] = 0.0
        timestamp = np.cumsum(timestamp)
        self.timestamp = timestamp
        self.pos_trajectory = np.array([generator.get_des_state(t).pos for t in timestamp])

        print("\033[0;32;40mPlanning finished.\033[0m")
    
    def _intersect_point(self, p): # find a collision-free point from p that is close to self.path and insert that point to self.path
        dist = np.Inf
        index = 0
        mid_point = None
        safe_dist = self.uav_radius + self.obstacle_geo[1] + 0.1 # delta = 0.1m for robustness
        for i in range(self.path.shape[0]-1):
            temp_dist = point2segment_dist(self.path[i], self.path[i+1], p)
            if temp_dist < dist:
                index = i
                dist = temp_dist
            if temp_dist <= 0.01:
                mid_point = p
                break
        if mid_point is None:
            project_point = point2line_project(self.path[index], self.path[index+1], p)
            direction = project_point - p
            direction_norm = np.linalg.norm(direction)
            if direction_norm >= safe_dist:
                mid_point = project_point
            else:
                mid_point = p + direction / direction_norm * safe_dist # TODO safety might not be guaranteed for this safe_dist
        
        for temp_p in self.path:
            if np.linalg.norm(mid_point - temp_p) <= self.path_insert_point_dist_min:
                return 0
        self.path = np.insert(self.path, index+1, mid_point, axis=0)
        return 1
    
    def trajectory_replan(self):
        collide_flag = False
        in_pos = []
        out_pos = []

        for p in self.pos_trajectory:
            if collide_flag is False:
                if self._if_pos_collide_with_obstacles(p) or self._if_pos_collide_with_gates(p):
                    in_pos.append(p)
                    collide_flag = True
            else:
                if not self._if_pos_collide_with_obstacles(p) and not self._if_pos_collide_with_gates(p):
                    out_pos.append(p)
                    collide_flag = False
        
        added_point = 0
        for i in range(len(out_pos)):
            added_point += self._intersect_point((in_pos[i]+out_pos[i])/2)
        if added_point is not 0:
            self._trajectory_plan()
            return True
        else:
            return False

    def plot_trajectory(self, pclient, URDF_DIR):
        for point in self.path:
            p.loadURDF(os.path.join(URDF_DIR, "sphere.urdf"),
                        basePosition=[point[0], point[1], point[2]],
                        baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                        physicsClientId=pclient,
                        useFixedBase=True)
        _step = int(self.ctrl_freq*self.episode_len_sec/200)
        for i in range(_step, self.pos_trajectory.shape[0], _step):
            p.addUserDebugLine(lineFromXYZ=[self.pos_trajectory[i-_step][0], self.pos_trajectory[i-_step][1], self.pos_trajectory[i-_step][2]],
                               lineToXYZ=[self.pos_trajectory[i][0], self.pos_trajectory[i][1], self.pos_trajectory[i][2]],
                               lineColorRGB=[1, 0, 0],
                               physicsClientId=pclient)
        p.addUserDebugLine(lineFromXYZ=[self.pos_trajectory[i][0], self.pos_trajectory[i][1], self.pos_trajectory[i][2]],
                           lineToXYZ=[self.pos_trajectory[-1][0], self.pos_trajectory[-1][1], self.pos_trajectory[-1][2]],
                           lineColorRGB=[1, 0, 0],
                           physicsClientId=pclient)
