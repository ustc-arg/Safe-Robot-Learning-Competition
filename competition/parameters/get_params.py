import numpy as np

test_nominal_gates_value = [[0.5, -2.5, 0, 0, 0, -1.57, 0], [2, -1.5, 0, 0, 0, 0, 1], [0, 0.2, 0, 0, 0, 1.57, 1], [-0.5, 1.5, 0, 0, 0, 0, 0]]
test_nominal_obs_value = [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]

class ParamAssign:
    def __init__(self, initial_info, episode_reseed):
        self.init_state_random_dict = initial_info['initial_state_randomization']
        self.inertial_random_dict = initial_info['inertial_prop_randomization']
        self.gates_obs_random_dict = initial_info['gates_and_obs_randomization']
        self.disturbance_dict = initial_info['disturbances']
        self.episode_reseed_flag = episode_reseed
        self.gate_obs_pos_change = not ((initial_info['nominal_gates_pos_and_type'] == test_nominal_gates_value) and (initial_info['nominal_obstacles_pos'] == test_nominal_obs_value))
        self._get_uncertainty_level()
    
    def _get_uncertainty_level(self):
        state_len = len(self.init_state_random_dict)
        inertial_len = len(self.inertial_random_dict)
        gate_obs_len = len(self.gates_obs_random_dict)
        if not self.gate_obs_pos_change:
            if state_len == 0 and inertial_len == 0:
                self._load_level0_param()
            elif gate_obs_len == 0:
                self._load_level1_param()
            elif not self.episode_reseed_flag:
                self._load_level2_param()
            else:
                self._load_level3_param()
        else:
            if state_len == 0 and inertial_len == 0:
                self._load_level0_param()
            elif gate_obs_len == 0:
                self._load_level1_param()
            else:
                self._load_default_param()
        
    def _load_level0_param(self):
        self.adjustable_params = {'max_recursion_num': 6,\
                                'path_insert_point_dist_min': 0.25,\
                                'gate_collide_angle': np.pi/3,\
                                'gate_waypoint_safe_dist': 0.3,\
                                'traj_max_vel': 2.5,\
                                'traj_gamma': 25000,\
                                'replan_attempt_num': 2,\
                                
                                'time_stretch_horizon': 0.68,\
                                'pass_gate_vel': 1.8,\
                                'time_ahead_bias': 0.35,\
                                
                                'hold_ratio': 0.8}

    def _load_level1_param(self):
        self.adjustable_params = {'max_recursion_num': 6,\
                                'path_insert_point_dist_min': 0.25,\
                                'gate_collide_angle': np.pi * (55/180),\
                                'gate_waypoint_safe_dist': 0.3,\
                                'traj_max_vel': 2.5,\
                                'traj_gamma': 25000,\
                                'replan_attempt_num': 2,\
                                
                                'time_stretch_horizon': 0.68,\
                                'pass_gate_vel': 1.8,\
                                'time_ahead_bias': 0.35,\
                                
                                'hold_ratio': 0.8}

    def _load_level2_param(self):
        self.adjustable_params = {'max_recursion_num': 6,\
                                'path_insert_point_dist_min': 0.25,\
                                'gate_collide_angle': np.pi*(55/180),\
                                'gate_waypoint_safe_dist': 0.3,\
                                'traj_max_vel': 2.0,\
                                'traj_gamma': 25000,\
                                'replan_attempt_num': 2,\
                                
                                'time_stretch_horizon': 0.7,\
                                'pass_gate_vel': 1.0,\
                                'time_ahead_bias': 0.35,\
                                
                                'hold_ratio': 0.8} # tested by lxh & zyj

    def _load_level3_param(self):
        self.adjustable_params = {'max_recursion_num': 6,\
                                'path_insert_point_dist_min': 0.25,\
                                'gate_collide_angle': np.pi*(55/180),\
                                'gate_waypoint_safe_dist': 0.3,\
                                'traj_max_vel': 2.0,\
                                'traj_gamma': 2000,\
                                'replan_attempt_num': 3,\
                                
                                'time_stretch_horizon': 0.7,\
                                'pass_gate_vel': 1.0,\
                                'time_ahead_bias': 0.3,\
                                
                                'hold_ratio': 0.8}
        
    def _load_default_param(self):
        self.adjustable_params = {'max_recursion_num': 6,\
                                'path_insert_point_dist_min': 0.2,\
                                'gate_collide_angle': np.pi/4,\
                                'gate_waypoint_safe_dist': 0.3,\
                                'traj_max_vel': 1.5,\
                                'traj_gamma': 500,\
                                'replan_attempt_num': 3,\
                                
                                'time_stretch_horizon': 0.8,\
                                'pass_gate_vel': 0.8,\
                                'time_ahead_bias': 0.2,\
                                
                                'hold_ratio': 0.8}
    
    def get_params(self):
        return self.adjustable_params
    
    def get_default_params(self):
        self._load_default_param()
        return self.adjustable_params