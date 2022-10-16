import numpy as np
import math

class Stretcher:
    def __init__(self, timestamp, pos_traj, nominal_gates, params):
        assert len(timestamp) == len(pos_traj)
        self.dt = params["dt"]
        self.max_vel = params["max_vel"]
        self.stretch_horizon = params["time_stretch_horizon"] # [-horizon,+horizon]范围
        self.pass_gate_vel = min(params["pass_gate_vel"], self.max_vel)
        self.time_ahead = params["time_ahead_bias"]
        self.add_target_pos = params["add_target_pos"]
        if params["add_target_pos"]:
            target_pos_gate = np.append(pos_traj[-1], [0.,0.,0.])
            nominal_gates = np.append(nominal_gates, [target_pos_gate], axis=0)
        self.original_timestamp = timestamp
        self._get_key_timestamp(pos_traj, nominal_gates)
    
    def _get_key_timestamp(self, pos_traj, nominal_gates):
        key_timestamp = []
        key_velstamp = []
        if len(nominal_gates) > 0:
            temp_gate_index = 0
            for i in range(len(pos_traj)-1):
                temp_dist = np.linalg.norm(pos_traj[i]-nominal_gates[temp_gate_index][0:3])
                if temp_dist <= 0.05 and temp_dist <= np.linalg.norm(pos_traj[i+1]-nominal_gates[temp_gate_index][0:3]):
                    temp_gate_index += 1
                    key_timestamp.append(self.original_timestamp[i])
                    key_velstamp.append(np.linalg.norm(pos_traj[i+1]-pos_traj[i])/self.dt)
                    if temp_gate_index >= len(nominal_gates):
                        break
        self.key_timestamp = np.array(key_timestamp)
        self.key_velstamp = np.array(key_velstamp)
        if self.add_target_pos: self.key_velstamp[-1] = 0.0
    
    def _coefficient(self, t):
        _c = 0.0
        sigma = self.stretch_horizon / 2.5
        for i in range(self.key_timestamp.shape[0]):
            k_t = self.key_timestamp[i]
            if i == self.key_timestamp.shape[0] - 1:
                flow_rate = 0.1
            else:
                flow_rate = min(self.pass_gate_vel / self.key_velstamp[i], 1.0) if self.key_velstamp[i] > 0 else 1.0
            _c += (1.0 - flow_rate) * math.exp(-math.pow((t-k_t+self.time_ahead), 2) / 2.0 / math.pow(sigma, 2))
        return 1.0 - _c
    
    def resample(self, generator):
        t = 0.0
        timestamp = []
        pos_traj = []
        vel_traj = []
        acc_traj = []
        yaw_traj = []
        while t < self.original_timestamp[-1]:
            timestamp.append(t)
            c = self._coefficient(t)
            pos_ref = generator.get_des_state(t).pos
            vel_ref = c * generator.get_des_state(t).vel
            acc_ref = c * generator.get_des_state(t).acc
            yaw_ref = c * generator.get_des_state(t).yaw
            t += c * self.dt
            pos_traj.append(pos_ref)
            vel_traj.append(vel_ref)
            acc_traj.append(acc_ref)
            yaw_traj.append(yaw_ref)
        self.timestamp = timestamp
        self.pos_traj = np.array(pos_traj)
        self.vel_traj = np.array(vel_traj)
        self.acc_traj = np.array(acc_traj)
        self.yaw_traj = np.array(yaw_traj)
        return self.pos_traj, self.vel_traj, self.acc_traj, self.yaw_traj