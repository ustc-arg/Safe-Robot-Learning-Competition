import numpy as np

class Modifier:
    def __init__(self, nominal_gates, params):
        self.nominal_gates = nominal_gates
        self.hold_ratio = params['hold_ratio']
        self.gates_detect_range = np.zeros(nominal_gates.shape[0])
        self.gates_delta_pos = np.zeros((nominal_gates.shape[0], 2), dtype=float)
    
    def refresh_gate_info(self, next_gate_id, next_gate_pos, temp_pos):
        if next_gate_id != -1:
            if (next_gate_pos != self.nominal_gates[next_gate_id]).all() and (self.gates_delta_pos[next_gate_id] == np.array([0.,0.])).all():
                self.gates_detect_range[next_gate_id] = np.linalg.norm(self.nominal_gates[next_gate_id] - temp_pos)
                self.gates_delta_pos[next_gate_id,:] = next_gate_pos - self.nominal_gates[next_gate_id]

    def get_des_pos_bias(self, temp_pos):
        bias = np.array([0.,0.])
        for i in range(self.nominal_gates.shape[0]):
            dist2gate = np.linalg.norm(self.nominal_gates[i] - temp_pos)
            bias += self._coeff(self.gates_detect_range[i], dist2gate) * self.gates_delta_pos[i,:]
        return bias

    def _coeff(self, detect_dist, temp_dist):
        if temp_dist >= detect_dist:
            return 0.
        elif temp_dist > self.hold_ratio * detect_dist:
            return (detect_dist - temp_dist) / ((1.0 - self.hold_ratio) * detect_dist)
        else:
            return 1.0
    
    def reset(self):
        self.gates_detect_range = np.zeros(self.nominal_gates.shape[0])
        self.gates_delta_pos = np.zeros((self.nominal_gates.shape[0], 2), dtype=float)
