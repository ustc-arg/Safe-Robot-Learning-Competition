import os
import numpy as np
import competition.param_identification.predict_mass_cal_compensation as Mass

class ParamIdentifier:

    def __init__(self, dt, nominal_value):
        # nominal_value: [g, m, Ixx, Iyy, Izz]
        self.dt = dt
        self.nominal_value = nominal_value
        self.mass_model_path = os.path.join(os.path.dirname(__file__), 'log/LSTM_Net_1005/model_v0.pt')
        self.mass_model = Mass.PredictNet(model_path=self.mass_model_path, nominal_value=self.nominal_value[1])
        self.temp_acc = np.zeros(3)
    
    def reset(self):
        self.mass_model.reset()

    def identify(self, obs_buffer, act_buffer, reference, flag):
        # obs_buffer[i]: [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
        # act_buffer[i]: [f1, f2, f3, f4]
        # reference: [ref_x, ref_y, ref_z, ref_vx, ref_vy, ref_vz, ref_ax, ref_ay, ref_az, ref_yaw]
        self.temp_acc = (np.array([0., 0., obs_buffer[-1][5] - obs_buffer[-2][5]]) if len(obs_buffer) > 1 else np.array([0., 0., obs_buffer[-1][5]])) / self.dt
        if flag:
            self.mass_model.predict(reference[2], obs_buffer[-1][4])
    
    def examine_validation(self):
        pass
        return True
    
    def reference_signal_bias(self):
        return self.mass_model.compensate(self.temp_acc)
