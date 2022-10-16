from collections import deque
import torch
import numpy as np
from lowpass_filter.lowpass_filter import LowPassFilter
from lowpass_filter.mean_filter import MeanFilter
from param_identification.net.chennet_lstm import *

class PrepareData:
    def __init__(self, num_timesteps_input, feature_num):
        self.num_timesteps_input = num_timesteps_input
        self.feature_num = feature_num
        self.data_image = np.zeros([self.num_timesteps_input, self.feature_num])
        self.data_image = self.data_image[np.newaxis, :, :]
        self.lpf = LowPassFilter(1)
        self.coef_z_error = 100
        
    def update_image_data(self, pos_ref_z, pos_z):    
        row = np.array([pos_ref_z - pos_z,
                        pos_ref_z - pos_z])
        row = row * self.coef_z_error
        for i in range(self.data_image.shape[1]-1):
            self.data_image[0][i] = self.data_image[0][i+1]
        self.data_image[0][-1] = row

        return self.data_image
    
    def reset(self):
        # important: choose whether to re-identify parameters
        self.data_image = np.zeros([self.num_timesteps_input, self.feature_num])
        self.data_image = self.data_image[np.newaxis, :, :]
        self.lpf.reset()

class Compensator:
    def __init__(self, nominal_mass, gravity=9.8):
        self.nominal_mass = nominal_mass
        self.mass_estimation = self.nominal_mass
        self.mean_filter = MeanFilter(total_steps=10, initial_value=0.0)
        self.gravity = np.array([0.0, 0.0, gravity])

    def cal_compensation(self, mass_error_percent, current_acc):
        mass_error_percent_filtered = self.mean_filter.filter(mass_error_percent)
        mass_error_percent_filtered = mass_error_percent_filtered[0]
        # todo: delta mass need check
        self.mass_estimation = (1 + mass_error_percent_filtered*0.01) * self.nominal_mass
        delta = mass_error_percent_filtered * 0.01 * self.gravity + mass_error_percent_filtered * 0.01 * current_acc
        return np.array([delta[0], delta[1], delta[2]])
    
    def reset(self, nominal_mass=None):
        if nominal_mass is not None:
            self.nominal_mass = nominal_mass
        self.mean_filter.reset()

class PredictNet:
    def __init__(self, model_path, nominal_value=0.03454):
        self.model = LSTM_Net(in_dim=2)
        self.model.load_state_dict(torch.load(model_path))
        self.buffer = PrepareData(num_timesteps_input=5, feature_num=2)
        self.compensator = Compensator(nominal_value)
        self.nominal_value = nominal_value
        self.mass_error_percent = 0.
        self.iter = 0
        self.error_deque = deque([], maxlen=20)
        self.success = False

        self.mean_filter = MeanFilter(total_steps=50, initial_value=0.0)

    def predict(self, pos_ref_z, pos_z):
        # input: reference z pos, real z pos
        # output: mass error percentage
        if not self.success:
            input = self.buffer.update_image_data(pos_ref_z, pos_z)
            input = torch.from_numpy(input).float()
            output = self.model(input).detach().numpy().squeeze()
            output_fitered = self.mean_filter.filter(output)
            self.error_deque.append(output_fitered[0])
            if self.iter <= 20:
                self.iter += 1
            else:
                self.validate()
    
    def validate(self):
        if self.success: return True
        if len(self.error_deque) >= 5:
            mean_val = sum(self.error_deque) / len(self.error_deque)
            if sum(list(map(abs, map(lambda x: x - mean_val, self.error_deque)))) <= 10:
                self.success = True
                self.mass_error_percent = sum(self.error_deque) / len(self.error_deque)
                print("\033[0;37;42mIdentified mass:\033[0m", self.nominal_value * (1 + self.mass_error_percent * 0.01), "kg")
    
    def reset(self, model_path=None):
        if model_path is not None:
            self.model = torch.load(model_path).eval()
        self.buffer.reset()
        self.compensator.reset()
        self.mass_error_percent = 0.
        self.error_deque = deque([], maxlen=20)
        self.success = False
    
    def compensate(self, acc_ref):
        if self.success:
            return self.compensator.cal_compensation(self.mass_error_percent, acc_ref)
        else:
            return np.zeros(3)