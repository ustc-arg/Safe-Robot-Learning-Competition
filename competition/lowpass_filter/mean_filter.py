import numpy as np
class MeanFilter:
    def __init__(self, total_steps, initial_value):
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.filter_buff = np.ones([self.total_steps, 1]) * np.array(self.initial_value)

    def reset(self):
        self.filter_buff = np.ones([self.total_steps, 1]) * np.array(self.initial_value)

    def filter(self, input):
        for i in range(len(self.filter_buff)-1):
            self.filter_buff[i] = self.filter_buff[i+1]
        self.filter_buff[-1] = input
        output = np.mean(self.filter_buff, axis=0)
        return output
        
        
