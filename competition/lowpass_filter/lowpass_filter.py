class LowPassFilterData:
    """
    data structure of low pass filter
    """

    interval_1 = 0.0

    interval_2 = [0.0, 0.0, 0.0]

    interval_3 = 0.0

    interval_4 = 0.0

    interval_5 = [0.0, 0.0]


class LowPassFilter:
    """
    third order low pass filter
    """

    def __init__(self, num_dim):
        self.num_dim = num_dim
        self.data = [LowPassFilterData()] * num_dim

    def updateAxis(self, i, val):

        self.data[i].interval_1 = 0.3333 * val + 1.4803e-16 * self.data[i].interval_2[1]
        self.data[i].interval_2[2] = self.data[i].interval_1 - 0.3333 * self.data[i].interval_2[0]
        self.data[i].interval_3 = self.data[i].interval_2[2] + 2 * self.data[i].interval_2[1]
        self.data[i].interval_4 = self.data[i].interval_3 + self.data[i].interval_2[0]
        self.data[i].interval_5[1] = 0.5 * self.data[i].interval_4 + 5.5511e-17 * self.data[i].interval_5[0]
        val_filtered = self.data[i].interval_5[1] + self.data[i].interval_5[0]

        return val_filtered

    def shiftInterVals(self):

        for i in range(self.num_dim):
            self.data[i].interval_2[0] = self.data[i].interval_2[1]
            self.data[i].interval_2[1] = self.data[i].interval_2[2]
            self.data[i].interval_5[0] = self.data[i].interval_5[1]

    def filter(self, inputs):

        outputs = [0.0] * self.num_dim

        for i in range(self.num_dim):
            outputs[i] = self.updateAxis(i, inputs[i])

        self.shiftInterVals()

        return outputs

    def reset(self):
        self.data = [LowPassFilterData()] * self.num_dim

