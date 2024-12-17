import math
import numpy as np
from scipy.stats import randint

class UniformGenModel:
    value_min = 0
    value_max = 2

    def init_params(self):
        pass

    def generate(self):
        return randint.rvs(self.value_min, self.value_max + 1)



