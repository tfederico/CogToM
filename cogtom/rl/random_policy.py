from numpy.random import dirichlet
from cogtom.core.actions import Actions
import numpy as np


class RandomPolicy:
    def __init__(self, alpha: tuple):
        self.alpha = alpha
        self.action_pool = Actions.get_actions()
        self.actions = len(self.action_pool)
        self.dirichlet = dirichlet([self.alpha]*self.actions)

    def get_action(self, current_pos):
        index = np.random.choice(range(self.actions), p=self.dirichlet)
        return self.action_pool[index]

    def update(self, *args, **kwargs):
        pass

