import numpy as np
from pyibl import Agent
from cogtom.core.actions import Actions
from typing import Tuple


class IBLObserver(Agent):
    def __init__(self, env, default_utility):
        super(IBLObserver, self).__init__(name="IBLObserver", attributes=["action", "state_x", "state_y"],
                                          default_utility=default_utility)
        self.action_pool = Actions.get_actions_vector()
        self.actions = len(self.action_pool)
        self.env = env

        self.options = []
        self.generate_options()

    def generate_options(self):
        options = {}
        for i in range(1, self.env.width - 1):
            for j in range(1, self.env.height - 1):
                options[(i, j)] = []
                for a in range(self.actions):
                    options[(i, j)].append({"action": a, "state_x": i, "state_y": j})
        self.options = options

    def get_action(self, current_pos):
        action_selected = self.choose(choices=self.options[current_pos])
        return action_selected["action"]

    def update(self, *args, **kwargs):
        pass

