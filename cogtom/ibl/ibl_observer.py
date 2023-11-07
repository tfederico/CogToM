import numpy as np
from pyibl import Agent
from cogtom.core.actions import Actions
from typing import Tuple


class IBLAgent(Agent):
    def __init__(self, env, default_utility, goal_map):
        super(IBLAgent, self).__init__(name="IBLAgent", attributes=["action", "state_x", "state_y"], default_utility=default_utility)
        self.goal = None
        self.action_pool = Actions.get_actions_vector()
        self.actions = len(self.action_pool)
        self.last_action = 0

        self.goal_map = goal_map

        self.env = env
        # self.select_position()
        self.options = []
        self.generate_options()
        self.outcome_goals = None

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

    def add_world(self, world):
        self.env = world

    def add_outcome(self, outcome_goals):
        self.outcome_goals = outcome_goals

    # def get_position(self):
    #     return self.x, self.y

    def get_last_action(self):
        return self.last_action

    def populate(self, choices, outcome, when=None):
        self.populate(choices=choices, outcome=outcome, when=when)

    def step(self):
        pass

    # def select_position(self):
    #     walls_cells = self.env.get_walls()
    #     goals_cells = list(self.env.get_goals().values())
    #
    #     cells = []
    #     for i in range(self.env.width):
    #         for j in range(self.env.height):
    #             cells.append((i, j))
    #
    #     empty_cells = list(set(cells) - set(walls_cells) - set(goals_cells))
    #
    #     # select a random cell from empty cells
    #     np.random.shuffle(empty_cells)
    #
    #     self.x = empty_cells[0][0]
    #     self.y = empty_cells[0][1]

