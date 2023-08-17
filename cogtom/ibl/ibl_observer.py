import numpy as np
from pyibl import Agent
from cogtom.core.actions import Actions


class IBLObserver(Agent):
    def __init__(self, world, default_utility):
        super(IBLObserver, self).__init__("My Agent", ["action", "state_y", "state_x"], default_utility=default_utility)
        self.y = None
        self.x = None
        self.goal = None
        self.action_pool = Actions.get_actions_vector()
        self.actions = len(self.action_pool)
        self.last_action = 0

        self.world = world
        self.select_position()
        self.options = []
        self.generate_options()
        self.outcome_goals = np.zeros(self.world.num_goal)

    def generate_options(self):
        options = {}
        for i in range(1, self.world.height - 1):
            for j in range(1, self.world.width - 1):
                options[(i, j)] = []
                for a in range(self.actions):
                    options[(i, j)].append({"action": a, "state_y": i, "state_x": j})
        self.options = options

    def select_action(self):
        action_selected = self.choose(*self.options[(self.y, self.x)])
        return action_selected["action"]

    def add_world(self, world):
        self.world = world

    def add_outcome(self, outcome_goals):
        self.outcome_goals = outcome_goals

    def get_position(self):
        return self.y, self.x

    def get_last_action(self):
        return self.last_action

    def move(self, init_x, init_y, max_step):
        self.last_goal_consumed = None
        self.trajectory = []
        self.action_history = []
        is_done = False
        self.set_position(init_x, init_y)

        self.trajectory.append((init_x, init_y))
        preferred_goal = np.argmax(self.outcome_goals)
        step_preferred_goal = max_step

        delay = {}
        count_step = 0
        for step in range(0, max_step):
            count_step += 1
            action = self.select_action()
            self.last_action = action
            new_pos = np.array([self.y, self.x]) + self.action_pool[action]
            if self.world.get_walls()[new_pos[0], new_pos[1]] == 1:
                new_pos = np.array([self.y, self.x])
                self.last_action = 4

            if np.sum(self.world.get_goals(), axis=0)[new_pos[0], new_pos[1]] == 1:
                is_done = True
                for g in range(self.world.num_goal):
                    if self.world.get_goals()[g, new_pos[0], new_pos[1]] == 1:
                        if g == preferred_goal:
                            step_preferred_goal += 1
                        self.last_goal_consumed = g
                        break
            delay[step] = self.respond()

            self.y = new_pos[0]
            self.x = new_pos[1]

            self.trajectory.append((self.x, self.y))
            self.action_history.append(self.last_action)

            if is_done:
                break
        if self.last_goal_consumed == preferred_goal:
            step_preferred_goal = step
        if self.last_goal_consumed is not None:
            for i in range(len(self.action_history)):
                if self.action_history[i] == 4:
                    delay[i].resolve(-0.05)
                else:
                    delay[i].resolve(self.outcome_goals[self.last_goal_consumed])
        else:
            for i in range(len(self.action_history)):
                if self.action_history[i] == 4:
                    delay[i].resolve(-0.05)
                else:
                    delay[i].resolve(-0.01)
        return self.last_goal_consumed, count_step, step_preferred_goal

    def select_position(self):
        i, j = np.where(self.world.get_walls() + np.sum(self.world.get_goals(), axis=0) == 0)
        empty_cells = np.random.permutation(len(i))
        self.y = i[empty_cells[0]]
        self.x = j[empty_cells[0]]

    def set_position(self, x, y):
        self.x = x
        self.y = y
