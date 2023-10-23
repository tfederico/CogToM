import numpy as np
from pyibl import Agent
from cogtom.core.actions import Actions
from typing import Tuple

class IBLObserver(Agent):
    def __init__(self, env, default_utility, goal_map):
        super(IBLObserver, self).__init__(name="My Agent", attributes=["action", "state_x", "state_y"], default_utility=default_utility)
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

    def move(self):
        self.last_goal_consumed = None
        self.trajectory = []
        self.action_history = []
        done = False
        # self.trajectory.append(self.env.agent_pos)
        preferred_goal = self.outcome_goals
        step_preferred_goal = self.env.max_steps

        delay = {}

        while not done:
            pos = self.env.agent_pos
            action = self.get_action(pos)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            new_pos = self.env.agent_pos
            self.last_action = action
            if pos == new_pos:
                self.last_action = 4
            if info["consumed_goal"]:
                done = True
                for g, g_pos in self.env.get_goals().items():
                    if g_pos == new_pos:
                        if g == preferred_goal:
                            step_preferred_goal += 1
                        self.last_goal_consumed = g
                        break
            delay[self.env.step_count-1] = self.respond()

            self.trajectory.append((pos, self.last_action))

            # self.x = new_pos[0]
            # self.y = new_pos[1]

            self.action_history.append(self.last_action)

            done = done or terminated or truncated

        self.trajectory.append((new_pos, None))

        if self.last_goal_consumed == preferred_goal:
            step_preferred_goal = self.env.step_count

        for i in range(len(self.action_history)):
            if self.action_history[i] == 4:
                delay[i].update(-0.05)
            else:
                if self.last_goal_consumed:
                    delay[i].update(self.goal_map[self.last_goal_consumed]["reward"])
                else:
                    delay[i].update(-0.01)

        return self.last_goal_consumed, self.env.step_count, step_preferred_goal

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

