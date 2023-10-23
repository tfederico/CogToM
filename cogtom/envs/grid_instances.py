from cogtom.core.actions import Actions
from cogtom.core.custom_grid import CustomGrid
from minigrid.core.mission import MissionSpace
from cogtom.core.custom_world_object import Goal
from cogtom.envs.custom_env import CustomMiniGridEnv
from minigrid.core.constants import COLOR_NAMES
from gymnasium.core import ObsType
from typing import Any
import numpy as np


class SimpleEnv(CustomMiniGridEnv):
    def __init__(
            self,
            size=10,
            agent_view_size=3,
            max_steps: int | None = None,
            goal_map=None,
            **kwargs,
    ):
        self.agent_pos = (-1, -1)
        mission_space = MissionSpace(mission_func=self._gen_mission)
        assert all([g in COLOR_NAMES for g in goal_map.keys()])
        self.goal_map = goal_map
        self.goals = None
        self.walls = None

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "CogTom"

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = CustomGrid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.goals = {}
        self.walls = []

        # generate random walls
        num_blocks = np.random.randint(1, 4)
        for _ in range(num_blocks):
            # 50% chance of a vertical wall
            if np.random.uniform() > 0.5:
                x = np.random.randint(1, width - 2)
                y1 = np.random.randint(1, height - 2)
                y2 = np.random.randint(1, height - 2)
                self.grid.vert_wall(x, min(y1, y2), abs(y2 - y1) + 1)
            else:
                y = np.random.randint(1, height - 2)
                x1 = np.random.randint(1, width - 2)
                x2 = np.random.randint(1, width - 2)
                self.grid.horz_wall(min(x1, x2), y, abs(x2 - x1) + 1)

        # Place the goals
        for color in self.goal_map.keys():
            pos = self.place_obj(Goal(color))
            self.goals[color] = pos

        # Save walls positions
        for i in range(width):
            for j in range(height):
                if self.grid.get(i, j) is not None and self.grid.get(i, j).type == "wall":
                    self.walls.append((i, j))

        self.mission = self._gen_mission()
        # Note: agent placement is handled by the outer env


    def _reward(self, action: Actions) -> tuple[float, bool, dict]:
        reward = 0
        terminated = False
        info = {
            "consumed_goal": None,
        }

        fwd_pos = tuple(map(sum, zip(self.agent_pos, Actions.get_action_vector(action))))
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos
            reward = -0.01

        if fwd_cell is not None and fwd_cell.type == "wall":
            reward = -0.05

        if fwd_cell is not None and fwd_cell.type == "goal":
            reward = self.goal_map[fwd_cell.color]["reward"]
            info["consumed_goal"] = fwd_cell.color
            terminated = True

        return reward, terminated, info

    def get_goals(self) -> Any:
        return self.goals

    def get_walls(self) -> Any:
        return self.walls
