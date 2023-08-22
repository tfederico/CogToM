from cogtom.core.actions import Actions
from cogtom.core.custom_grid import CustomGrid
from minigrid.core.mission import MissionSpace
from cogtom.core.custom_world_object import Goal
from cogtom.envs.custom_env import CustomMiniGridEnv
from minigrid.core.constants import COLOR_NAMES
from gymnasium.core import ObsType
from typing import Any


class SimpleEnv(CustomMiniGridEnv):
    def __init__(
            self,
            size=10,
            agent_start_pos=(1, 1),
            agent_view_size=3,
            max_steps: int | None = None,
            goal_map=None,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        mission_space = MissionSpace(mission_func=self._gen_mission)
        assert all([g in COLOR_NAMES for g in goal_map.keys()])
        self.goal_map = goal_map

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

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = CustomGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # TODO generate random walls


        # Place the goals
        for color in self.goal_map.keys():
            # TODO replace position with random
            self.put_obj(
                Goal(color),
                self.goal_map[color]["pos"][0],
                self.goal_map[color]["pos"][1]
            )

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
        else:
            self.place_agent()

        self.mission = self._gen_mission()

    def _reward(self, action) -> tuple[float, bool, dict]:
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
