from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from cogtom.core.custom_grid import CustomGrid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from cogtom.custom_manual_control import CustomManualControl
from cogtom.custom_env import CustomMiniGridEnv


class SimpleEnv(CustomMiniGridEnv):
    def __init__(
            self,
            size=10,
            agent_start_pos=(1, 1),
            agent_view_size=5,
            max_steps: int | None = None,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

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

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
        else:
            self.place_agent()

        self.mission = self._gen_mission()


def main():
    env = SimpleEnv(render_mode="human")
    env.reset()
    print(env)

    # enable manual control for testing
    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()