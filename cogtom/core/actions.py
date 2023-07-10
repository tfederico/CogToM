from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    # Move left, move right, move up, move down
    left = 0
    right = 1
    up = 2
    down = 3

    # Done completing task
    done = 4

    # Define method to get the corresponding action vector
    @staticmethod
    def get_action_vector(action: Actions) -> tuple[int, int]:
        if action == Actions.left:
            return -1, 0
        elif action == Actions.right:
            return 1, 0
        elif action == Actions.up:
            return 0, -1
        elif action == Actions.down:
            return 0, 1
        else:
            return 0, 0
