from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    # Move left, move right, move up, move down
    left = 0
    right = 1
    up = 2
    down = 3


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
            raise ValueError(f"Invalid action {action}")

    @staticmethod
    def get_actions() -> list[Actions]:
        return list(Actions)

    @staticmethod
    def get_action_name(action: Actions) -> str:
        if action == Actions.left:
            return "left"
        elif action == Actions.right:
            return "right"
        elif action == Actions.up:
            return "up"
        elif action == Actions.down:
            return "down"
        else:
            raise ValueError(f"Invalid action {action}")

    @staticmethod
    def get_action_from_name(action_name: str) -> Actions:
        if action_name == "left":
            return Actions.left
        elif action_name == "right":
            return Actions.right
        elif action_name == "up":
            return Actions.up
        elif action_name == "down":
            return Actions.down
        else:
            raise ValueError(f"Invalid action name {action_name}")

    @staticmethod
    def get_actions_vector() -> list[tuple[int, int]]:
        return [Actions.get_action_vector(action) for action in Actions.get_actions()]

