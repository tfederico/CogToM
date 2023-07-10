from __future__ import annotations

from gymnasium import Env
from minigrid.manual_control import ManualControl
from cogtom.core.actions import Actions


class CustomManualControl(ManualControl):
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        super().__init__(env, seed)

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.up,
            "down": Actions.down,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)