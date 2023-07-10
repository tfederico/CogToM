from __future__ import annotations

from gymnasium.envs.registration import register

__version__ = '0.0.1'


def register_custom_envs():
    register(
        id="",
        entry_point="cogtom.envs:CustomMiniGridEnv",
        kwargs={}
    )
