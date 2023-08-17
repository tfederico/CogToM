from gymnasium.envs.registration import register

register(
    id="SimpleEnv-v1",
    entry_point="cogtom.envs.grid_instances:SimpleEnv",
    kwargs={}
)

