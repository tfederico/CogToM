import json
import gymnasium as gym
from cogtom.core.actions import Actions
import numpy as np
import time
from cogtom.utils.plotting import plot_q_table


MAX_STEPS = 30
SIZE = 11


def test():
    with open("goal_map.json", "r") as f:
        goal_map = json.load(f)

    env = gym.make("SimpleEnv-v1", render_mode="human", size=SIZE, max_steps=MAX_STEPS, goal_map=goal_map)
    env.reset()

    # load q-table
    with open("q_table.json", "r") as f:
        q_table = json.load(f)

    # change the q-table keys from str to tuple
    q_table = {eval(k): v for k, v in q_table.items()}

    done = False
    cumulated_reward = 0
    while not done:
        action = np.argmax(q_table[env.agent_pos])

        print("Best action: ", Actions(action).name)

        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulated_reward += reward
        done = terminated or truncated
        env.render()
        # time.sleep(1)

    print("Cumulated reward: ", cumulated_reward)

    plot_q_table(q_table, SIZE)


if __name__ == "__main__":
    test()
