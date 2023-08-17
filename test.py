import json
import gymnasium as gym
from cogtom.core.actions import Actions
import numpy as np
import time
from matplotlib import pyplot as plt
# reduce font size
plt.rcParams.update({'font.size': 12})


def test():

    with open("goal_map.json", "r") as f:
        goal_map = json.load(f)

    env = gym.make("SimpleEnv-v1", render_mode="human", size=11, goal_map=goal_map)
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

    # plot the q-table
    plt.imshow(np.swapaxes(np.array(list(q_table.values())).reshape(11, 11, 4), 1, 0).max(axis=2))
    # write the value of the best action in each state and the corresponding string using argmax
    for i, (k, v) in enumerate(q_table.items()):
        plt.text(
            i // 11,
            i % 11,
            f"{round(v[np.argmax(v)], 2)}: {Actions(np.argmax(v)).name}",
            ha="center",
            va="center",
            color="white",
        )
    # plot the legend
    plt.colorbar(ticks=range(4))

    plt.show()


if __name__ == "__main__":
    test()
