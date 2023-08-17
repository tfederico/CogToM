import cogtom
import gymnasium as gym
from cogtom.custom_manual_control import CustomManualControl
from cogtom.rl.policy import Policy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json


def train():
    # hyperparameters
    learning_rate = 0.1
    n_episodes = 500
    epsilon = 0.2
    discount_factor = 0.95

    with open("goal_map.json", "r") as f:
        goal_map = json.load(f)

    env = gym.make("SimpleEnv-v1", render_mode="rgb_array", size=11, goal_map=goal_map)
    env.reset()

    p = Policy(
        learning_rate=learning_rate,
        epsilon=epsilon,
        discount_factor=discount_factor,
    )
    p.init_q_table(env.width, env.height, env.action_space.n)

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            pos = env.agent_pos
            action = p.get_action(env.action_space, pos)

            next_obs, reward, terminated, truncated, info = env.step(action)
            new_pos = env.agent_pos

            # update the agent
            p.update(pos, new_pos, action, reward, terminated)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs


    rolling_length = 50
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
            np.convolve(
                np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
            np.convolve(
                np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
            np.convolve(np.array(p.training_error), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()

    # # enable manual control for testing
    # manual_control = CustomManualControl(env, seed=42)
    # manual_control.start()

    # convert q-table keys to string for json serialization
    q_values_str = {str(k): v.tolist() for k, v in p.q_values.items()}

    # save the q-table
    with open("q_table.json", "w") as f:
        json.dump(q_values_str, f)


if __name__ == "__main__":
    train()