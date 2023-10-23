from typing import Tuple, List, Any

import gymnasium as gym
import numpy as np

from cogtom.rl.policy import Policy
from numpy import ndarray
from tqdm import tqdm
import json
from cogtom.utils.saving import save_q_table, save_gridworld, save_q_table_image
from cogtom.utils.plotting import plot_training_results, plot_q_table
from gymnasium import Env
from cogtom.ibl.ibl_observer import IBLObserver
import math


# hyperparameters
LR = 0.1
N_EPISODES = 500
EPSILON = 0.1
DISCOUNT = 0.95
MAX_STEPS = 30
SIZE = 11
N_AGENTS = 100
N_PAST = 1


def train_one_episode(env: Env, agent: Policy) -> tuple[list, int]:
    done = False
    trajectory = []
    # play one episode
    while not done:
        pos = env.agent_pos
        action = agent.get_action(env.action_space, pos)
        next_obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos

        # update the agent
        agent.update(pos, new_pos, action, reward, terminated)

        trajectory.append((pos, action if pos != new_pos else 4))  # 4 means bumping into the wall

        # update if the environment is done and the current obs
        done = terminated or truncated

    trajectory.append((next_obs, None))

    return trajectory, info["consumed_goal"]


def train_agent(env: Env, agent: Policy | IBLObserver) -> tuple[
    Env, Policy | IBLObserver, list[list], ndarray | Any, list[int]]:
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=N_EPISODES)
    trajectories = []
    consumed_goals = []
    for _ in range(N_EPISODES):
        env.reset()
        if isinstance(agent, Policy):
            trajectory, consumed_goal = train_one_episode(env, agent)
        else:
            consumed_goal, step_count, step_preferred_goal = agent.move()
            trajectory = agent.trajectory
        consumed_goals.append(consumed_goal if consumed_goal is not None else -1)  # -1 means no goal consumed, important!
        trajectories.append(trajectory)

    goal_map = env.goal_map

    if len(consumed_goals) > 0 and (consumed_goals.count(None) != len(consumed_goals)):
        rate = [consumed_goals.count(i) for i in goal_map.keys()]
        rate = np.array(rate)/sum(rate)
    else:
        rate = np.random.dirichlet(alpha=(3,)*len(goal_map))

    return env, agent, trajectories, rate, consumed_goals


def extract_trajectory(init_pos: tuple, env: Env, q_table: dict) -> tuple[list, int]:
    trajectory = []
    pos = init_pos
    env.reset()
    while True:
        action = int(np.random.choice(np.argwhere(q_table[pos] == np.amax(q_table[pos])).flatten()))
        next_obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos
        trajectory.append((pos, action if pos != new_pos else 4))  # 4 means bumping into the wall TODO check why important
        pos = new_pos
        if terminated or truncated:
            break
    trajectory.append((next_obs, None))
    return trajectory, info["consumed_goal"]


def train():

    results = {}
    results["goal_consumed"] = np.zeros(1)
    results["first_action"] = np.zeros(1)
    std_result = {}
    std_result["goal_consumed"] = np.zeros(1)
    std_result["first_action"] = np.zeros(1)
    detailed_results = {}
    detailed_results["goal_consumed"] = np.zeros(N_AGENTS)
    detailed_results["first_action"] = np.zeros(N_AGENTS)

    with open("goal_map.json", "r") as f:
        goal_map = json.load(f)

    past_env = gym.make("SimpleEnv-v1", render_mode="rgb_array", max_steps=MAX_STEPS, size=SIZE, goal_map=goal_map)
    current_env = gym.make("SimpleEnv-v1", render_mode="rgb_array", max_steps=MAX_STEPS, size=SIZE, goal_map=goal_map)

    policy = Policy(
        learning_rate=LR,
        epsilon=EPSILON,
        discount_factor=DISCOUNT,
    )

    num_goal = len(goal_map)

    goal_consumed_rate = np.zeros(num_goal)

    for a in tqdm(range(N_AGENTS)):

        past_env.reset(hard_reset=True)
        save_gridworld(f"images/agent/agent_{a}_world.png", past_env)

        ibl_observer = IBLObserver(past_env, 0.1, goal_map)

        policy.init_q_table(past_env.width, past_env.height, past_env.action_space.n)

        for _ in range(N_PAST):
            past_env, policy, past_trajectories, past_rate, past_goals_consumed = train_agent(past_env, policy)
            goal_consumed_rate += past_rate/N_PAST
        # plot_training_results(env, policy)

        past_q_table = policy.q_values.copy()

        outcome_observed = np.zeros(num_goal)
        g = np.argmax(goal_consumed_rate)
        outcome_observed[g] = 1
        # print("Outcome observed: ", outcome_observed)

        current_env.reset(hard_reset=True)
        policy.init_q_table(current_env.width, current_env.height, current_env.action_space.n)

        init_pos = current_env.agent_pos
        current_env, policy, current_trajectories, current_rate, current_goals_consumed = train_agent(current_env, policy)

        current_q_table = policy.q_values.copy()

        true_trajectory, true_goal_consumed = extract_trajectory(init_pos, current_env, current_q_table)
        past_first_action = past_trajectories[0][0][1]
        true_first_action = true_trajectory[0][1]

        current_env.reset()

        ibl_observer.add_world(current_env)
        ibl_observer.add_outcome(list(goal_map.values())[g])

        # run IBL
        current_env, ibl_observer, trajectories_ibl, rate_ibl, goal_consumed_ibl = train_agent(current_env, ibl_observer)

        first_action_predicted = [t[0][1] for t in trajectories_ibl]

        results["goal_consumed"] += sum([gl == true_goal_consumed for gl in goal_consumed_ibl])/(N_EPISODES * N_AGENTS)
        results["first_action"] += sum(np.array(first_action_predicted) == true_first_action) / (N_EPISODES * N_AGENTS)
        detailed_results["goal_consumed"][a] = sum([gl == true_goal_consumed for gl in goal_consumed_ibl]) / N_EPISODES
        detailed_results["first_action"][a] = sum(np.array(first_action_predicted) == true_first_action) / N_EPISODES

        predicted_trajectory = ibl_observer.trajectory
        predicted_actions = ibl_observer.action_history

        # utility.plot_grid(current_grid, predicted_trajectory, "IBLObserver prediction")

        goal_prob, _ = np.histogram([list(goal_map.keys()).index(g) if g != -1 else -1 for g in goal_consumed_ibl],
                                    bins=range(5), density=True)
        action_prob, _ = np.histogram(predicted_actions, bins=range(5), density=True) # 4 is number of actions
        # print("Object prob:", goal_prob)
        # print("Action prob:", action_prob)

        # enable manual control for testing
        # manual_control = CustomManualControl(env, seed=42)
        # manual_control.start()

        save_q_table_image(f"images/q_table/agent_{a}_q_table.png", policy.q_values, SIZE)



    print("Goal consumed rate: ", goal_consumed_rate)
    past_env.close()
    current_env.close()

    std_result["goal_consumed"] = np.array(np.std(detailed_results["goal_consumed"])/math.sqrt(N_AGENTS))
    std_result["first_action"] = np.array([np.std(detailed_results["first_action"])/math.sqrt(N_AGENTS)])
    print(results)
    print(std_result)




if __name__ == "__main__":
    train()