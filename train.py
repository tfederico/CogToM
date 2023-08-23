import gymnasium as gym
import numpy as np

from cogtom.rl.policy import Policy
from tqdm import tqdm
import json
from cogtom.utils.saving import save_q_table, save_gridworld, save_q_table_image
from cogtom.utils.plotting import plot_training_results, plot_q_table

# hyperparameters
LR = 0.1
N_EPISODES = 500
EPSILON = 0.1
DISCOUNT = 0.95
MAX_STEPS = 30
SIZE = 11
N_AGENTS = 100
N_PAST = 1


def train_one_episode(env, policy):
    done = False
    trajectory = []
    # play one episode
    while not done:
        pos = env.agent_pos
        action = policy.get_action(env.action_space, pos)

        next_obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos

        # update the agent
        policy.update(pos, new_pos, action, reward, terminated)

        trajectory.append((pos, action if pos != new_pos else 4))  # 4 means bumping into the wall TODO check why important

        # update if the environment is done and the current obs
        done = terminated or truncated

    trajectory.append((next_obs, None))

    return trajectory, info["consumed_goal"]


def train_agent(env, policy):
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=N_EPISODES)
    trajectories = []
    consumed_goals = []
    for _ in range(N_EPISODES):
        env.reset()
        trajectory, consumed_goal = train_one_episode(env, policy)
        consumed_goals.append(consumed_goal)  # -1 means no goal consumed, important!
        trajectories.append(trajectory)

    if len(consumed_goals) > 0:
        rate = [consumed_goals.count(i) for i in env.goal_map.keys()]
        rate = np.array(rate)/sum(rate)
    else:
        rate = np.random.dirichlet(alpha=(3,)*len(env.goal_map))

    return env, policy, trajectories, rate


def extract_trajecotory(init_pos, env, q_table):
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

        policy.init_q_table(past_env.width, past_env.height, past_env.action_space.n)

        for _ in range(N_PAST):
            past_env, policy, past_trajectories, past_rate = train_agent(past_env, policy)
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
        current_env, policy, current_trajectories, current_rate = train_agent(current_env, policy)

        current_q_table = policy.q_values.copy()

        true_trajectory, true_goal_consumed = extract_trajecotory(init_pos, current_env, current_q_table)
        past_first_action = past_trajectories[0][0][1]
        true_first_action = true_trajectory[0][1]

        current_env.reset()

        # enable manual control for testing
        # manual_control = CustomManualControl(env, seed=42)
        # manual_control.start()

        save_q_table_image(f"images/q_table/agent_{a}_q_table.png", policy.q_values, SIZE)



    print("Goal consumed rate: ", goal_consumed_rate)
    past_env.close()


if __name__ == "__main__":
    train()