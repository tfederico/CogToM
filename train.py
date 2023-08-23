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
    first_action = None
    # play one episode
    while not done:
        pos = env.agent_pos
        action = policy.get_action(env.action_space, pos)

        if first_action is None:
            first_action = action

        next_obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos

        # update the agent
        policy.update(pos, new_pos, action, reward, terminated)

        # update if the environment is done and the current obs
        done = terminated or truncated

    return first_action, info["consumed_goal"]


def train_agent(env, policy):
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=N_EPISODES)
    first_actions = []
    consumed_goals = []
    for _ in range(N_EPISODES):
        env.reset()
        first_action, consumed_goal = train_one_episode(env, policy)
        consumed_goals.append(consumed_goal if consumed_goal else -1)
        first_actions.append(first_action)

    if len(consumed_goals) > 0:
        rate = [consumed_goals.count(i) for i in env.goal_map.keys()]
        rate = np.array(rate)/sum(rate)
    else:
        rate = np.random.dirichlet(alpha=(3,)*len(env.goal_map))

    return env, policy, first_actions, rate


def train():
    with open("goal_map.json", "r") as f:
        goal_map = json.load(f)

    env = gym.make("SimpleEnv-v1", render_mode="rgb_array", max_steps=MAX_STEPS, size=SIZE, goal_map=goal_map)

    policy = Policy(
        learning_rate=LR,
        epsilon=EPSILON,
        discount_factor=DISCOUNT,
    )

    num_goal = len(goal_map)

    goal_consumed_rate = np.zeros(num_goal)

    for a in tqdm(range(N_AGENTS)):

        env.reset(hard_reset=True)
        save_gridworld(f"images/agent/agent_{a}_world.png", env)

        policy.init_q_table(env.width, env.height, env.action_space.n)
        for _ in range(N_PAST):
            env, policy, first_actions, rate = train_agent(env, policy)
            goal_consumed_rate += rate/N_PAST
        # plot_training_results(env, policy)

        outcome_observed = np.zeros(num_goal)
        g = np.argmax(goal_consumed_rate)
        outcome_observed[g] = 1

        # print("Outcome observed: ", outcome_observed)

        # enable manual control for testing
        # manual_control = CustomManualControl(env, seed=42)
        # manual_control.start()

        save_q_table(policy.q_values, "q_table.json")

        save_q_table_image(f"images/q_table/agent_{a}_q_table.png", policy.q_values, SIZE)

    print("Goal consumed rate: ", goal_consumed_rate)
    env.close()


if __name__ == "__main__":
    train()