from cogtom.rl.policy import Policy
from cogtom.ibl.ibl_observer import IBLObserver
from gymnasium import Env
import gymnasium as gym
import numpy as np
from pprint import pp
from matplotlib import pyplot as plt


LR = 0.1
N_EPISODES = 500
EPSILON = 0.1
DISCOUNT = 1
MAX_STEPS = 30
SIZE = 11
N_AGENTS = 100


def train_one_episode(env: Env, agent: Policy) -> tuple[list, int, list]:
    done = False
    trajectory = []
    rewards = []
    # play one episode
    while not done:
        pos = env.agent_pos
        action = agent.get_action(env.action_space, pos)
        next_obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos

        # update the agent
        agent.update(pos, new_pos, action, reward, terminated)

        trajectory.append((pos, action))  # 4 means bumping into the wall
        rewards.append(reward)

        # update if the environment is done and the current obs
        done = terminated or truncated

    trajectory.append((new_pos, -9999))

    return trajectory, info["consumed_goal"], rewards


def train_agent(env: Env, agent: Policy) -> tuple[Env, list[list], list[int], list[list]]:

    trajectories = []
    consumed_goals = []
    rewards = []
    for _ in range(N_EPISODES):
        env.reset()

        trajectory, consumed_goal, reward = train_one_episode(env, agent)

        consumed_goals.append(consumed_goal if consumed_goal is not None else -1)  # -1 means no goal consumed, important!
        trajectories.append(trajectory)
        rewards.append(reward)

    return env, trajectories, consumed_goals, rewards


def compare_actions(expected: list, ibl: list) -> list:
    length = min(len(expected), len(ibl))

    exp = expected[:length]
    ibl_copy = ibl[:length]

    diff = [e[1] == i[1] for e, i in zip(exp, ibl_copy)]

    diff += [False] * abs(len(ibl) - len(expected))

    return diff


def test_ibl_observer():

    all_res = {}

    n_pasts = range(0, 1)

    for n_past in n_pasts:

        goal_results = []


        for a in range(N_AGENTS):

            goals_values = np.random.dirichlet([0.01]*4)

            goal_map = {"blue": {"reward": goals_values[0]},
                        "green": {"reward": goals_values[1]},
                        "yellow": {"reward": goals_values[2]},
                        "purple": {"reward": goals_values[3]}}
            env = gym.make("SimpleEnv-v1", render_mode="rgb_array", max_steps=MAX_STEPS, size=SIZE,
                           goal_map=goal_map)
            env.reset(hard_reset=True)
            agent = Policy(
                learning_rate=LR,
                epsilon=EPSILON,
                discount_factor=DISCOUNT,
            )
            agent.init_q_table(env.width, env.height, env.action_space.n)
            observer = IBLObserver(env, default_utility=-5)

            env, trajectories, goals_consumed, rewards = train_agent(env, agent)

            for (state, action), r in zip(trajectories[-1], rewards[-1]):
                observer.populate(choices=[{"action": action, "state_x": state[0], "state_y": state[1]}], outcome=r)

            env.reset(hard_reset=True)

            expected_trajectory, expected_goal, expected_reward = train_one_episode(env, agent)

            env.reset()
            ibl_trajectory = []
            ibl_goal = None

            init_pos = env.agent_pos

            for _ in range(MAX_STEPS):

                pos = env.agent_pos
                action = observer.get_action(pos)
                next_obs, reward, terminated, truncated, info = env.step(action)
                ibl_trajectory.append((pos, action))
                pos = env.agent_pos

                # check if the goal is consumed
                if info["consumed_goal"] is not None:
                    ibl_goal = info["consumed_goal"]
                    break

                observer.respond()

            comparison_traj = compare_actions(expected_trajectory[:-1], ibl_trajectory)
            all_res[a] = {
                "trajectory": sum(comparison_traj) / len(comparison_traj),
                "goal": int(expected_goal == ibl_goal)
            }

            observer.reset()

            # print(f"n_past: {n_past}, accuracy: {np.mean(results)} +/- "
            #       f"{np.std(results)/np.sqrt(len(results))}")
            # all_res[n_past] = (np.mean(results), np.std(results)/np.sqrt(len(results)))

    # print mean and std
    print(f"actions: {np.mean([r['trajectory'] for k, r in all_res.items()])} +/- "
          f"{np.std([r['trajectory'] for k, r in all_res.items()])/np.sqrt(len(all_res))}")
    print(f"goal: {np.mean([r['goal'] for k, r in all_res.items()])} +/- "
            f"{np.std([r['goal'] for k, r in all_res.items()])/np.sqrt(len(all_res))}")

    # # plot using pyplot.fill_between
    # fig, ax = plt.subplots()
    #
    # ax.plot(n_pasts, [r[0] for k, r in all_res.items()], marker='o', linestyle='--')
    # ax.fill_between(n_pasts, [r[0] - r[1] for k, r in all_res.items()], [r[0] + r[1] for k, r in all_res.items()],
    #                 alpha=0.2)
    #
    # ax.set_xlabel("n_past")
    # ax.set_ylabel("accuracy")
    # plt.ylim([0, 1])
    # plt.show()


if __name__ == "__main__":
    test_ibl_observer()