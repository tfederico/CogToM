from cogtom.rl.random_policy import RandomPolicy
from cogtom.ibl.ibl_observer import IBLObserver
from gymnasium import Env
import gymnasium as gym
import numpy as np
from pprint import pp
from matplotlib import pyplot as plt


MAX_STEPS = 30
SIZE = 11
N_AGENTS = 100


def train_one_episode(env: Env, agent: RandomPolicy) -> tuple[list, int, list]:
    done = False
    trajectory = []
    rewards = []
    # play one episode
    while not done:
        pos = env.agent_pos
        action = agent.get_action(pos)
        next_obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos

        # update the agent
        agent.update(pos, new_pos, action, reward, terminated)

        trajectory.append((pos, action))  # 4 means bumping into the wall
        rewards.append(reward)

        # update if the environment is done and the current obs
        done = terminated or truncated

    trajectory.append((new_pos, None))

    return trajectory, info["consumed_goal"], rewards


def train_agent(env: Env, agent: RandomPolicy, n_past: int) -> tuple[Env, list[list], list[int], list[list]]:

    trajectories = []
    consumed_goals = []
    rewards = []
    for _ in range(n_past):
        env.reset()

        trajectory, consumed_goal, reward = train_one_episode(env, agent)

        consumed_goals.append(consumed_goal if consumed_goal is not None else -1)  # -1 means no goal consumed, important!
        trajectories.append(trajectory)
        rewards.append(reward)

    return env, trajectories, consumed_goals, rewards


def test_ibl_observer():

    all_res = {}

    for n_past in [0, 1, 5]:

        for alpha in [0.01,  0.03, 0.1, 1.0, 3.0]:

            results = []

            for a in range(N_AGENTS):

                goal_index = np.random.randint(0, 4)
                goals_values = [0] * 4
                goals_values[goal_index] = 1
                goal_map = {"blue": {"reward": goals_values[0]},
                            "green": {"reward": goals_values[1]},
                            "yellow": {"reward": goals_values[2]},
                            "purple": {"reward": goals_values[3]}}
                env = gym.make("SimpleEnv-v1", render_mode="rgb_array", max_steps=MAX_STEPS, size=SIZE,
                               goal_map=goal_map)
                env.reset(hard_reset=True)
                agent = RandomPolicy(alpha)
                observer = IBLObserver(env, default_utility=-5)

                env, trajectories, goals_consumed, rewards = train_agent(env, agent, n_past)

                for trajectory, reward in zip(trajectories, rewards):
                    for (state, action), r in zip(trajectory, reward):
                        observer.populate(choices=[{"action": action, "state_x": state[0], "state_y": state[1]}], outcome=r)


                init_pos = env.agent_pos
                action = agent.get_action(init_pos)
                predicted_action = observer.get_action(init_pos)

                results.append(action == predicted_action)
                observer.reset()

            print(f"alpha: {alpha}, n_past: {n_past}, accuracy: {np.mean(results)} +/- "
                  f"{np.std(results)/np.sqrt(len(results))}")
            all_res[(alpha, n_past)] = (np.mean(results), np.std(results)/np.sqrt(len(results)))

    # plot using pyplot.fill_between
    fig, ax = plt.subplots()
    alphas = [0.01, 0.03, 0.1, 1.0, 3.0]

    for n_past in [0, 1, 5]:
        ax.plot(alphas, [all_res[(alpha, n_past)][0] for alpha in alphas], label=f"{n_past}")
        ax.fill_between(alphas, [all_res[(alpha, n_past)][0] - all_res[(alpha, n_past)][1] for alpha in alphas],
                        [all_res[(alpha, n_past)][0] + all_res[(alpha, n_past)][1] for alpha in alphas], alpha=0.3,
                        label='_nolegend_')

    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.03, 0.1, 1.0, 3.0])
    ax.set_xticklabels(["0.01", "0.03", "0.1", "1.0", "3.0"])
    ax.legend(title="n_past")
    plt.ylim([0, 1])
    plt.show()


if __name__ == "__main__":
    test_ibl_observer()