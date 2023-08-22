import matplotlib.pyplot as plt
import numpy as np
from cogtom.core.actions import Actions
# reduce font SIZE
plt.rcParams.update({'font.size': 12})


def plot_training_results(env, policy):
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
            np.convolve(np.array(policy.training_error), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()


def plot_q_table(q_table, size):
    # plot the q-table
    plt.imshow(np.swapaxes(np.array(list(q_table.values())).reshape(size - 2, size - 2, 4), 1, 0).max(axis=2))
    # write the value of the best action in each state and the corresponding string using argmax
    for i, (k, v) in enumerate(q_table.items()):
        plt.text(
            i // (size - 2),
            i % (size - 2),
            f"{round(v[np.argmax(v)], 2)} \n {Actions(np.argmax(v)).name}",
            ha="center",
            va="center",
            color="white",
        )
    # plot the legend
    plt.colorbar(ticks=range(-1, 1))

    plt.show()