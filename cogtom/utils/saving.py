import json
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from cogtom.core.actions import Actions


def save_q_table(q_values, path):
    # convert q-table keys to string for json serialization
    q_values_str = {str(k): v.tolist() for k, v in q_values.items()}
    # save the q-table
    with open(path, "w") as f:
        json.dump(q_values_str, f)


def save_gridworld(path, env):
    # save the gridworld as an image
    img = env.render()
    img = Image.fromarray(img)
    img.save(path)


def save_q_table_image(path, q_table, size):

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
    plt.savefig(path)
    plt.close()