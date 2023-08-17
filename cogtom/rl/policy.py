import gymnasium
import numpy as np
from gymnasium import Space, Env
from collections import defaultdict
from typing import Tuple
from cogtom.core.actions import Actions


class Policy:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        self.q_values = {}

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon

        self.training_error = []

    def init_q_table(self, env_width: int, env_height: int, action_space_n: int):
        for i in range(env_width):
            for j in range(env_height):
                self.q_values[(i, j)] = np.zeros(action_space_n)

        # initialize the q_values for the walls to -inf
        for i in range(env_width):
            self.q_values[(i, 0)][Actions.up] = -np.inf
            self.q_values[(i, env_height - 1)][Actions.down] = -np.inf
        for j in range(env_height):
            self.q_values[(0, j)][Actions.left] = -np.inf
            self.q_values[(env_width - 1, j)][Actions.right] = -np.inf

    def get_action(self, action_space: Space, current_pos: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """

        # with probability epsilon return a random action to explore the environment
        if np.random.uniform(0, 1) < self.epsilon:
            return action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            action_argmax = np.argwhere(self.q_values[current_pos] == np.amax(self.q_values[current_pos]))
            return int(np.random.choice(action_argmax.flatten()))

    def update(
        self,
        obs: Tuple[int, int],
        next_obs: Tuple[int, int],
        action: Actions,
        reward: float,
        terminated: bool,
    ):

        """Updates the Q-value of an action."""

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
