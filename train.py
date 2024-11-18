import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import pickle

class BreakoutEnvironment:
    def __init__(self):
        # Register ALE environments
        gym.register_envs(ale_py)
        # Create the Breakout environment
        self.env = gym.make('ALE/Breakout-v5')
        self.obs, self.info = self.env.reset()

    def run_random_actions(self, num_episodes):
        """Run random actions in the environment to display game frames."""
        for episode in range(num_episodes):
            self.obs, self.info = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()  # Take a random action
                self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                done = terminated or truncated

                # Display the game frame
                plt.imshow(self.obs)
                plt.axis('off')  # Hide axes for better visualization
                plt.pause(0.01)  # Pause to create a frame-by-frame effect

            print(f"Episode {episode + 1} finished")

        self.env.close()
        plt.show()  # Display the final frame


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        # Define Q-learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.min_epsilon = 0.01  # Minimum exploration rate

        # Initialize Q-table
        self.q_table = np.zeros((self.env.observation_space.shape[0], self.env.action_space.n))

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def train(self, num_episodes):
        """Train the agent using Q-learning."""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Update Q-value using the Q-learning update rule
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error

                state = next_state
                total_reward += reward

            # Decay epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode + 1} finished with total reward: {total_reward}")

        # Save the Q-table to a file
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)


if __name__ == "__main__":
    # Create the Breakout environment and agent
    breakout_env = BreakoutEnvironment()

    # Running random actions to display the game frames
    breakout_env.run_random_actions(num_episodes=5)

    # Initialize QLearningAgent and start training
    q_learning_agent = QLearningAgent(breakout_env.env)
    q_learning_agent.train(num_episodes=15)
