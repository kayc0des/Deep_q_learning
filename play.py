import pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

class BreakoutEnvironment:
    def __init__(self):
        # Register ALE environments
        gym.register_envs(ale_py)
        # Create the Breakout environment
        self.env = gym.make('ALE/Breakout-v5')

    def reset(self):
        """Reset the environment and return initial state."""
        return self.env.reset()

    def step(self, action):
        """Take an action and return the results."""
        return self.env.step(action)

    def close(self):
        """Close the environment."""
        self.env.close()


class QLearningAgent:
    def __init__(self, q_table_path):
        # Load the Q-table from a file
        with open(q_table_path, 'rb') as f:
            self.q_table = pickle.load(f)

    def choose_action(self, state):
        """Choose the best action based on the Q-table."""
        return np.argmax(self.q_table[state])

    def play(self, env, num_episodes):
        """Play the game using the Q-table."""
        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                state = next_state
                total_reward += reward

                # Display the game frame
                plt.imshow(state)
                plt.axis('off')  # Hide axes for better visualization
                plt.pause(0.01)  # Pause to create a frame-by-frame effect

            print(f"Episode {episode + 1} finished with total reward: {total_reward}")


if __name__ == "__main__":
    # Initialize environment and agent
    breakout_env = BreakoutEnvironment()
    q_learning_agent = QLearningAgent('q_table.pkl')

    # Play using the loaded Q-table
    q_learning_agent.play(breakout_env, num_episodes=5)

    # Close the environment and display the final frame
    breakout_env.close()
    plt.show()
