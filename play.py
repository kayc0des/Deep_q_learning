import gym
import ale_py
from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt

# Create the Breakout environment
env = gym.make('ALE/Breakout-v5')

# Load the trained policy model
model = load_model('policy.h5')

# Define the memory buffer (it should match the one used during training)
memory = SequentialMemory(limit=1000000, window_length=4)

# Define the Greedy policy
policy = GreedyQPolicy()

# Define the DQN agent
agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, policy=policy)

# Load the agent's weights (if needed)
agent.load_weights('policy.h5')

# Play the game using the trained agent
num_episodes = 5
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.forward(state)  # Get the action from the agent's policy
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        # Display the game frame
        plt.imshow(state)
        plt.axis('off')
        plt.pause(0.01)
    
    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()
plt.show()
