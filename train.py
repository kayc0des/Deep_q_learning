import gym
import ale_py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

# Create the Breakout environment
env = gym.make('ALE/Breakout-v5')

# Define the DQN model
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=env.observation_space.shape))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))  # Q-values for each action

# Define the memory buffer
memory = SequentialMemory(limit=1000000, window_length=4)

# Define the exploration strategy (Epsilon-greedy)
policy = EpsGreedyQPolicy(epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

# Define the DQN agent
agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, policy=policy,
                 nb_steps_warmup=10000, target_model_update=1000, gamma=0.99, 
                 train_interval=4, delta_clip=1.)

# Compile the agent
agent.compile(Adam(lr=0.00025), metrics=['mae'])

# Train the agent
agent.fit(env, nb_steps=50000, visualize=True, verbose=2)

# Save the trained policy network
agent.save_weights('policy.h5', overwrite=True)
