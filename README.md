# Atari Breakout RL Agent

This project implements a Reinforcement Learning (RL) agent using **Deep Q-Learning** to play Atari's **Breakout** game. The project includes two Python scripts:

- **train.py**: Trains the agent using Keras-RL and saves the trained policy to a file (`policy.h5`).
- **play.py**: Loads the trained model and runs the agent on the game, displaying its performance.

## Prerequisites

Before running the scripts, you need to set up the environment. This project requires the following Python packages:

- gymnasium
- ale-py
- keras
- keras-rl
- numpy
- matplotlib

You can install these dependencies using `pip`:

```bash
pip install gymnasium ale-py keras keras-rl numpy matplotlib
```

## Task 1: Training Script (train.py)

The train.py script sets up the Breakout environment and trains the RL agent using Deep Q-Learning (DQN). It uses keras-rl to define the DQN agent and the SequentialMemory class for experience replay. The exploration strategy is managed using the EpsGreedyQPolicy. The trained model is saved as policy.h5.

### Steps:

1. Set up the Breakout environment using gymnasium and ale-py.
2. Define the DQN agent using keras-rl, along with:
    - SequentialMemory for experience replay.
    - EpsGreedyQPolicy for exploration.
3. Train the agent for a sufficient number of steps (e.g., 50,000 steps).
4. Save the trained policy as policy.h5.

```bash
python train.py
```

### Example output:

```chsarp
Training agent...
Episode 1 finished with total reward: 10
Episode 2 finished with total reward: 12
...
Model saved as policy.h5
```

## Task 2: Playing Script (play.py)

The play.py script loads the trained model (policy.h5) and uses the GreedyQPolicy to play the Breakout game. It displays the game and shows how well the agent performs after training.

### Steps:

1. Load the trained model from the policy.h5 file.
2. Set up the Breakout environment using gymnasium and ale-py.
3. Run a few episodes and display the agent’s performance in real-time.

```bash
python play.py
```

### Example output:

```chsarp
Episode 1 finished with total reward: 50
Episode 2 finished with total reward: 60
...
```

## Additional Notes 

- The project uses gymnasium and ale-py to interact with the Breakout environment, which is an Atari 2600 game.
- The agent is trained using Deep Q-Learning (DQN), a model-free RL algorithm.
- The policy.h5 file contains the trained neural network model which is used in the play.py script to evaluate the agent's performance.

## Conclusion

This project demonstrates how to train a reinforcement learning agent to play the Breakout game using deep Q-learning. By using keras-rl and gymnasium, the project implements an efficient way to handle training and evaluation of the RL agent with minimal setup.