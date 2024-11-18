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

## Getting Started

To get started with this project, follow these steps:

### 1. Clone the Repository

Begin by cloning the project repository to your local machine:

```bash
git clone https://github.com/kayc0des/Deep_q_learning/
```

### 2. Navigate to the Project Directory

Once the repository is cloned, navigate to the project folder:

```bash
cd Deep_q_learning
```

### 3. Set Up the Virtual Environment

It's recommended to use a virtual environment to manage project dependencies. You can set up a virtual environment with the following commands:

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install the Dependencies

It's recommended to use a virtual environment to manage project dependencies. You can set up a virtual environment with the following commands:

```bash
pip install -r requirements.txt
```

### 5. Run the Project

Once all dependencies are installed, you can proceed with running the scripts.

- Train the Agent: To begin training the agent, run the following:

```bash
python train.py
```

This will start the training process on the Breakout environment. The agent will explore the environment, train using a deep Q-network, and save the trained policy to policy.h5.

- Play with the Trained Agent: After training, you can load the model and watch the agent play:

```bash
python play.py
```

This script loads the trained model from policy.h5 and uses the learned policy to play the Breakout game. The game will be displayed in real-time.

## Additional Notes 

- The project uses gymnasium and ale-py to interact with the Breakout environment, which is an Atari 2600 game.
- The agent is trained using Deep Q-Learning (DQN), a model-free RL algorithm.
- The policy.h5 file contains the trained neural network model which is used in the play.py script to evaluate the agent's performance.

## Conclusion

This project demonstrates how to train a reinforcement learning agent to play the Breakout game using deep Q-learning. By using keras-rl and gymnasium, the project implements an efficient way to handle training and evaluation of the RL agent with minimal setup.