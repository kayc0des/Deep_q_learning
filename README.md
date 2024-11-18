# Training and Playing Breakout with Q-learning

This project demonstrates how to train and play the Breakout game using Q-learning. The train.py script trains an agent using Q-learning, while the play.py script allows you to play the game using the trained Q-table.

## Project Structure

- train.py   # Script to train the agent using Q-learning
- play.py    # Script to play the game using the trained Q-table
- q_table.pkl # Saved Q-table after training


## Requirements:

- Python 3.x
- `gymnasium` library (for the OpenAI Gym environment)
- `ale-py` library (for Atari Learning Environment)
- `numpy`
- `matplotlib`

You can install the necessary dependencies using:

```bash
pip install gymnasium ale-py numpy matplotlib
```

## Overview

- `train.py:`

    - This script trains a Q-learning agent on the Breakout environment from the Atari Learning Environment (ALE).
    - It initializes a Q-table and uses an epsilon-greedy policy for exploration and exploitation.
    - The Q-table is updated after each action taken by the agent using the Q-learning update rule.
    - After training, the Q-table is saved to a file (q_table.pkl).

- `play.py:`

    - This script allows you to load the saved Q-table from q_table.pkl and play the Breakout game.
    - The agent plays the game by choosing the action with the highest Q-value in each state.
    - It displays the game frames and prints the total reward at the end of each episode.

## How to Use

- Training the Agent:

```bash
python train.py
```

    - The agent will play the Breakout game for 15 episodes, updating the Q-table after each step.
    - Once training is complete, the Q-table will be saved to a file named q_table.pkl.

- Playing the Game:

```bash
python play.py
```

    - The agent will play the game for 5 episodes, choosing actions based on the Q-table learned during training.
    - It will display the game frames and print the total reward at the end of each episode.

## Screenshots

Below are some example outputs from running the play.py script, displaying the Breakout game frames.

