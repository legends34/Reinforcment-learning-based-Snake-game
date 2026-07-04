# Reinforcment-learning-based-Snake-game
# Snake Game Bot (Deep Q-Learning)

This is a Snake game bot that learns how to play itself using Reinforcement Learning. It starts out knowing absolutely nothing about the game and eventually figures out how to survive and eat the food without crashing into walls or its own tail.

## How it works

I built the game environment using `pygame` and the neural network with `TensorFlow/Keras`. The bot uses a Deep Q-Network (DQN) and updates its choices using the Bellman Equation, balancing random exploration with its trained model.The DQN part was made from scratch , TensorFlow was used for the Neural Networks only

### The State (What the snake sees)
It makes decisions based on a 12-boolean array:
* **Danger:** Is there a wall or tail straight ahead, to the left, or to the right?
* **Direction:** Which way is it currently moving (Up, Down, Left, Right)?
* **Food:** Where is the food relative to the head (Up, Down, Left, Right)?

### The Actions
Instead of moving North/South/East/West, the actions are relative to its current direction:
* Go straight `[1, 0, 0]`
* Turn right `[0, 1, 0]`
* Turn left `[0, 0, 1]`

### Rewards
To get the bot to actually learn and not just run in circles, the reward system is set up like this:
* **Eats food:** +15 * length
* **Dies (hits wall/self):** -20 * length
* **Moves closer to food:** +2
* **Moves away from food:** -2

## Tech Stack
* Python 3
* TensorFlow / Keras
* Pygame
* Matplotlib (for the live training graph)
* Pandas & NumPy


P.S. : This was made 5-6 months ago , IDK why I didn't commit