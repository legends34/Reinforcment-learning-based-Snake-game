"""
Snake Game Bot using Deep Q-Learning (Reinforcement Learning)

Reward System:
    * +15 * length: Eating food
    * -20 * length: Game over (collision)
    * +2: Moving towards food
    * -2: Moving away from food

Action Space:
    * [1, 0, 0]: Move Straight
    * [0, 1, 0]: Turn Right
    * [0, 0, 1]: Turn Left

State Space (12 Features - 0 for False, 1 for True):
    * Danger: [Up, Down, Left, Right]
    * Current Direction: [Up, Down, Left, Right]
    * Food Location: [Up, Down, Left, Right]

Neural Network Architecture:
    * Input Layer: 12 features (State space)
    * Hidden Layers: 64 nodes (ReLU) -> 32 nodes (ReLU)
    * Output Layer: 3 nodes (Linear) for Action space Q-values
"""

import random
import time
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

# Initialize Pygame
pygame.init()

# Define Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

# Screen Dimensions & Game Settings
WIDTH = 600
HEIGHT = 400
BLOCK_SIZE = 10
SPEED = 150

# Set up the display
dis = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake Game Bot - Deep Q-Learning')
clock = pygame.time.Clock()

# Fonts
score_font = pygame.font.SysFont("comicsansms", 35)


def plot(scores, mean_scores):
    """Plots the training progress dynamically."""
    plt.clf() 
    plt.title('Training progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def Your_score(score):
    """Displays the current score on the Pygame screen."""
    value = score_font.render("Score: " + str(score), True, YELLOW)
    dis.blit(value, [0, 0])


def our_snake(block_size, snake_list):
    """Draws the snake on the Pygame screen."""
    for x in snake_list:
        pygame.draw.rect(dis, GREEN, [x[0], x[1], block_size, block_size])


def message(msg, color):
    """Displays a message in the center of the screen."""
    font_style = pygame.font.SysFont("bahnschrift", 25)
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [WIDTH / 6, HEIGHT / 3])


model = Sequential([
    Dense(units=64, activation='relu', input_shape=(12,)),
    Dense(units=32, activation='relu'),
    Dense(units=3, activation='linear')
])

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-3))

# Global variables for training data and metrics
X = []
Y = []
all_scores = []
all_mean_scores = []
total_score = 0
record = 0
prev_direction = [0, 0, 0, 1]
count = 0


def game(action, x1, y1, snake_List, Length_of_snake, foodx, foody, game_close, game_over, prev_direction=[0, 0, 0, 1]):
    """
    Executes a single step in the game environment based on the AI's action.
    Returns the new state, reward, updated coordinates, and game status.
    """
    x1_change = 0
    y1_change = 0

    # Interpret the action array [straight, left, right] based on previous direction [up, down, left, right]
    if action[0]: 
        K_UP, K_DOWN, K_LEFT, K_RIGHT = prev_direction[0], prev_direction[1], prev_direction[2], prev_direction[3]
    elif action[1]:
        K_UP, K_DOWN, K_LEFT, K_RIGHT = prev_direction[3], prev_direction[2], prev_direction[0], prev_direction[1]
    elif action[2]:
        K_UP, K_DOWN, K_LEFT, K_RIGHT = prev_direction[2], prev_direction[3], prev_direction[1], prev_direction[0]

    reward = 0
    state = np.zeros(12)

    # Apply movement and update direction state
    if K_LEFT:
        x1_change = -BLOCK_SIZE
        y1_change = 0
        state[6] += 1
    elif K_RIGHT:
        x1_change = BLOCK_SIZE
        y1_change = 0
        state[7] += 1
    elif K_UP:
        y1_change = -BLOCK_SIZE
        x1_change = 0
        state[4] += 1
    elif K_DOWN:
        y1_change = BLOCK_SIZE
        x1_change = 0
        state[5] += 1

    prev_direction = [state[4], state[5], state[6], state[7]] 

    # Calculate distance BEFORE move
    dist_old = ((x1 - foodx)**2 + (y1 - foody)**2)**0.5

    x1 += x1_change
    y1 += y1_change
    dis.fill(BLACK)

    # Calculate distance AFTER move
    dist_new = ((x1 - foodx)**2 + (y1 - foody)**2)**0.5

    # Reward for moving closer to the food to speed up training
    if dist_new < dist_old:
        reward += 2  
    else:
        reward -= 2 

    # Check boundary collision
    if x1 >= WIDTH or x1 < 0 or y1 >= HEIGHT or y1 < 0:
        game_close = True
        reward = -20 * Length_of_snake
    
    pygame.draw.rect(dis, RED, [foodx, foody, BLOCK_SIZE, BLOCK_SIZE])
    
    # State: Find food location relative to snake head
    if foody > y1: state[8] += 1
    elif foody < y1: state[9] += 1

    if foodx > x1: state[11] += 1
    elif foodx < x1: state[10] += 1

    # State: Detect danger (boundaries or self)
    if x1 + BLOCK_SIZE >= WIDTH or [x1 + BLOCK_SIZE, y1] in snake_List: state[3] += 1
    if x1 - BLOCK_SIZE < 0 or [x1 - BLOCK_SIZE, y1] in snake_List: state[2] += 1
    if y1 + BLOCK_SIZE >= HEIGHT or [x1, y1 + BLOCK_SIZE] in snake_List: state[1] += 1
    if y1 - BLOCK_SIZE < 0 or [x1, y1 - BLOCK_SIZE] in snake_List: state[0] += 1

    # Update snake body
    snake_Head = [x1, y1]
    snake_List.append(snake_Head)
    
    if len(snake_List) > Length_of_snake:
        del snake_List[0]

    # Check self-collision
    for x in snake_List[:-1]:
        if x == snake_Head:
            game_close = True
            reward = -20 * Length_of_snake

    our_snake(BLOCK_SIZE, snake_List)
    Your_score(Length_of_snake - 1)
    pygame.display.update()

    # Check if food is eaten
    if x1 == foodx and y1 == foody:
        foodx = round(random.randrange(0, WIDTH - BLOCK_SIZE) / 10.0) * 10.0
        foody = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / 10.0) * 10.0
        Length_of_snake += 1
        reward = 15 * Length_of_snake

    clock.tick(SPEED)

    return state, reward, x1, y1, snake_List, Length_of_snake, foodx, foody, game_close, game_over, prev_direction

def gameLoop():
    global record, total_score, all_scores, all_mean_scores, prev_direction, count
    
    game_over = False
    game_close = False

    x1 = WIDTH / 2
    y1 = HEIGHT / 2

    snake_List = []
    Length_of_snake = 1

    foodx = round(random.randrange(0, WIDTH - BLOCK_SIZE) / 10.0) * 10.0
    foody = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / 10.0) * 10.0

    # Initialize starting state
    s0 = [0] * 12
    if foody > y1: s0[8] += 1
    elif foody < y1: s0[9] += 1
    if foodx > x1: s0[11] += 1
    elif foodx < x1: s0[10] += 1
    
    # Reinforcement Learning hyperparameters
    gamma = 0.9
    epsilon = 1
    steps = 0
    total_reward = 0
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
                
        if game_over:
            break

        # Determine current state
        if len(X) == 0:
            s = s0
        else:
            s = X[-1]

        # Epsilon-Greedy Action Selection (Explore vs Exploit)
        rand_prob = random.uniform(0, 1)
        action = [0, 0, 0]
        
        if rand_prob < epsilon:
            action_index = random.randint(0, 2) 
            action[action_index] += 1
        else:
            prediction = model.predict(np.array([s]), verbose=0)
            action_index = prediction.argmax(axis=1)[0]
            action[action_index] += 1

        # Execute action in the environment
        s_n, r_s, x1, y1, snake_List, Length_of_snake, foodx, foody, game_close, game_over, prev_direction = game(
            action, x1, y1, snake_List, Length_of_snake, foodx, foody, game_close, game_over, prev_direction
        )

        # Decay exploration rate
        epsilon = epsilon * 0.995
        count += 1
        total_reward += r_s
        steps += 1

        # Bellman Equation / Update Q-Values
        X.append(s_n)
        y = model.predict(np.array([s]), verbose=0)[0]
        
        # NewQ(s,a) = R(s,a) + gamma * max(Q(s',a'))
        y[action_index] = r_s + gamma * np.max(model.predict(np.array([s_n]), verbose=0))
        Y.append(y)

        # Experience Replay / Batch Training
        indices = random.sample(range(len(X)), min(32, len(X)))
        X_batch = [np.array(X)[i] for i in indices] 
        Y_batch = [np.array(Y)[i] for i in indices]
        
        if count % 10 == 0:  
            model.fit(np.array(X_batch), np.array(Y_batch), verbose=0)
            
            # Maintain memory limit
            if len(X) > 600:
                X.pop(0)
                Y.pop(0)

        # Handle Game Over condition
        if game_close:
            score = Length_of_snake - 1
            all_scores.append(score)
            total_score += score
            mean_score = total_score / len(all_scores)
            all_mean_scores.append(mean_score)
            
            plot(all_scores, all_mean_scores)

            # Reset environment for next game
            x1 = WIDTH / 2
            y1 = HEIGHT / 2
            foodx = round(random.randrange(0, WIDTH - BLOCK_SIZE) / 10.0) * 10.0
            foody = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / 10.0) * 10.0

            s0 = [0] * 12
            if foody > y1: s0[8] += 1
            elif foody < y1: s0[9] += 1
            if foodx > x1: s0[11] += 1
            elif foodx < x1: s0[10] += 1
    
            snake_List = []
            Length_of_snake = 1

            print(f'Game: {len(all_scores)}, Score: {score}, Record: {record}, Mean: {mean_score:.2f}, Reward: {total_reward}')
            
            if score > record:
                record = score
                # model.save('best_snake_model.h5') # Uncomment to save model locally
                
            game_close = False
            total_reward = 0

    pygame.quit()

if __name__ == "__main__":
    gameLoop()
    
    # Save training logs upon exit
    try:
        combined = pd.DataFrame({'X': X, 'Y': Y})
        combined.to_csv('Reinforced_Learning/game_logs.csv', index=False)
        print('Game logs saved successfully.')
    except Exception as e:
        print(f"Could not save logs: {e}")