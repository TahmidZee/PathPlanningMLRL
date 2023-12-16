#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().system('pip install tqdm')


import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm

from IPython.display import HTML

class AdvancedRobotEnv(gym.Env):


    def __init__(self):
        super(AdvancedRobotEnv, self).__init__()
        self.grid_size = 10  # Define the grid size
        self.action_space = spaces.Discrete(4)  # Define the action space (up, down, left, right)
        self.num_static_obstacles = 3  # Define the number of static obstacles
        self.num_dynamic_obstacles = 2  # Define the number of dynamic obstacles
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(2 + 2 * self.num_static_obstacles + 2 * self.num_dynamic_obstacles,), dtype=np.int32)

        self.state = None  # Initialize the state
        self.goal_position = np.array([self.grid_size - 1, self.grid_size - 1])  # Set the goal position
        self.ani = None  # To keep the reference to the animation

        # Generate static and dynamic obstacles
        self.static_obstacles = [self._random_position() for _ in range(self.num_static_obstacles)]
        self.dynamic_obstacles = [self._random_position() for _ in range(self.num_dynamic_obstacles)]
        self.reset()

    def reset(self):
        """Reset the environment state."""
        self.state = np.array([0, 0])  # Starting position of the robot
        # Add positions of static and dynamic obstacles to the state
        full_state = np.concatenate((self.state, np.array(self.static_obstacles).flatten(), np.array(self.dynamic_obstacles).flatten()))
        return full_state

    def step(self, action):
        """Update the environment state based on an action."""
        # Move the robot
        self.state[:2] = self._move_robot(self.state[:2], action)

        # Update dynamic obstacles
        for i in range(self.num_dynamic_obstacles):
            self.dynamic_obstacles[i] = self._move_obstacle(self.dynamic_obstacles[i])

        # Check for collisions with static and dynamic obstacles
        collision_static = any(np.array_equal(self.state[:2], obstacle) for obstacle in self.static_obstacles)
        collision_dynamic = any(np.array_equal(self.state[:2], obstacle) for obstacle in self.dynamic_obstacles)
        collision = collision_static or collision_dynamic

        # Determine if the goal has been reached
        done = np.array_equal(self.state[:2], self.goal_position)
        reward = self._calculate_reward(done, collision)

        # Concatenate robot position with obstacle positions to form the full state
        full_state = np.concatenate((self.state, np.array(self.static_obstacles).flatten(), np.array(self.dynamic_obstacles).flatten()))
        return full_state, reward, done, {}

    def _random_position(self):
        """Generate a random position within the grid."""
        return np.array([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])

    def _move_robot(self, position, action):
        """Move the robot based on the action."""
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        new_position = position + moves[action]
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        return new_position.astype(int)

    def _move_obstacle(self, position):
        """Randomly move a dynamic obstacle to a new position."""
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # Add a stationary move
        move = random.choice(moves)
        new_position = position + move
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        return new_position.astype(int)

    def _check_collision(self):
        """Check if the robot has collided with any obstacle."""
        for obstacle in self.static_obstacles + self.dynamic_obstacles:
            if np.array_equal(self.state[:2], obstacle):
                return True
        return False

    

    def _calculate_reward(self, done, collision):
        if done:
        # High reward for reaching the goal
            return 100
        elif collision:
        # High penalty for collision
            return -10
        else:
            # Calculate normalized distance to the goal
            max_distance = np.sqrt(2 * (self.grid_size ** 2))  # Diagonal distance of the grid
            distance_to_goal = np.linalg.norm(self.state[:2] - self.goal_position)
            normalized_distance = distance_to_goal / max_distance

            # Calculate proximity reward (higher when closer to the goal)
            proximity_reward = (1 - normalized_distance) * 5  

            # Small penalty for each step to encourage efficiency
            return proximity_reward - 1


    def animate_step(self, i, fig, ax):
        """Animation step function."""
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        plt.grid()

        # Update robot and obstacles positions
        action = self.action_space.sample()  #Show environment demo
        self.step(action)

        # Draw the robot
        robot = patches.Circle((self.state[1] + 0.5, self.grid_size - self.state[0] - 0.5), 0.4, color='blue')
        ax.add_patch(robot)

        # Draw static obstacles
        for obstacle in self.static_obstacles:
            static = patches.Rectangle((obstacle[1] + 0.1, self.grid_size - obstacle[0] - 0.9), 0.8, 0.8, color='grey')
            ax.add_patch(static)

        # Draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            dynamic = patches.Rectangle((obstacle[1] + 0.1, self.grid_size - obstacle[0] - 0.9), 0.8, 0.8, color='orange')
            ax.add_patch(dynamic)

        # Draw the goal
        goal = patches.Rectangle((self.goal_position[1] + 0.1, self.grid_size - self.goal_position[0] - 0.9), 0.8, 0.8, color='green')
        ax.add_patch(goal)

    def start_animation(self, steps=50):
        """Start the environment animation."""
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.animate_step, frames=steps, fargs=(fig, ax), interval=200)
        return HTML(ani.to_jshtml())

# Testing the environment with animation
env = AdvancedRobotEnv()
state = env.reset()
print("State shape:", state.shape)
env.start_animation(steps=50)  # Start the animation


# In[3]:


import tensorflow as tf
from tensorflow.keras.optimizers import legacy
from tensorflow.keras import layers, models
import numpy as np
import random
from collections import deque

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.85  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep Q-learning Model
        model = models.Sequential()
        model.add(layers.Dense(128, input_shape=(self.state_size,), activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=legacy.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def save_checkpoint(agent, episode, checkpoint_dir='dqn_checkpoints', filename='dqn_checkpoint.h5'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    agent.model.save(os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.h5'))

# Function to load a checkpoint
def load_checkpoint(agent, filename):
    if not os.path.exists(filename):
        print(f"Checkpoint file {filename} not found.")
        return False
    try:
        agent.model.load_weights(filename)
        agent.update_target_model()
        print(f"Loaded checkpoint from {filename}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def find_latest_checkpoint(checkpoint_dir='dqn_checkpoints'):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if not checkpoint_files:
        return None
    latest_file = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_file)

    
env = AdvancedRobotEnv()
state_size = env.observation_space.shape[0]  
action_size = env.action_space.n
# Initialize DQN agent
agent = DQN(state_size, env.action_space.n)

latest_checkpoint = find_latest_checkpoint()
if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    if load_checkpoint(agent, latest_checkpoint):
        # Extract the episode number from the checkpoint filename
        start_episode = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
    else:
        start_episode = 0
else:
    print("No checkpoint found, starting from scratch")
    start_episode = 0

EPISODES = 250
BATCH_SIZE = 16

import time

episode_path_lengths = []
episode_computation_times = []
episode_scores = []

for e in tqdm(range(start_episode, EPISODES), desc='Training Progress'):
    start_time = time.time()
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time_step in range(250):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            agent.update_target_model()
            break

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    episode_duration = time.time() - start_time
    episode_path_lengths.append(time_step + 1)
    episode_computation_times.append(episode_duration)
    episode_scores.append(total_reward)
    print(f"Episode: {e+1}/{EPISODES}, Duration: {episode_duration:.2f}s, Steps: {time_step + 1}, Score: {total_reward}")

    if e % 25 == 0:
        save_checkpoint(agent, e)

# Plotting results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(episode_path_lengths)
plt.title('Episode Path Lengths')
plt.xlabel('Episode')
plt.ylabel('Path Length')
plt.subplot(1, 3, 2)
plt.plot(episode_computation_times)
plt.title('Episode Computation Times')
plt.xlabel('Episode')
plt.ylabel('Computation Time (s)')
plt.subplot(1, 3, 3)
plt.plot(episode_scores)
plt.title('Episode Scores')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




