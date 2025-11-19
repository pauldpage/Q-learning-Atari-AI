import os
import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import cv2

# --- CHANGE DIRECTORY TO ROMS LOCATION OR IT WON'T WORK ---
os.environ["ALE_ROM_DIR"] = "/Users/Directory/.atari_roms"

# Hyperparameters
learning_rate = 0.00025
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.997
epsilon_min = 0.1
episodes = 500
max_steps = 5000
frame_skip = 4           # Skip frames = speed up gameplay
train_every = 4          # Train every 4 steps
batch_size = 64

replay_memory = deque(maxlen=50000)

# Create MsPacman environment with live window 
env = gym.make("ALE/MsPacman-v5", render_mode="human")
num_actions = env.action_space.n

# Preprocess frames 
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return normalized

# Build Q-network 
def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(84, 84, 1)),
        layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

model = build_model()

#  Training function
def replay_train():
    if len(replay_memory) < batch_size:
        return

    minibatch = random.sample(replay_memory, batch_size)
    states = np.array([m[0] for m in minibatch])
    next_states = np.array([m[3] for m in minibatch])


    if len(states.shape) == 3:
        states = np.expand_dims(states, axis=-1)
    if len(next_states.shape) == 3:
        next_states = np.expand_dims(next_states, axis=-1)

    q_values = model.predict(states, verbose=0)
    q_next_values = model.predict(next_states, verbose=0)

    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        target = reward if done else reward + gamma * np.max(q_next_values[i])
        q_values[i][action] = target

    model.fit(states, q_values, batch_size=batch_size, verbose=0)

# --- Training Loop ---
for ep in range(episodes):
    obs, _ = env.reset()
    state = preprocess_frame(obs)
    state = np.expand_dims(state, axis=-1)
    total_reward = 0

    for step in range(max_steps):
        # Epsilon action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = model.predict(np.expand_dims(state, axis=0), verbose=0)
            action = np.argmax(q_vals)

        reward_sum = 0
        for _ in range(frame_skip):  # Skip frames to speed up learning
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break

        next_state = preprocess_frame(obs)
        next_state = np.expand_dims(next_state, axis=-1)

        replay_memory.append((state, action, reward_sum, next_state, terminated or truncated))
        state = next_state
        total_reward += reward_sum

        # Train every few steps
        if step % train_every == 0:
            replay_train()

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep+1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

env.close()
