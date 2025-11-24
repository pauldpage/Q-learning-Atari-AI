import os
import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import cv2

# Configure This directory for the ALE roms or the script won't find it 
os.environ["ALE_ROM_DIR"] = "/Users/directory/.atari_roms"

# Hyperparameters 
learning_rate = 0.00025
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.997
epsilon_min = 0.1
episodes = 500
max_steps = 5000
frame_skip = 4
train_every = 4
batch_size = 64

# Replay buffer
replay_memory = deque(maxlen=50000)

# --- Create Atari environment ---
env = gym.make("ALE/MsPacman-v5", render_mode="human")
num_actions = env.action_space.n

# --- Preprocess raw Atari frames ---
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0


# --- Build CNN Q-network ---
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

model = build_model()


# --- Train from replay memory ---
def replay_train():
    if len(replay_memory) < batch_size:
        return

    minibatch = random.sample(replay_memory, batch_size)

    states = np.array([m[0] for m in minibatch])
    next_states = np.array([m[3] for m in minibatch])

    if len(states.shape) == 3:  # (64, 84, 84)
        states = np.expand_dims(states, -1)
    if len(next_states.shape) == 3:
        next_states = np.expand_dims(next_states, -1)

    q_values = model.predict(states, verbose=0)
    q_next = model.predict(next_states, verbose=0)

    # Build target values
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        target = reward if done else reward + gamma * np.max(q_next[i])
        q_values[i][action] = target

    model.fit(states, q_values, verbose=0)


# --- Main Training Loop ---
for ep in range(episodes):
    obs, _ = env.reset()

    state = preprocess_frame(obs)
    state = np.expand_dims(state, axis=-1)

    total_reward = 0

    for step in range(max_steps):

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = model.predict(np.expand_dims(state, axis=0), verbose=0)
            action = np.argmax(q_vals)

        # Frame skipping loop
        reward_sum = 0
        for _ in range(frame_skip):
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break

        next_state = preprocess_frame(obs)
        next_state = np.expand_dims(next_state, axis=-1)

        # Store transition
        replay_memory.append((
            state,
            action,
            reward_sum,
            next_state,
            terminated or truncated
        ))

        state = next_state
        total_reward += reward_sum

        # Train periodically
        if step % train_every == 0:
            replay_train()

        # End of episode
        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(
        f"Episode {ep+1}/{episodes}  |  "
        f"Total Reward: {total_reward:.2f}  |  "
        f"Epsilon: {epsilon:.3f}"
    )

env.close()
