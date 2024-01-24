import numpy as np
import tensorflow as tf
import gym

# Utility function
 
def preprocess_state(state):
    if isinstance(state, tuple) and len(state) > 1:
        state = np.array(state[0], dtype=np.float32).reshape(1, -1)
    else:
        state = np.array(state, dtype=np.float32).reshape(1, -1)
    return state

# Deep Q-Network model

class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_actions)

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)

# Replay buffer

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.capacity:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.capacity] = []
        self.buffer.extend(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]

# Environment

env = gym.make('CartPole-v1')
num_actions = env.action_space.n

# DQN model and target network

model = DQN(num_actions)
target_model = DQN(num_actions)
target_model.set_weights(model.get_weights())

# Optimizer & loss function

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

# Replay buffer

replay_buffer = ReplayBuffer(capacity=10000)

# Training parameters

gamma = 0.99
epsilon = 0.1
batch_size = 64

# Training loop

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    episode_reward = 0

    while True:

        # Epsilon-greedy policy

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values)

        
        step_result = env.step(action)
        next_state, reward, done, _ = step_result[0], step_result[1], step_result[2], step_result[3]
        next_state = preprocess_state(next_state)
        episode_reward += reward

        replay_buffer.add([(state, action, reward, next_state, done)])
      
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
       
        target_q_values = target_model.predict(np.vstack(next_states))
        targets = np.array(rewards) + gamma * np.max(target_q_values, axis=1) * (1 - np.array(dones))

        with tf.GradientTape() as tape:
            
            q_values = model(np.vstack(states), training=True)

            selected_q_values = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), axis=1)

            loss = loss_fn(targets, selected_q_values)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())

        if done:
            break

        state = next_state

    print(f"Episode: {episode + 1}, Reward: {episode_reward}")

# Save

model.save_weights('dqn_cartpole.h5')