import numpy as np

def train_q_learning(env, no_episodes, 
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy",
                     render_during_training=False):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    for episode in range(no_episodes):
        agent_state, _= env.reset()
        agent_state = tuple(agent_state)
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[agent_state[0], agent_state[1]])

            next_state, reward, done, info, _ = env.step(action)

            if render_during_training:
                env.render()

            next_state = tuple(next_state)
            q_table[agent_state[0], agent_state[1], action] += alpha * (
                reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[agent_state[0], agent_state[1], action]
            )

            agent_state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()
    np.save(q_table_save_path, q_table)
    print(f"Training completed. Q-table saved as {q_table_save_path}")
    return q_table
    