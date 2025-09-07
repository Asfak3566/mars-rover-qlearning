from environment import MarsEnv2
from visualization import visualize_q_table_heatmaps
from qlearning import train_q_learning
import numpy as np

train = True
visualize_results = True

if __name__ == "__main__":
    env = MarsEnv2(grid_size=10)

    if train:
        q_table = train_q_learning(env=env,
                         no_episodes=2000,
                         epsilon=1.0,
                         epsilon_min=0.1,
                         epsilon_decay=0.999,
                         alpha=0.1,
                         gamma=0.99,
                         render_during_training=True
                         )
    else:
        q_table = np.load("q_table.npy")

    if visualize_results:
        visualize_q_table_heatmaps(
            q_table,
            goal_state=env.goal_state,
            terrain=env.terrain,
            crater_coordinates=env.craters.tolist(),
            recharge_point=env.recharge_point
        )