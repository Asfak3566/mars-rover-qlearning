import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_q_table_heatmaps(q_table,
                               goal_state,
                               terrain,
                               crater_coordinates,
                               recharge_point):
    actions = ['Right', 'Left', 'Up', 'Down']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    grid_size = q_table.shape[0]
    x_ticks = np.arange(grid_size)
    y_ticks = np.arange(grid_size)

    for i, action in enumerate(actions):
        ax = axes[i]
        heatmap_data = q_table[:, :, i].copy()

        mask = np.zeros_like(heatmap_data, dtype=bool)
        mask[tuple(goal_state)] = True
        mask[tuple(recharge_point)] = True
        for crater in crater_coordinates:
            mask[tuple(crater)] = True
        for row in range(terrain.shape[0]):
            for col in range(terrain.shape[1]):
                if terrain[row, col] in [1, 2]: 
                    mask[row, col] = True

        sns.heatmap(
            heatmap_data, annot=True, fmt=".2f", cmap="viridis",
            ax=ax, cbar=False, mask=mask, annot_kws={"size": 6},
            xticklabels=x_ticks, yticklabels=y_ticks,linecolor='black', linewidths=0.5
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')

        ax.text(goal_state[1] + 0.5, goal_state[0] + 0.5, 'G', color='green',
                ha='center', va='center', weight='bold', fontsize=14)
        ax.text(recharge_point[1] + 0.5, recharge_point[0] + 0.5, 'R', color='blue',
                ha='center', va='center', weight='bold', fontsize=14)
        for crater in crater_coordinates:
            ax.text(crater[1] + 0.5, crater[0] + 0.5, 'X', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
        for row in range(terrain.shape[0]):
            for col in range(terrain.shape[1]):
                if terrain[row, col] == 1:
                    ax.text(col + 0.5, row + 0.5, 'S', color='orange',
                            ha='center', va='center', weight='bold',fontsize=14)
                elif terrain[row, col] == 2:
                    ax.text(col + 0.5, row + 0.5, 'K', color='purple',
                            ha='center', va='center', weight='bold', fontsize=14)
                    
        ax.set_title(f'Action: {action}')
        ax.set_xticks(x_ticks + 0.5)
        ax.set_yticks(y_ticks + 0.5)
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)
        plt.suptitle("Q-Table Heatmaps for Mars Rover Env", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

    

    