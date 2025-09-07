# Assignment 2: Q-Learning Agent for Custom Mars Rover Environment

## Description
This project implements a Q-learning agent that learns to navigate a custom Mars rover environment (Assignment 1) using reinforcement learning. The environment is built with OpenAI Gymnasium and visualized with Pygame. The Q-table is visualized using matplotlib and seaborn heatmaps.

## Files Included
- `main.py`                : Main script to train and/or visualize the Q-table
- `environment.py`         : MarsEnv2 custom environment (OpenAI Gymnasium compatible)
- `qlearning.py`           : Q-learning training logic
- `visualization.py`       : Q-table heatmap visualization
- `q_table.npy`            : (Generated) Trained Q-table
- All image files          : rover.png, goal.png, sand.png, rock.png, crater.png, recharge.png
- (Optional) requirements.txt : Python dependencies
- (Optional) README.txt    : This file

## How to Run
1. **Install dependencies:**
   ```
   pip install gymnasium numpy pygame matplotlib seaborn
   ```
2. **Ensure all image files are in the same directory as the scripts.**
3. **Train the agent:**
   - In `main.py`, set `train = True` and run:
     ```
     python main.py
     ```
   - This will train the agent and save the Q-table as `q_table.npy`.
4. **Visualize the Q-table:**
   - Set `train = False` and `visualize_results = True` in `main.py` and run:
     ```
     python main.py
     ```
   - This will load the Q-table and display the heatmaps.

## Notes
- The environment penalizes the agent for falling into craters, stepping on sand/rock, and rewards for reaching the goal or recharge point.
- The Q-table is saved as `q_table.npy` and can be visualized without retraining.
- You can adjust hyperparameters (episodes, learning rate, etc.) in `main.py`.
- If you run multiple experiments, save Q-tables and images with different names for clarity.

