# mars-rover-qlearning
Q-learning environment for Mars Rover simulation using Gymnasium &amp; Pygame
=====================================================
 Mars Rover Q-Learning Environment
=====================================================

This project is a custom grid-based simulation of a Mars Rover
trained with Q-learning using Gymnasium and Pygame. The rover
learns to navigate an environment with different terrains,
avoid hazards, and reach recharge stations.

-----------------------------------------------------
 Environment Description
-----------------------------------------------------
The environment is represented as a 2D grid with the following elements:

- Water       : Positive reward (goal state)
- Craters     : Negative terminal state (episode restarts)
- Sand / Rock : Negative terrain (small penalties)
- Recharge    : Positive reward (encourages exploration)

-----------------------------------------------------
 Features
-----------------------------------------------------
- Gymnasium API compatible
- Visualization with Pygame
- Q-learning implementation with adjustable hyperparameters
- Training progress visualization with Matplotlib / Seaborn

-----------------------------------------------------
 Project Structure
-----------------------------------------------------
mars-rover-qlearning/
│-- env.py               : Custom Mars Rover environment
│-- qlearning_agent.py   : Q-learning agent implementation
│-- main.py              : Training script
│-- visualization.py     : Visualization of Q-table / results
│-- requirements.txt     : Dependencies
│-- README.txt           : Project documentation

-----------------------------------------------------
 Installation
-----------------------------------------------------
1. Clone the repository:
   git clone https://github.com/YOUR-USERNAME/mars-rover-qlearning.git
   cd mars-rover-qlearning

2. Install dependencies:
   pip install -r requirements.txt

-----------------------------------------------------
 Usage
-----------------------------------------------------
1. Train the agent:
   python main.py

2. Visualize results or Q-table:
   python visualization.py

3. Run the rover with a trained policy:
   python qlearning_agent.py --test

-----------------------------------------------------
 Hyperparameters
-----------------------------------------------------
The main Q-learning hyperparameters are in qlearning_agent.py:
- alpha    : Learning rate
- gamma    : Discount factor
- epsilon  : Exploration rate
- episodes : Number of training episodes

-----------------------------------------------------
 License
-----------------------------------------------------
This project is licensed under the MIT License.

-----------------------------------------------------
 Contributions
-----------------------------------------------------
Contributions are welcome! Please open an issue or submit a pull request.
