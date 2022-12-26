# Optimal-Horizon-Length-Prediction-in-Model-Predictive-Control
A group project for the course Control and Optimization with Applications in Robotics at the University of Pennsylvania.

- [Project Report PDF](https://github.com/Zador-Pataki/Optimal-Horizon-Length-Prediction-in-Model-Predictive-Control/files/7730444/MEAM_517_Group_11_Final_Report.1.pdf)
- [Project Poster PDF](https://github.com/Zador-Pataki/Optimal-Horizon-Length-Prediction-in-Model-Predictive-Control/files/7730431/group_11_presentation.pdf)
- [Project Pitch PDF](https://github.com/Zador-Pataki/Optimal-Horizon-Length-Prediction-in-Model-Predictive-Control/files/7730434/517.Project.Pitch.pdf)

MPC controllers predict input sequences over finite time horizons of planned paths. If the time-horizon is too long, the computational time of the problem is increased, introducing latency into the system. On the ohter hand, predictions over shorter time-horizons may be suboptimal. In this project, we explore the ability of neural networks to select time-horizons based on planned trajectories, optimizing the trade-off between complexity and optimality. 

## Code
### Horizon_Predictor.ipynb
File containing Neural Network frameowrk, and is also responsible for training and gridsearch of model and optmizer parameters. The criterion used to guide the optimization of our model is a scalar sum of the horizon length at any given time step, and the trajectory error: distance between the planned trajectory and the resulting trajectory as a result of the planned sequence of inputs. 

### quad_sim.py
Given quadrotor model and trajectory, this function is used to simulate the MPC process of the quadrotor.

### quadrotor.py
This code is used to model the dynamics of the quadrotor.

### quadrotorTrajectory.ipynb
Extracted from one of the homeworks of the course, this notebook governs the the entire framework of the quadrotor. 

### quadrotorTrajectory_dataGeneration.ipynb
Extracted from one of the homeworks of the course, this notebook governs the the entire framework of the quadrotor. Additionally, the notebook was modified to be used as a data extractor. Using various trajectories, MPC is applied with different horizon lenghts, and states and observations are collected throughout the process.





