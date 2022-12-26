# Drone-Checkpoint Assignment through Graph Based Euclidean Embedding 

This repository contains the code used to run the experiments of the project "Drone-Checkpoint  Assignment  through  Graph  Based  EuclideanEmbeddings" by Zador Pataki and  CarlosCarlos A. Guija Deza, for the course Advanced topics in control.

- [Project Report PDF](https://github.com/Zador-Pataki/aticcode/files/7707152/ATIC_project.pdf)
 
## System description
The file "world.py" contains a class which stores all relevant functions and variables for constructing and manipulating the graphs of the framework at individual process increments.

## Controller
The file "controller.py" contains a class which stores all relevant funcitons and variables for solving the assignment problems at each increment, and for generating the resulting directions in which the drones have to head in the upcoming time intervals.

## Simulation framework
The file "simulation.py" contains a class which stores all the relevent functions and variables to simulate a given set up, leveraging the classes from the "controller.py" and "world.py".

## Core experiments
The file "run_experiments.py" contains a script for running the core experiments of this work. The results of these experiments were those presented in the results section of the paper.

## Appendix experiments
The file "appendix.py" contains a script for running the experiments of the appendix of this work. The results of these experiments were those presented in the appendix of the paper.

## Open source code leveraged
In our code, we utilized open source code from the libraries ScipPy [[1]](#1) and Scikit-learn [[2]](#2).

<a id="1">[1]</a> 
E. Jones, T. Oliphant, P. Petersonet al., “SciPy: Open source scientific tools for Python,” 2001–. [Online]. Available: http://www.scipy.org/

<a id="2">[2]</a> 
F. Pedregosa,  G.  Varoquaux,  A.  Gramfort,  V.  Michel,  B.  Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Van-derplas,  A.  Passos,  D.  Cournapeau,  M.  Brucher,  M.  Perrot,  andE.  Duchesnay,  “Scikit-learn:  Machine  learning  in  Python,” Journalof Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
