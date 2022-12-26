# Bayesian Optimization Task

Following the approaches from the Gelbart et. all paper, the missing functions were implemented.
As described in the task description, as objective and constraint model a Gaussian Process with RBF and ConstantKernel is used. Initially we recommend point (3,3) in the middle of the domain space, afterwards we use the given optimize_acquisation_function.
As acquisation function, we multiply the expected improvement with the probability, that this point satisfies our solution (c<=0).
Last but not least we take the minimum of our sampled points, that satisfies the condition c<=0-sigma_v, in order to select for sure a valid point.

## Run solution
Navigate to this directory, than run

```
bash runner.sh
```

## Main code

The code is implemented in solution.py.

## Autors
In the scope of the ETH lecture Probablistic Artificial Intelligence (HS 2021), the following group solved the task:

Pascal Lieberherr, Zador Pataki, Timo Schönegg