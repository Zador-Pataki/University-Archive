# Bayesian Neural Nets

The missing parts of the code were implemented accordingly. Special things to mention are:

Cost function: we use cross entropy loss in combination with an regularizer on the network weights

Dropout: Not used, as results got worse

Posterior initialisation: We initialize the posterior mu, rho parameters with uniform distribution. The uniform distribution for rho is negative in 
order to have small initial sigma.

MultivariateDiagonalGaussian sample function: We sample the weights using w = mu + softplus(rho)*sigma as described in the Blundell et al. paper.

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
