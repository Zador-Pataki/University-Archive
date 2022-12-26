# Gaussian Progress Regression

To solve the problem, the Nystroem transformation is used with a kernel composed of an RBF kernel, WhiteKernel, Matern kernel and RationalQuadratic kernel. The model is than trained and finally predicts the mean and std of the Gaussian Process. Due to the special cost function, means slightly below 35.5 will be increased to 35.5. Means that are below 35.5 will be slightly reduced to avoid over-prediction.

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
