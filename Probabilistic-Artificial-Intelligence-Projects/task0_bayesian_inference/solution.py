# log_posterior_probs function was implemented in October 2021 by
# Pascal Lieberherr, Zador Pataki, Timo Schönegg

import numpy
from scipy.stats import laplace, norm, t
import scipy
import math
import numpy as np
from scipy.special import logsumexp

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])


def generate_sample(n_samples, seed=None):
    """ data generating process of the Bayesian model """
    random_state = np.random.RandomState(seed)
    hypothesis_idx = np.random.choice(3, p=PRIOR_PROBS)
    dist = HYPOTHESIS_SPACE[hypothesis_idx]
    return dist.rvs(n_samples, random_state=random_state)


""" Solution """

from scipy.special import logsumexp


def log_posterior_probs(x):
    """
    Computes the log posterior probabilities for the three hypotheses, given the data x

    Args:
        x (np.ndarray): one-dimensional numpy array containing the training data
    Returns:
        log_posterior_probs (np.ndarray): a numpy array of size 3, containing the Bayesian log-posterior probabilities
                                          corresponding to the three hypotheses
    """
    assert x.ndim == 1

    # p1 = sum(log p(xi | H1)) + log(p(H1)) - log(prod(p(x1|H1))*p(H1)+prod(p(x2|H2))*p(H2)+prod(p(x3|H3))*p(H3))
    # p1 ~ sum(log p(xi | H1)) + log(p(H1)) - LSE(log(prod(p(x1|H1))*p(H1)), prod(p(x2|H2))*p(H2), prod(p(x3|H3))*p(H3))   # https://en.wikipedia.org/wiki/LogSumExp
    # p1 ~ sum(log p(xi | H1)) + log(p(H1)) - LSE(sum(log(p(x1|H1))) + log(p(H1)), ...)

    log_p_h1 = np.log(PRIOR_PROBS[0])
    log_p_h2 = np.log(PRIOR_PROBS[1])
    log_p_h3 = np.log(PRIOR_PROBS[2])

    sum_log_p_x_given_h1 = 0
    sum_log_p_x_given_h2 = 0
    sum_log_p_x_given_h3 = 0

    for xi in np.nditer(x):
        sum_log_p_x_given_h1 += np.log(norm.pdf(xi, loc=0.0, scale=math.sqrt(VARIANCE)))
        sum_log_p_x_given_h2 += np.log(laplace.pdf(xi, loc=0.0, scale=laplace_scale))
        sum_log_p_x_given_h3 += np.log(t.pdf(xi, df=student_t_df))
    
    lse_123 = logsumexp(np.array([sum_log_p_x_given_h1 + log_p_h1, sum_log_p_x_given_h2 + log_p_h2, sum_log_p_x_given_h3 + log_p_h3]))

    log_p_1 = sum_log_p_x_given_h1 + log_p_h1 - lse_123
    log_p_2 = sum_log_p_x_given_h2 + log_p_h2 - lse_123
    log_p_3 = sum_log_p_x_given_h3 + log_p_h3 - lse_123

    log_p = np.array([log_p_1, log_p_2, log_p_3])
    
    assert log_p.shape == (3,)
    return log_p


def posterior_probs(x):
    return np.exp(log_posterior_probs(x))


""" """


def main():
    """ sample from Laplace dist """
    dist = HYPOTHESIS_SPACE[1]
    x = dist.rvs(1000, random_state=28)

    print("Posterior probs for 1 sample from Laplacian")
    p = posterior_probs(x[:1])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 100 samples from Laplacian")
    p = posterior_probs(x[:50])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 1000 samples from Laplacian")
    p = posterior_probs(x[:1000])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior for 100 samples from the Bayesian data generating process")
    x = generate_sample(n_samples=100)
    p = posterior_probs(x)

    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))


if __name__ == "__main__":
    main()
