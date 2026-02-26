import numpy as np

def from_probability_to_N(priors, limits):
    l = 1
    tau = np.array(limits[1:]) - np.array(limits[:-1])
    N = []
    for j in range(len(tau)):
        N.append(
            np.round(tau[j]/(-2*np.log(1 - (priors[j]/l))))
        )
        l *= np.exp(-tau[j]/(2*N[j]))
    return N


def from_N_to_probability(N, limits):
    tau = np.array(limits[1:]) - np.array(limits[:-1])
    l = 1
    p = []
    for j in range(len(tau)):
        p.append(
            l * (1 - np.e**(-tau[j]/(2*N[j])))
        )
        l *= np.exp(-tau[j]/(2*N[j]))
    return p + [1-sum(p)]
