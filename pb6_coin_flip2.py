import numpy as np
import random
from scipy.stats import binom
from scipy.special import comb

def create_mixture_data(p,r,pi):
    m = 1000
    coin_prob = pi
    bias = np.array([p,r])
    coin_choice= np.random.binomial(n=1,p=coin_prob,size=m)
    mixed_data = []
    for i in range(m):
        coin = coin_choice[i]
        flip_sequence = np.random.binomial(n=1,p=bias[coin],size=10)
        mixed_data.append(flip_sequence)
    return mixed_data


def coin_em(qk,pik,rolls, iteration=5):
    # Iterate
    z_ik = np.zeros((len(rolls), len(qk)))  # 1000 x 2
    y_n = np.array([np.count_nonzero(roll) for roll in rolls])  # 1000 x 1
    print("initial values","q ",qk,"pi ",pik)

    for iter in range(iteration):
        """E step"""
        for q, pi, num in zip(qk, pik, range(len(z_ik[0]))):
            for i,roll in enumerate(rolls):
                px = coin_likelihood(roll, q)
                z_all = [pi_k * coin_likelihood(roll, q_k) for q_k, pi_k in zip(qk, pik)]
                deno = np.sum(z_all, axis=0)
                z_ik[i, num] = (pi * px) / deno
        """M STep"""
        qk = []
        pik= []
        for k in range(len(z_ik[0])):
            #update mu
            z_sum = np.sum(z_ik[:, k])  #1 x 1
            numer = np.dot(z_ik[:, k],y_n)
            _sum = np.sum(numer, axis=0)
            # q_k = (1 / z_sum * 10) * _sum  #1x1
            q_k =  _sum / (z_sum * len(rolls[0]))
            qk.append(q_k) # 1x2
            pik.append(z_sum / len(rolls))
        print("iter: ",iter,"q: ", np.round(qk,2),"pi: ", np.round(pik[::-1],2))
    pik = pik[::-1]
    # return qk,pik
    print("final values")
    print("q ",qk, "pi ",pik)

#binomial distribution
def coin_likelihood(roll, bias):
    # P(X | bias)
    # numHeads = np.count_nonzero(roll == 1)
    numHeads = np.count_nonzero(roll)
    flips = len(roll)
    # do nchoose k
    nck = comb(flips, numHeads, exact=True)
    return (nck * pow(bias, numHeads) * pow(1 - bias, flips - numHeads))



if __name__ == '__main__':
    p = 0.9
    r = 0.2
    pi = 0.9
    qk = np.array([p, r])
    pik = np.array([pi,1-pi])
    data = create_mixture_data(p,r,pi)
    data = np.array(data)
    print("mixed data shape",np.shape(data))
    print(data)
    print("=====================================================================")
    coin_em(qk,pik,data)

