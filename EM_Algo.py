import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import norm


# Function importing Dataset
def importdata():
    # data = pd.read_csv(
    #     '2gaussian.txt',
    #     sep=' ', header=None)
    data = pd.read_csv(
        '3gaussian.txt',
        sep=' ', header=None)
    print(data.shape)
    return data.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, data.shape[1]-1]
    return x, y


def em_update(X,mu,cov,pi,num_cluster):
    iter = 100
    log_likelihoods = []  # log likehoods per iteration for checking convergence
    count1 = 0
    count2 = 0
    count3 = 0
    z_im = 0
    reg_cov = 1e-6 * np.identity(len(X[0]))

    #z_im gives us the fraction of the probability that x_i belongs to cluster c
    # over the probability that x_i belonges to any of the cluster c
    # (Probability that x_i occurs given the 3 Gaussians)


    for i in range(iter):
        """E Step"""
        z_im = np.zeros((len(X), len(cov))) # 6000 x 2
        for mu_val, co, p, num in zip(mu, cov, pi, range(len(z_im[0]))):
            co += reg_cov
            mn = multivariate_normal(mean=mu_val, cov=co)
            # total of <z> for all m  #6000 x 2 matrix
            z_all = [pi_c * multivariate_normal(mean=mu_c, cov=cov_c).pdf(X) for pi_c, mu_c, cov_c in zip(pi, mu, (cov + reg_cov))]
            deno = np.sum(z_all, axis=0)
            z_im[:, num] = p * mn.pdf(X) / deno   # expected val for each cluster(mixture)

        # print(z_im[:10,:])
        """M Step"""
        mu = []
        cov = []
        pi = []
        log_likelihood = []

        for c in range(len(z_im[0])):
            #update mu
            m_c = np.sum(z_im[:, c], axis=0)
            mu_c = (1 / m_c) * np.sum(X * z_im[:, c].reshape(len(X), 1), axis=0) # mu_c 1x2
            mu.append(mu_c)
            #update cov
            cov.append(((1 / m_c) * np.dot((np.array(z_im[:, c]).reshape(len(X), 1) * (X - mu_c)).T,
                                           (X - mu_c))) + reg_cov)
            pi.append(m_c / np.sum(z_im))  #check
            # np.sum(z_im) = the number of data points

        # The elements in pi_new must add up to 1
        log_likelihoods.append(np.log(np.sum([pi_k * multivariate_normal(mu[i], cov[j]).pdf(X) for pi_k, i, j in
                                              zip(pi, range(len(mu)), range(len(cov)))])))

    # print(z_im)
    if num_cluster == 2:
        for prob in z_im:
            if prob[0] > prob[1]:
                count1 += 1
            else:
                count2 += 1
    else:
        for prob in z_im:
            if prob[0] > prob[1] and prob[0] > prob[2] :
                count1 += 1
            elif prob[1] > prob[0] and prob[1] > prob[2]:
                count2 += 1
            elif prob[2] > prob[0] and prob[2] > prob[1]:
                count3 += 1

    # mean_ = np.unique(np.round(mu), axis=0)
    mean_ = np.round(mu)
    # cov_ = np.unique(np.round(cov), axis=0)
    # cov_ = np.round(cov)
    cov_ = cov
    # print(log_likelihoods)
    print("mean", mean_)
    print("cov", cov_)
    if num_cluster == 2:
        print("n1",count1,"n2",count2)
    else:
        print("n1", count1, "n2", count2,"n3",count3)

    #plot log-likelihood curve to check for convergence
    fig2 = plt.figure(figsize=(10, 10))
    ax1 = fig2.add_subplot(111)
    ax1.set_title('Log-Likelihood')
    ax1.plot(range(0, iter, 1), log_likelihoods)  #to check convergence
    plt.show()
    # return np.round(mu), np.round(cov)

    # """Plot the data"""
    # fig = plt.figure(figsize=(10, 10))
    # ax0 = fig.add_subplot(111)
    # for i in range(len(z_im)):
    #     ax0.scatter(X[i][0], 0.0, c=np.array([z_im[i][0], z_im[i][1]]), s=100)
    # """Plot the gaussians"""
    # for g, c in zip([norm(loc=mu[0], scale=cov[0]).pdf(np.linspace(-20, 20, num=60)),
    #                  norm(loc=mu[1], scale=cov[1]).pdf(np.linspace(-20, 20, num=60))],['r','g']):
    #     ax0.plot(np.linspace(-20, 20, num=60), g, c=c)




if __name__ == '__main__':
    seed = 1
    mixture_data = importdata()
    num_cluster = 3
    X = mixture_data

    #Step1: Initialization of parameters mu cov pi

    # M or C = num_cluster , N = dimensions
    mu = np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(num_cluster, len(X[0])))   # M x N
    cov = np.zeros((num_cluster, len(X[0]), len(X[0]))) # M X N x N

    #to create symmetric covariance matrices with ones on the diagonal
    for dim in range(len(cov)):
        np.fill_diagonal(cov[dim], 5)


    pi = np.ones(num_cluster) / num_cluster # equally distributed mixture
    em_update(X,mu,cov,pi,num_cluster)







