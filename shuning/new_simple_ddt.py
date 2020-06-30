"""
A script to run simple ddt
"""
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from trees.util import plot_tree
from trees.ddt import GaussianLikelihoodModel
from shuning import  DirichletDiffusionTree, Inverse
#from trees.ddt import DirichletDiffusionTree, GaussianLikelihoodModel
from shuning import MetropolisHastingsSampler
#from trees.new_mcmc import SPRSampler
from tqdm import tqdm
from shuning import Generate

"""
This script is to generate ddt from c and sigma2
"""

if __name__ == "__main__":
	# set 2-d dimension
    # generate dirichlet diffusion tree data for simulation
    # test mcmc for one-sample case!!!!!!!!!!!!
    n_samples = 1
    n = 50
    p = 1
 
    all_dat = np.zeros((n_samples, n, p))
    print("Simulate the First Tree Data")
    simulator = Generate(c=1)
    #all_dat2 = np.zeros((n_samples, n))
    #print("Simulate the second Tree Data")
    #simulator2 = Generate(c=0.5)
    
    # test for code correctness
    # simulate 30 datasets each with 100 datapoints in 1-d dimension
    # generate 2d uniform distribution
    for i in range(n_samples):
        data, _ = simulator.gen(n, p)
        #data = np.random.uniform(0,1,(n, 2))
        all_dat[i] = data
        
        #data2, tree2 = simulator2.gen(n, p)
        #all_dat2[i] = data2.ravel()
        #true_lik2.append(tree2.calculate_marg_log_likelihood(c=10))
    print("Finish Simulating Data")

    ######
    #D = 2 
    #N = 100
    #X = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D), size=N).astype(np.float32)
    #print(X.shape)
    df = Inverse()
    lm = GaussianLikelihoodModel(sigma=np.eye(p), mu0=np.zeros(p), sigma0=np.eye(p))
    print("Get initial likelihood")
    #df2 = Inverse()
    #lm2 = GaussianLikelihoodModel(sigma=np.eye(p), mu0=np.zeros(p), sigma0=np.eye(p))
    #print("Get initial likelihood")

    ddt = DirichletDiffusionTree(df=df, likelihood_model=lm)
    #ddt2 = DirichletDiffusionTree(df=df2, likelihood_model=lm2)


    print("Initialize Dirichlet Diffusion Tree")
    tree_list = []
    #tree_list2 = []

    """"""
    for i in range(n_samples):
        mh = MetropolisHastingsSampler(ddt, all_dat[i].reshape(n,p), df, 1)
        #spr = SPRSampler(ddt, all_dat[i], 0.5)
        #mh2 = MetropolisHastingsSampler(ddt2, all_dat2[i].reshape(n,1), df2, 5)
        mh.initialize_assignments()

        #mh2.initialize_assignments()
        #spr.initialize_assignments()
        tree_list += [mh.tree]
        print(mh.tree.dfs(mh.tree.root))
        #tree_list2 += [mh2.tree]

    #print(tree_list[0].marg_log_likelihood(c=1))

    """
    Start of Markov-chain monte carlo simulation;
    Assume we are going to get the first mcmc update for tree
    """
    # number of mcmc is 1;
    n_mcmc = 1
    divergence = [0]*n_mcmc
    #divergence2 = [0]*n_mcmc
    tree_lik = np.zeros((n_mcmc, n_samples))
    #tree_lik2 = np.zeros((n_mcmc, n_samples))

    i = 0
    for _ in tqdm(range(n_mcmc)):
    #for i in range(5000):
        if i%500 == 0:
            print(i)
        # update c three times after updating tree_likelihood

        """
        No need to update c in our testing, so update_c=0
        See the wrapper function in new_mcmc.py
        """
        if i > 5000:
            tree_list, tree_lik[i,] = mh.wrapper(n_samples, tree_list, update_c=0)
            #tree_list2, tree_lik2[i,] = mh2.wrapper(n_samples, tree_list2)
        else:

            """
            The testing part is in the mh.wrapper function, see wrapper function for reference;
            """
            tree_list, tree_lik[i,] = mh.wrapper(n_samples, tree_list, update_c=0)
            # tree_list2, tree_lik2[i,] = mh2.wrapper(n_samples, tree_list2)
        divergence[i] = mh.c
        #divergence2[i] = mh2.c
        i += 1
    
    # save divergence and treelikelihood in this line
    # No need to plot at this step;
    """
    plt.figure() 
    plt.hist(divergence[:], bins = 20)
    plt.show()

    #plt.figure()
    #plt.hist(divergence2[:], bins=20)
    #plt.show()

    plt.figure()
    plt.plot(tree_lik[:,0])
    plt.show()
    
    #plt.plot(tree_lik[1,:])
    #plt.show()
    
    plt.figure()
    print("Get Tree Structure")
    plot_tree(tree_list[0])
    plt.show()

    #mh = MetropolisHastingsSampler(ddt, data, df, 1)
    #mh.initialize_assignments() 
    #print(ddt.root.children[0].children[0].children[0].get_state('time'))
    """
    #likelihoods = []

    #print(mh.tree.root.children[0].get_state('time'))
    #assert mh.tree is not None

    #for i in range(1000):
    #mh.sample(3000)
        #tree = mh.tree.copy()
        #print(mh.tree._marg_log_likelihood)
        #c = mh.c
        #if i == 0:
            # calculate marginal log likelihood for the first time
        #     mh.tree._marg_log_likelihood = mh.tree.marg_log_likelihood(c)
        #else:
        #    random.choice([mh.parent_move, mh.update_latent])()
        #    mh.update_divergence()
            # store marginal likelihood for all structure
            # store all tree structure likelihood
            #self.tree_likelihood.append(self.tree.calculate_marg_tree_structure(self.c))
            
        #    likelihoods.append(mh.tree._marg_log_likelihood)
        #    divergence.append(mh.c)
            #print(self.divergence)

    #plt.figure()
    #print("Get likelihood")
    #plt.plot(likelihoods)
    #plt.show()
    #plt.plot(mh.tree_likelihood)
    #plt.plot(divergence)
    #plt.show()
    #plt.savefig("divergence.png")
    #plt.show()
    """
    Check tree structures
    """

