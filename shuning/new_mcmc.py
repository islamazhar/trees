import random
import logging
import numpy as np
from math import log, sqrt, isnan, exp
from sympy import *

# from .ddt import likelihood
"""
MH class for ddt sampling
parent_move is function to remove subtree and retach subtree proposed by Neal
update_latent is function to sample latent X in tree structure
"""


# add for several samples !!!!!!!!!
class MetropolisHastingsSampler(object):
    # def __init__(self, tree, X, df):
    def __init__(self, tree, X, df, c):
        self.tree = tree  # a list of tree
        self.X = X  # np array for data
        # self.df = df # divergence function
        self.c = c  # common c
        self.likelihoods = []

        # self.tree_likelihood = []
        # add divergence list
        # self.divergence = []
        # add c in the mh class

    def initialize_assignments(self):
        self.tree.initialize_from_data(self.X)
        # add new line to inference

    # def parent_move(self):
    """
    In Neal's code, he select every non-terminal node to update the tree structure; Maybe it 
    is more efficient for updating the tree;
    --- Shuning 
    """

    def parent_move(self):
        logging.debug("Copying the tree....")
        tree = self.tree.copy()
        c = self.c
        """
        old marginal likelihood before sampling new tree
        """
        # old_likelihood = self.tree.marg_log_likelihood()

        """
        select a node randomly
        """
        old_likelihood = self.tree.marg_log_likelihood(c=c)
        # print(old_likelihood)
        assert old_likelihood is not None

        node = tree.choice()
        # debug for randomly chosen node

        # get randomly selected node's parent's assignment to avoid of leaf
        old_assignment = tree.get_assignment(node.parent)
        # print(node.parent.get_state('time'))

        # get index and state of parent of randomly chosed node to remove it
        old_index, old_state = old_assignment
        # print(old_index)

        # detach the subtree
        subtree = node.detach()

        # main function to calculate transition probability
        # backward_likelihood = tree.log_prob_assignment(old_assignment)
        backward_likelihood = tree.log_prob_assignment(assignment=old_assignment, c=c)

        points = set()

        time = float('inf')

        trial = 0

        # select a time which has smaller divergence time than subtree
        while time > subtree.get_state('time'):
            (assignment, forward_likelihood) = tree.sample_assignment(c=c, points=points, state=old_state)
            # if assignment[-1] == -1:
            #    return
            logging.debug("Candidate assignment: %s", str(assignment))
            (index, state) = assignment
            time = state['time']
            trial += 1
            if trial > 100:
                return

        tree.assign_node(subtree, assignment)
        new_likelihood = tree.marg_log_likelihood(c=c)
        assert new_likelihood is not None

        # prob for MH sampler
        a = min(1, np.exp(new_likelihood + backward_likelihood - old_likelihood - forward_likelihood))
        print(a)
        # case when we accept the current proposal
        if np.random.random() < a:
            self.tree = tree
            self.tree._marg_log_likelihood = new_likelihood
            return

    """
    This is my rewrite version close to Neal's C code;
    This function may be more closer to the C result. However, there is difference when 
    using the location of non-terminal nodes, I doubt difference between python version and C version
    ----Shuning 
    """

    # def parent_move(self):
    def parent_move2(self):
        # print("inside shuning's parent move")
        logging.debug("Copying the tree....")
        tree = self.tree.copy()
        c = self.c

        """
        select a node randomly Mazhar: But the node can not be a leaf node. 
        """
        # for node in tree.dfs():
        #    if node.is_root() or node.is_leaf():
        #        continue

        # y = node.parent.get_state('time')
        # dft_log_prob_node

        node = tree.choice()
        while node.is_leaf():
            node = tree.choice()

        logprob0 = tree.dft_log_prob_node(node)
        if isnan(logprob0) or logprob0 == -float('inf'):
            assert "logprob0 is incorrect"

        logprob0 += tree.dft_log_prob_paths(node, c)
        # print("LogProb0", logprob0)
        logprob0 -= tree.dft_log_prob_path(node, c)
        # there is nan in the update of tree probabilibty

        # get randomly selected node's parent's assignment to avoid of leaf
        # old_assignment = tree.get_assignment(node.parent)
        old_assignment = tree.get_assignment(node.parent)

        # get index and state of parent of randomly chosen node to remove it
        old_index, old_state = old_assignment

        # detach the subtree
        subtree = node.detach()

        """
        Save detached node
        """
        points = set()
        time = float('inf')
        trial = 0
        # select a time which has smaller divergence time than subtree
        while time > subtree.get_state('time'):
            (assignment, forward_likelihood) = tree.sample_assignment(c=c, points=points, state=old_state)
            logging.debug("Candidate assignment: %s", str(assignment))
            (index, state) = assignment
            time = state['time']
            trial += 1
            if trial > 100:
                return

                # return -0.5
        """
        Save newly generated divergence time and tree structure.
        """
        # assign the node to the new location;
        tree.assign_node(subtree, assignment)
        # use subtree in a changed new tree to calculate likelihood;

        logprob1 = tree.dft_log_prob_node(subtree)
        if isnan(logprob0):
            assert "log prob can not be NAN"
        logprob1 += tree.dft_log_prob_paths(subtree, c)
        logprob1 -= tree.dft_log_prob_path(subtree, c)

        # print(tree.dft_log_prob_path(subtree, c))

        #######  do not use exp in calculating the acceptance ratio;
        ####### use log-version directly;
        # print(logprob0, " =  ", logprob1)

        delta = (logprob0 - logprob1)[0]
        # print(delta)

        # a = min(1, np.exp(logprob0 - logprob1))
        # case when we accept the current proposal
        # if np.random.random() < a:
        #    self.tree = tree
        #    #self.tree._marg_log_likelihood = new_likelihood

        # return delta for testing;
        if np.random.random() < min(1.0, exp(delta)):
            self.tree = tree
            print('Accepting the alternate tree...')
            # return delta for testing
            return delta

        print("Not accepting the alternate tree...")
        return -1

    """
    Function corresponding to met_terminals in Neal's C code
    """

    def met_terminals(self):
        return

    """
    Function corresponding to met_nonterminals in Neal's C code;
    """

    def met_nonterminals(self):
        return

    """
    Function corresponding to slice-positions in Neal's C code;
    Later revise and update;

    def slice_positions(self):
        for (y = 1; y<=N_train; y++)
      {
        it->slice_calls += 1;

        /* Find node along path to update. */
        switch (method)
        {
          case 1: /* Random position along path */
          {
            int cnt;

            cnt = 0;
            for (a = st[dt].parents[y]; a!=0; a = st[dt].parents[a])
            { cnt += 1;
            }

            b = y;
            a = st[dt].parents[b];
            for (cnt = rand_int(cnt); cnt>0; cnt--)
            { if (a==0) abort();
              b = a;
              a = st[dt].parents[b];
            }

            b = dft_sibling (st[dt].parents, st[dt].nodes, b);

            break;
          }
        return
    """

    """
    Original python code for latent locations;
    """

    def update_latent(self):
        self.tree.sample_latent()

    """
    A function to update divergence parameter c

    def update_divergence(self):
        tree = self.tree.copy() # copy tree for benefit of update
        old_c = self.c
        #print(self.tree._marg_log_likelihood)
        # get old_c tree likelihood

        old_likelihood = self.tree.marg_log_likelihood(old_c)
        #print(old_likelihood)
        # sample new c in line
        new_c = np.random.lognormal(old_c, 0.1, 1)[0]

        # forward and backward probability
        backward_likelihood = self.log_normal_pdf(old_c, new_c)
        forward_likelihood = self.log_normal_pdf(new_c, old_c)

        logging.debug("Calculate new likelihood")

        # check for this step to calculate new tree likelihood based on new c
        #*************************************************
        new_likelihood = tree.marg_log_likelihood(c=new_c)
        #*************************************************


        # to check if we have changed c to change tree-likelihood!!!!!!!!
        #assert new_likelihood != old_likelihood

        a = min(1, np.exp(new_likelihood + backward_likelihood - old_likelihood - forward_likelihood))


        # if we accept the new_c, we will assign it to self.c field
        if np.random.random() < a:
            self.c = new_c
            return 
    """

    # set hyperparameters(sigma2) for log-normal distribution is 1
    # def log_normal_pdf(self, x, mu):
    # this is incorrect form of log-normal distribution
    #    pdf = -np.log(x * np.sqrt(2.0 * np.pi)) - 0.5 * (np.log(x) - mu) ** 2
    #    return pdf
    def log_normal_pdf(self, logx, logc, sigma=1):
        pdf = - np.log(np.sqrt(2.0 * np.pi) * sigma) - 0.5 * ((logx - logc) ** 2) / float(sigma ** 2)
        return pdf

    def lognormal_pdf(self, newc, oldc, sigma=1):
        pdf = - np.log(np.sqrt(2.0 * np.pi) * sigma) - np.log(newc) - \
              0.5 * ((np.log(newc) - np.log(oldc)) ** 2) / float(sigma ** 2)
        return pdf

    def get_Jn(self, node):
        cl_counts = [cl.leaf_count() for cl in node.children]
        # print(counts)
        n_left = cl_counts[0]
        n_right = cl_counts[1]
        counts = node.leaf_count()
        return harmonic(n_left - 1) + harmonic(n_right - 1) - harmonic(counts - 1)

    """
    This is a helper function to traverse a tree to find all of internal nodes;
    """

    def helper(self, tree):
        node = tree.root
        # if the node is leaf, return the values
        if node.is_leaf():
            return 0

        # okay to add root in this line since node_time for root is 0.0
        node_time = node.get_state('time')
        left_child, right_child = node.children
        cur_value = self.harmonic(node) * log(1 - node_time)

        return cur_value + self.helper(left_child) + self.helper(right_child)

    def sample(self, c):
        self.tree = self.tree.copy()
        random.choice([self.parent_move, self.update_latent])()
        self.likelihoods.append(self.tree.marg_log_likelihood(c=c))

    """
    wrapper function for multiple functional data by mcmc
    a_c: shape parameter for gamma prior
    b_c: rate parameter for gamma prior
    """

    def wrapper(self, n_samples, tree_list, update_c=0, a_c=1, b_c=1):
        likelihoods = [0] * n_samples
        new_likelihoods = [0] * n_samples
        # add true likelihood to test
        # true_likelihood = [0] * n_samples
        # true_c = 0.25
        old_c = self.c

        # summation of tree likelihood in one group
        for i in range(n_samples):
            # update for each tree, store likelihood for convenient
            # previous tree list
            self.tree = tree_list[i].copy()
            """
            Important function to update tree structure.
            """
            # do K scans in mcmc
            self.parent_move2()

            """
            Save tree in self.parent_move2 step, including the tree structure, detached node, new generated 
            divergence time.....
            """
            # random.choice([self.parent_move, self.update_latent])()
            # self.parent_move
            tree_list[i] = self.tree.copy()
            # likelihoods[i] = tree_list[i].marg_log_likelihood(c=old_c)
            likelihoods[i] = tree_list[i].calculate_marg_tree_structure(c=old_c)
            print(likelihoods[0])

        """
        No need to look at this part in testing;
        """
        # update c periodic
        if update_c == 1:
            old_tree_lik = sum(likelihoods)
            old_logc = np.log(old_c)
            # change sigma for deviance in log-normal distribution
            # new_c = np.random.lognormal(old_c, 0.5, 1)[0]
            new_logc = np.random.normal(loc=old_logc, scale=1.0)
            new_c = np.exp(new_logc)
            # print(new_c)

            # *******************Log-normal distribution********************
            backward_likelihood = self.log_normal_pdf(old_logc, new_logc) - old_logc
            forward_likelihood = self.log_normal_pdf(new_logc, old_logc) - new_logc
            # *******************Log-normal distribution********************

            logging.debug("Calculate new likelihood")
            # check for this step to calculate new tree likelihood based on new c
            # *************************************************

            for i in range(n_samples):
                self.tree = tree_list[i].copy()
                new_likelihoods[i] = self.tree.calculate_marg_tree_structure(c=new_c)

            new_tree_lik = sum(new_likelihoods)
            if isnan(new_tree_lik):
                assert "log prob can not be NAN"
            # true_tree_lik = sum(true_likelihood)

            # print(true_tree_lik - new_tree_lik)

            # *************************************************
            # assert new_likelihood != old_likelihood
            a = min(1, np.exp(new_tree_lik + backward_likelihood - old_tree_lik - forward_likelihood))

            if np.random.random() < a:
                self.c = new_c

            print(self.c)

        # propose a discrete uniform distribution for updating c
        if update_c == 2:
            proposed_c = [0.5, 5, 10]
            max_new_tree_lik = -1000000
            for c in proposed_c:
                for i in range(n_samples):
                    self.tree = tree_list[i].copy()
                    new_likelihood[i] = self.tree.marg_log_likelihood(c=c)
                new_tree_lik = sum(new_likelihood)
                # update c to search for discrete maximum likelihood
                if new_tree_lik > max_new_tree_lik:
                    max_new_tree_lik = new_tree_lik
                    self.c = c

        # using gibbs sampling for c(gamma distribution for c to check out)
        """      
        if update_c == 3:
            # traverse for all internal nodes in this line to get posterior rate parameter
            val = 0
            num_internal = 0
            for i in range(n_samples):
                self.tree, tree = tree_list[i].copy()
                val += self.helper(tree) 
                #num_internal += self.

            # generate a gamma random variable by updated parameters
            c = np.random.gamma(shape = a_c + num_internal, scale = 1 / (b_c + val))
        """

        """
        Function corresponding to Neal's slice_div in C code;
        """
        if update_c == 4:
            alpha = 0.1  # see neal's code for reference;
            width = 0.5  # see Neal's code for reference;
            w = 1
            # if self.width == 0 or prior == 0:
            #    break
            omega = 1 / (width * width)

            cur_ll = sum(likelihoods)
            cur_val = -2 * log(old_c)  # revise this line
            low_val = cur_val - w * np.random.uniform(0.0, 1.0)
            high_val = low_val + w
            slice_lp = cur_val * alpha / 2 - exp(cur_val) * alpha / (2 * omega) + cur_ll - (
                -log(np.random.uniform(0.0, 1.0)))
            # rand_uniopen(), unif(0,1)

            new_val = low_val + (high_val - low_val) * np.random.uniform(0.0, 1.0)
            new_c = exp(-new_val / 2)
            for i in range(n_samples):
                # self.tree = tree_list[i].copy()
                new_likelihoods[i] = self.tree.calculate_marg_tree_structure(c=new_c)
            new_ll = sum(new_likelihoods)
            new_lp = new_val * alpha / 2 - exp(new_val) * alpha / (2 * omega) + new_ll

            while new_lp < slice_lp:
                if (new_val > cur_val):
                    high_val = new_val
                else:
                    low_val = new_val

                new_val = low_val + (high_val - low_val) * np.random.uniform(0.0, 1.0)
                new_c = exp(-new_val / 2)
                # new c level;
                for i in range(n_samples):
                    # self.tree = tree_list[i].copy()
                    new_likelihoods[i] = self.tree.calculate_marg_tree_structure(c=new_c)

                new_ll = sum(new_likelihoods)
                new_lp = new_val * alpha / 2 - exp(new_val) * alpha / (2 * omega) + new_ll

            # print(self.c)
            self.c = new_c
            print(self.c)
        return tree_list, likelihoods

