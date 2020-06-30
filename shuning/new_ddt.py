import logging
import numpy as np
from trees.tree import Tree
from math import log, exp
import theano.tensor as T
from theanify.theanify import theanify, Theanifiable
from theano.tensor.shared_randomstreams import RandomStreams

"""
DirichletDiffusionTree is object inherant Tree object
initialize_from_data to initialize tree strutcure from data
calculate_node_likelihood: calculate likelihood for each node recursively
"""
class DirichletDiffusionTree(Tree):
    def __init__(self, n, root=None, **params):
        super(DirichletDiffusionTree, self).__init__(root=root,**params)
        self.params = params
        self._marg_log_likelihood = None
        # add harmonic function in this line;

        # store harmonic function in this line for future computation;
        self.harmonic = np.zeros(n+1)
        self.harmonic[0] = 0
        for i in range(1,n+1):
            self.harmonic[i] = self.harmonic[i-1] + 1/i

    def initialize_from_data(self, X):
        logging.debug("Initializing tree from data...")
        X = np.array(X)
        N, _ = X.shape
        points = set(range(N))
        super(DirichletDiffusionTree, self).initialize_assignments(points)
        self.reconfigure_subtree(self.root, X)

    def reconfigure_subtree(self, root, X):
        if root.is_root():
            root_time = 0.0
        else:
            root_time = root.get_state('time')
        for node in self.dfs(node=root):
            if node == root:
                node.set_state('time', root_time)
                node.set_state('latent_value', sum(n.get_state('latent_value') for n in node.children) /
                               float(len(node.children)))
            elif node.is_leaf():
                node.set_state('time', 1.0)
                node.set_state('latent_value', X[node.point].ravel())
            else:
                min_time = min(n.get_state('time') for n in node.children)
                new_time = min_time - (min_time - root_time) / 2.0
                node.set_state('time', new_time)
                node.set_state('latent_value', sum(n.get_state('latent_value') for n in node.children) /
                                           float(len(node.children)))

    def get_leaves(self, node):
        if node.is_leaf():
            return 1
        if not node.is_root() and node is None:
            return 0
        else:
            left_child, right_child = node.children
            return self.get_leaves(node.left_child) + self.get_leaves(node.right_child)

    """
    A recursive function to calculate node likelihood
    *****************************************
    *********** Check the formula ***********
    *****************************************
    """
    def calculate_node_likelihood(self, c, node=None):
        """
        if we do not input the node, the node will be root
        """
        df = self.df
        node = node or self.root

        if node.is_leaf():
            """
            tree prob is 0 if the node is a leaf
            path count is 1 if the node is a leaf
            data prob is transition_probability(brownian motion) 
            if the node is a leaf
            """
            #return 1, 0, self.likelihood_model.transition_probability(node.parent, node)
            return 0, self.likelihood_model.transition_probability(node.parent, node)
        node_time = node.get_state('time')
        left_child, right_child = node.children

        if node.is_root():
            tree_prob, data_prob = self.calculate_node_likelihood(c, left_child)
            right_tree_prob, right_data_prob = self.calculate_node_likelihood(c, right_child)
            return tree_prob + right_tree_prob, data_prob + right_data_prob


        if not node.is_root():
            """
            checking formula for log_no_divergence in Neal
            """
            num_leaves = self.get_leaves(node)

            tree_prob = - df.log_no_divergence(node.parent.get_state('time'), node_time, c) * self.harmonic[num_leaves-1]
            tree_prob += df.log_divergence(node_time, c)

        data_prob = self.likelihood_model.transition_probability(node.parent, node)

        #path_count, tree_prob, data_prob = self.calculate_node_likelihood(c=c, node=left_child)
        #tree_prob, data_prob = self.calculate_node_likelihood(c=c, node=left_child)

        """
        tree prob is determined by divergence time and non-divergence time
             divergence time is determined by path count
             no divergence time is determined by number of leaf nodes in each path
        data prob is determined by brownian motion
        """
        tree_prob += self.calculate_node_likelihood(c=c, node=left_child)[0]
        tree_prob += self.calculate_node_likelihood(c=c, node=right_child)[0]

        data_prob += self.calculate_node_likelihood(c=c, node=left_child)[1]
        data_prob += self.calculate_node_likelihood(c=c, node=right_child)[1]
        
        return tree_prob, data_prob

    
    def calculate_marg_log_likelihood(self, c):
        assert self.root is not None
        tree_structure, data_structure = self.calculate_node_likelihood(c=c, node=None)
        self._marg_log_likelihood = tree_structure + data_structure

    def get_tree_log_likelihood(self, c):
        assert self.root is not None
        tree_structure, data_structure = self.calculate_node_likelihood(c=c, node=None)
        return tree_structure

    
    # return tree structure log_likelihood to check for convergence 
    def calculate_marg_tree_structure(self, c):
        assert self.root is not None
        tree_structure, data_structure = self.calculate_node_likelihood(c=c, node=None)
        return tree_structure

    
    def marg_log_likelihood(self, c):
        # calculate marg log likelihood
        self.calculate_marg_log_likelihood(c=c)
        return self._marg_log_likelihood

    
    def sample_assignment(self, c, node=None, points=None, index=None, state=None):
        df = self.df
        # divergence function class
        node = node or self.root
        points = points
        # index is direction of node pointing to root
        index = index or ()
        # state is time and location of a node
        state = state or {}
        
        if node.children is None:
            state = {}
            state['time'] = 2.0
            return ((-1,), state), 1

        # if we haven't find a divergence time 

        #if node.children is None:
        #    return 
        #if node.children is None:

        # debug for recursion algorithm
        # print(1)
        # assert node.children is not None
        counts = [cl.leaf_count() for cl in node.children]
        logging.debug("Path counts: %s" % str(counts))
        total = float(sum(counts))

        left_prob = counts[0] / total
        u = np.random.random()
        choice = None
        idx = -1

        # randomly sample going left or right
        # choice is a node in current tree structure
        if choice is None:
            if u < left_prob:
                choice = node.children[0]
                idx = 0 # going to left
            else:
                choice = node.children[1]
                idx = 1 # going to right

        # branching probability in building the tree
        prob = np.log(counts[idx]) - np.log(total)
        logging.debug("Branching: %f" % prob)
        
        node_time = node.get_state('time')
        choice_time = choice.get_state('time')

        assert node_time <= 1.0 and choice_time <= 1.0
        assert node_time < choice_time
        #print(node_time)
        #print(choice_time)

        no_diverge_prob = (df.cumulative_divergence(node_time, c=c) - df.cumulative_divergence(choice_time, c=c)) / \
            counts[idx]
        #no_diverge_prob = (df.cumulative_divergence(node_time) - df.cumulative_divergence(choice_time)) / \
        #    counts[idx]
        #print(no_diverge_prob)
    
        # sample to decide if it will diverge or not   
        u = np.random.random()

        if u < np.exp(no_diverge_prob):
            # print(np.exp(no_diverge_prob))
            # case when we decide not to diverge
            prob += no_diverge_prob
            assignment, p = self.sample_assignment(c=c, node=node.children[idx],
                                                   points=points,
                                                   index=index + (idx,),
                                                   state=state)
            return assignment, prob + p

        else: # in the end, we will diverge

            # the case we decide to diverge
            #sampled_time, _ = df.sample(node_time, choice_time, counts[idx])
            sampled_time, _ = df.sample(node_time, choice_time, counts[idx], c=c)
            diverge_prob = df.log_pdf(sampled_time, node_time, counts[idx], c=c)
            #diverge_prob = df.log_pdf(sampled_time, node_time, counts[idx])
            logging.debug("Diverging at %f: %f" % (sampled_time, diverge_prob))
            prob += diverge_prob
            state['time'] = sampled_time
            return (index + (idx,), state), prob

    """
    A recursive function to get prob of node from root(default node is root)
    """
    def log_prob_assignment(self, assignment, c, node=None):
        node = node or self.root

        (idx, state) = assignment
        # print(idx)
        time = state['time']
        # assert for debug, node is not root
        assert idx is not ()

        df = self.df

        first, rest = idx[0], idx[1:]
        
        # get counts for left and right subtrees
        counts = [cl.leaf_count() for cl in node.children]
        #print(counts)
        total = float(sum(counts))
        # get branching prob in this line
        prob = np.log(counts[first]) - np.log(total)

        logging.debug("Branching prob: %f" % prob)


        node_time = node.get_state('time')
        # finish recursively building likelihood
        # we will compute divergence likelihood in this moment
        if len(idx) == 1:
            #diverge_prob = df.log_pdf(node_time, time, counts[first])
            diverge_prob = df.log_pdf(node_time, time, counts[first], c=c)
            logging.debug("Diverging at %f: %f" %(time, diverge_prob))
            return prob + diverge_prob

        choice = node.children[first]
        choice_time = choice.get_state('time')
        
        no_diverge_prob = (df.cumulative_divergence(node_time, c=c) - df.cumulative_divergence(choice_time, c=c)) / \
            counts[first]
        #no_diverge_prob = (df.cumulative_divergence(node_time) - df.cumulative_divergence(choice_time)) / \
        #    counts[first]
        # recursively finding the path to the current node
        return prob + no_diverge_prob + self.log_prob_assignment((rest, state), c=c, node=node.children[first])

    def assign_node(self, node, assignment):
        (idx, state) = assignment
        assignee = self.get_node(idx)
        assignee.attach(node)
        node.parent.state.update(state)
    
    """
    Function to sample internal node 
    input: tree object
    """
    def sample_latent(self):
        for node in self.dfs():
            if not node.is_leaf():
                lv = self.likelihood_model.sample_transition(node, node.parent)
                node.set_state('latent_value', lv)
                node.delete_cache("likelihood")

    def node_as_string(self, node):
        return str(node.get_state('time'))

    def get_parameters(self):
        return {
            "df",
            "likelihood_model",
        }
