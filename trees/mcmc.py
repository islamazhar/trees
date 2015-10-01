import random
import logging
import numpy as np

class MetropolisHastingsSampler(object):

    def __init__(self, tree, X, constraints=[]):
        self.tree = tree
        self.X = X
        self.last_move = None
        self.constraints = constraints

    def initialize_assignments(self):
        self.tree.initialize_from_data(self.X)

    def parent_move(self):
        tree = self.tree.copy()

        old_likelihood = self.tree.marg_log_likelihood()
        logging.debug("Old Marginal Likelihood: %f" % old_likelihood)

        node = tree.choice()
        old_assignment = tree.get_assignment(node.parent)
        parent = node.detach()

        backward_likelihood = tree.log_prob_assignment(old_assignment)
        logging.debug("Backward Likelihood: %f" % backward_likelihood)

        points = set()
        if len(self.constraints) > 0:
            points = parent.points()

        time = float('inf')

        try_counter = 0
        while time > parent.children[0].get_state('time'):
            (assignment, forward_likelihood) = tree.sample_assignment(constraints=self.constraints,
                                                                    points=points)
            (index, time) = assignment
            try_counter += 1
            if try_counter > 500:
                return

        tree.assign_node(parent, assignment)
        new_likelihood = tree.marg_log_likelihood()

        logging.debug("New Marginal Likelihood: %f" % old_likelihood)
        logging.debug("Forward Likelihood: %f" % forward_likelihood)

        a = min(1, np.exp(new_likelihood + backward_likelihood - old_likelihood - forward_likelihood))
        if np.random.random() < a:
            logging.debug("Accepted new tree with probability: %f" % a)
            self.tree = tree
            return
        logging.debug("Rejected new tree with probability: %f" % a)

    def update_latent(self):
        self.tree.sample_latent()

    def sample(self):
        random.choice([self.parent_move, self.update_latent])()
