import logging
import numpy as np
from trees.tree import Tree

class DirichletDiffusionTree(Tree):

    def __init__(self, root=None, constraints=[], **params):
        self.node_ID = 0
        super(DirichletDiffusionTree, self).__init__(root=root,
                                                     constraints=constraints,
                                                     **params)
        self._marg_log_likelihood = None

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

    def calculate_node_likelihood(self, c, node=None):
        """
        if we do not input the node, the node will be root
        """
        node = node or self.root

        if 'likelihood' in node.cache:
            return node.get_cache('likelihood')

        if node.is_leaf():
            """
            tree prob is 0 if the node is a leaf
            path count is 1 if the node is a leaf
            data prob is transition_probability(browian motion) 
            if the node is a leaf
            """
            return 1, 0, self.likelihood_model.transition_probability(node.parent, node)

        node_time = node.get_state('time')
        left_child, right_child = node.children
        path_count, tree_prob, data_prob = self.calculate_node_likelihood(c=c, node=left_child)
        """
        tree prob is determined by divergence time and non-divergence time
             divergence time is determined by path count
             no divergence time is determined by number of leaf nodes in each path
        data prob is determined by brownian motion76UY
        """
        if not node.is_root():
            """
            checking formula for log_no_divergence in Neal
            """
            tree_prob += self.df.log_no_divergence(node.parent.get_state('time'), node_time, path_count, c)
            tree_prob += self.df.log_divergence(node_time, c)
            #tree_prob += self.df.log_no_divergence(node.parent.get_state('time'), node_time, path_count)
            #tree_prob += self.df.log_divergence(node_time)

        data_prob += self.likelihood_model.transition_probability(node.parent, node)

        right_path_count, right_tree_prob, right_data_prob = self.calculate_node_likelihood(c, node=right_child)
        result = path_count + right_path_count, tree_prob + right_tree_prob, data_prob + right_data_prob
        # store node cache in this line
        node.set_cache('likelihood', result)
        return result

    def calculate_marg_log_likelihood(self):
        assert self.root is not None
        _, tree_structure, data_structure = self.calculate_node_likelihood()
        self._marg_log_likelihood = tree_structure + data_structure

    def marg_log_likelihood(self):
        if self._marg_log_likelihood is None:
            self.calculate_marg_log_likelihood()
        return self._marg_log_likelihood

    def sample_assignment(self, node=None, constraints=None, points=None, index=None,
                          state=None):
        node = node or self.root
        constraints = constraints or self.constraints
        points = points or frozenset()
        index = index or ()
        df = self.df

        state = state or {}

        logging.debug("Sampling assignment at index: %s" % str(index))

        counts = [c.leaf_count() for c in node.children]
        logging.debug("Path counts: %s" % str(counts))
        total = float(sum(counts))

        if len(constraints) > 0:
            for idx, child in enumerate(node.children):
                if child.is_required(constraints, points):
                    constraints = node.prune_constraints(constraints, points, idx)
                    logging.debug("Child is required: %u" % idx)
                    return self.sample_assignment(node=node.children[idx],
                                                constraints=constraints,
                                                points=points,
                                                index=index + (idx,),
                                                state=state)
        left_prob = counts[0] / total
        u = np.random.random()
        choice = None
        idx = -1

        if len(constraints) > 0:
            for i, child in enumerate(node.children):
                if child.is_path_required(constraints, points):
                    idx = i
                    choice = child
                    break
                if child.is_path_banned(constraints, points):
                    idx = 1 - i
                    choice = node.children[idx]
                    break

        if choice is None:
            if u < left_prob:
                choice = node.children[0]
                idx = 0
            else:
                choice = node.children[1]
                idx = 1

        prob = np.log(counts[idx]) - np.log(total)
        logging.debug("Branching: %f" % prob)

        node_time = node.get_state('time')
        choice_time = choice.get_state('time')

        if choice.is_banned(constraints, points):
            sampled_time, _ = df.sample(node_time, choice_time, counts[idx])
            diverge_prob = df.log_pdf(node_time, sampled_time, counts[idx])
            prob += diverge_prob
            state['time'] = sampled_time
            return (index + (idx,), state), prob

        constraints = node.prune_constraints(constraints, points, idx)

        no_diverge_prob = (df.cumulative_divergence(node_time) - df.cumulative_divergence(choice_time)) / \
            counts[idx]
        u = np.random.random()
        if u < np.exp(no_diverge_prob):
            logging.debug("Not diverging: %f" % no_diverge_prob)
            prob += no_diverge_prob
            assignment, p = self.sample_assignment(node=node.children[idx],
                                                   constraints=constraints,
                                                   points=points,
                                                   index=index + (idx,),
                                                   state=state)
            return assignment, prob + p
        else:
            sampled_time, _ = df.sample(node_time, choice_time, counts[idx])
            diverge_prob = df.log_pdf(sampled_time, node_time, counts[idx])
            logging.debug("Diverging at %f: %f" % (sampled_time, diverge_prob))
            prob += diverge_prob
            state['time'] = sampled_time
            return (index + (idx,), state), prob

    def log_prob_assignment(self, assignment, node=None):
        node = node or self.root


        (idx, state) = assignment
        time = state['time']
        assert idx is not ()

        df = self.df

        first, rest = idx[0], idx[1:]

        counts = [c.leaf_count() for c in node.children]
        total = float(sum(counts))
        prob = np.log(counts[first]) - np.log(total)
        logging.debug("Branching prob: %f" % prob)

        node_time = node.get_state('time')

        if len(idx) == 1:
            diverge_prob = df.log_pdf(node_time, time, counts[first])
            logging.debug("Diverging at %f: %f" % (time, diverge_prob))
            return prob + diverge_prob

        choice = node.children[first]
        choice_time = choice.get_state('time')

        no_diverge_prob = (df.cumulative_divergence(node_time) - df.cumulative_divergence(choice_time)) / \
            counts[first]
        logging.debug("Not diverging: %f" % no_diverge_prob)

        return prob + no_diverge_prob + self.log_prob_assignment((rest, state), node=node.children[first])

    def assign_node(self, node, assignment):
        (idx, state) = assignment
        assignee = self.get_node(idx)
        assignee.attach(node)
        node.parent.state.update(state)

    def sample_latent(self):
        for node in self.dfs():
            if not node.is_leaf():
                lv = self.likelihood_model.sample_transition(node, node.parent)
                node.set_state('latent_value', lv)
                node.delete_cache("likelihood")

    def node_as_string(self, node):
        return str(node.get_state('time'))

    def write_node(self, cur_node, file):
        if not cur_node.is_leaf():
            l, r = cur_node.children
            l_id = self.write_node(l, file)
            r_id = self.write_node(r, file)
            self.node_ID += 1
            node_id = self.node_ID
        else:
            node_id = l_id = r_id = -1

        if cur_node.is_root():
            is_root = "1"
        else:
            is_root = "0"

        file.write(str(node_id)+" "+is_root+" "+str(l_id)+ " "+str(r_id)+" "+str(cur_node.get_state('time'))+"\n")

        return self.node_ID

    def write_tree_file(self, tree_id, node_numbers,  file_name):
        self.node_ID = 0
        file = open(file_name, "w")
        file.write(str(tree_id)+ " "+str(2*node_numbers - 1)+"\n")
        self.write_node(self.root, file)
        file.close()

    def get_parameters(self):
        return {
            "df",
            "likelihood_model",
        }
