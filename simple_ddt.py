import seaborn as sns
sns.set_style('white')
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import  os

from examples.generate_data import Generate
from trees.ddt.ddt import DirichletDiffusionTree
from trees.ddt.df import  Inverse
from trees.ddt.likelihood import GaussianLikelihoodModel



if __name__ == "__main__":


    simulator = Generate(c=1)
    num_nodes_per_tree = 10
    dim = 1
    data, _ = simulator.gen(num_nodes_per_tree, dim)

    df = Inverse(c=1, m =2)
    lm = GaussianLikelihoodModel(sigma=np.eye(dim) / 4.0, mu0=np.zeros(dim), sigma0=np.eye(dim) / 4.0)
    ddt = DirichletDiffusionTree(num_nodes_per_tree, df=df,likelihood_model=lm)

    ddt.initialize_from_data(data)
    tree_prob, data_prob = ddt.calculate_node_likelihood(c=1)

    # print("path_count = ", path_count)
    print("tree_prob = ", tree_prob)
    print("data_prob = ", data_prob)

    ddt.write_tree_file(node_numbers=num_nodes_per_tree, file_name=os.path.join("../","c-version", "tree.txt"))



