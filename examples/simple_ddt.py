import seaborn as sns
sns.set_style('white')
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from examples.generate_data import Generate
from trees.ddt import DirichletDiffusionTree, Inverse, GaussianLikelihoodModel



if __name__ == "__main__":
    D = 2
    N = 100
    X = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D), size=N).astype(np.float32)
    df = Inverse(c=1)
    simulator = Generate(c=1)
    num_nodes_per_tree = 10
    dim = 1
    data, _ = simulator.gen(num_nodes_per_tree, dim)


    lm = GaussianLikelihoodModel(sigma=np.eye(D) / 4.0, mu0=np.zeros(D), sigma0=np.eye(D))
    ddt = DirichletDiffusionTree(df=df,
                                 likelihood_model=lm)

    ddt.initialize_from_data(data)
    path_count, tree_prob, data_prob = ddt.calculate_node_likelihood(c=1)

    print("path_count = ", path_count)
    print("tree_prob = ", tree_prob)
    print("data_prob = ", data_prob)

    ddt.write_tree_file(tree_ID=1, node_numbers=num_nodes_per_tree, file_name="tree.txt")



