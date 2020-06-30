from math import exp, sqrt
import numpy as np

class Generate(object):
    """
    c is divergence parameter
    sigma2 is browian motion parameters
    p is dimension of data
    n is # of samples
    """
    def __init__(self, c, sigma2=1):
        self.c = c
        self.sigma2 = sigma2

    # inverse of cumulative divergence function 
    # GENERATE A DATA SET FROM A DIFFUSION TREE PRIOR.  Each of the n data items 
    # will consist of p variables.  The function Ainv is the inverse of the 
    # cumulative divergence function.  The value returned is a list containing
    # the data and the tree.  Note that the sigma parameter of the Dirichlet
    # diffusion tree is always taken to be one.s
    def gen(self, n, p):
        all_data = np.zeros((n, p))
        all_data[0] = np.random.normal(size=(1, p))
        #tree = list(time=1, count=1, data=data[1,]) 
        tree = dict({"time":1, "count":1, "data":all_data[0]})
 
        if n == 1: return all_data, tree
        for i in range(1, n): 
            start_pos = np.zeros((1, p))
            res = self.add_one(n, p, tree, start_pos)
            all_data[i] = res['data']
            tree = res['tree']
        #return data, tree
        return all_data, tree


    # ADD ONE MORE DATA ITEM TO A TREE.  Generates one additional data item,
    # consisting of p variables, to add to the data set whose underlying 
    # tree is passed as an argument.  The function Ainv is the inverse of the
    # cumulative divergence function.  The value returned is a list consisting 
    # of the new data vector and the updated tree.


    # The start.time, start.A, and start.pos arguments default to zero for the 
    # outside use of this function.  When this function calls itself recursively,
    # they are set to the starting time of the diffusion for the current subtree,
    # to the cumulative divergence function evaluated at that time, and to the
    # position in data space reached at that time.
    def add_one(self, n, p, tree, start_pos, start_time=0.0, start_A=0.0):
        # generate probability using exponential distribution
        A = start_A + np.random.exponential(scale=tree['count'], size=1)#generate exponential random variable
        time = self.Ainv1(A)

        # some corner case to jump out of scripts
        assert time >= start_time
        assert time <= 1

        if time == start_time:
            tree = dict({"time":time, "A":A, "count":tree["count"]+1,"data":start_pos,\
            "left":tree,"right":dict({"time":1, "count":1, "data":np.random.normal(start_pos, sqrt(1-time), (1, p))})})
            return dict({"data":tree['right']['data'], "tree":tree})
        
        # case when divergence occured before the segment
        if time < tree['time']:
            precision = 1.0/(time-start_time) + 1.0/(tree['time']-time)
            pos = np.random.normal((start_pos/(time-start_time) \
                + tree['data']/(tree['time']-time)) / precision, 1.0/sqrt(precision), (1,p))

            tree = dict({"time":time, "A":A, "count":tree["count"] + 1,"data":pos,\
            "left":tree,"right":dict({"time":1, "count":1, "data":np.random.normal(pos, sqrt(1.0-time), (1,p))})})
            return dict({"data":tree['right']['data'], "tree": tree})
        
        # case when divengence occured when time==1
        if tree['time'] == 1:
            tree['count'] += 1
            return dict({"data": tree['data'], "tree": tree})         
        # debugging 
        assert tree['left']['count'] + tree['right']['count'] == tree['count']

        # Case where no divergence occurs, and so we must select a branch to go down.
        if (np.random.uniform(size=1) < tree['left']['count']/float(tree['count'])):
            # the data will go down to the left subtree
            res = self.add_one(n, p, tree['left'], tree['data'], tree['time'], tree['A'])
            # add tree into the left subtree
            tree['left'] = res['tree']
            tree['count'] += 1
        else:
            res = self.add_one(n, p, tree['right'], tree['data'], tree['time'], tree['A'])
            # add tree into the right subtree
            tree['right'] = res['tree']
            tree['count'] += 1

        return dict({"data": res['data'], "tree": res['tree']})


    def Ainv1(self, e):
        c = self.c
        return 1 - exp(-float(e)/c) 

    def Ainv2(self, e):
        c = self.c
        return 1 - (c / float(e))


