import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from trees.ddt.theanify import theanify, Theanifiable


"""
Class for divergence function c/(1-t)

Change divergence funciton for a(t) = c/((1-t)^2)
"""
class DivergenceFunction(Theanifiable):
    def __init__(self, **parameters):
        super(DivergenceFunction, self).__init__()
        #for param in self.get_parameters():
        #    assert param in parameters, 'Missing parameter %s' % param
        self.parameters = parameters
        self.compile()

    #def __getattr__(self, key):
    #    if key in self.parameters:
    #        return self.parameters[key]

    # def __init__(self, c):
    #    self.c = c

    @theanify(T.dscalar('t'), T.dscalar('c'))
    def log_divergence(self, t, c):
        return T.log(self.divergence(t, c))

    @theanify(T.dscalar('s'), T.dscalar('t'), T.dscalar('c'))
    def no_divergence(self, s, t, c):
        return T.exp(self.log_no_divergence(s, t, c))

    @theanify(T.dscalar('t1'), T.dscalar('t2'), T.dscalar('m'), T.dscalar('c'))
    def log_pdf(self, t1, t2, m, c):
        z = (self.cumulative_divergence(t1, c) - self.cumulative_divergence(t2, c))/m
        p = T.log(self.divergence(t2, c)) -  T.log(m)
        return z + p

class Inverse(DivergenceFunction):
    """
    Divergence function is c/(1-t)^2
    """
    @theanify(T.dscalar('t'), T.dscalar('c'))
    def divergence(self, t, c):
        return c / (1 - t)
        #return c / ((1 - t) ** 2)

    @theanify(T.dscalar('t'), T.dscalar('c'))
    def cumulative_divergence(self, t, c):
        return -c * T.log(1 - t)
        #return c / (1 - t)

    @theanify(T.dscalar('s'), T.dscalar('t'), T.dscalar('c'))
    def log_no_divergence(self, s, t, c):
        #return self.cumulative_divergence(s, c) - self.cumulative_divergence(t, c) - T.log(m)
         return self.cumulative_divergence(s, c) - self.cumulative_divergence(t, c)

    @theanify(T.dscalar('t'), T.dscalar('t1'), T.dscalar('m'), T.dscalar('c'))
    def cdf(self, t, t1, m, c):
        return (1 / (1 - t1)) ** (c / m) - ((1 - t) / (1 - t1)) ** (c / m)
        #return T.exp(c / (m * (1 - t1))) - T.exp(c / m * (t / (1 - t1)))

    @theanify(T.dscalar('t1'), T.dscalar('t2'), T.dscalar('m'), T.dscalar('c'))
    def sample(self, t1, t2, m, c):
        y = RandomStreams().uniform()
        lower = self.cdf(t1, t1, m, c)
        upper = self.cdf(t2, t1, m, c)
        y = lower + y * (upper - lower)
        t = 1-((1-t1)**(c/m)*((1-t1)**(-c/m)-y))**(m/c)
        #t = m * (1 - t1) / c * T.log(T.exp(c / m * 1/(1-t1)) - y)
        return t, self.log_pdf(t, t1, m, c)



    #def get_parameters(self):
    #    return {"c"}
