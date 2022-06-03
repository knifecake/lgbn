from functools import lru_cache

import numpy as np


class BaseScore:
    '''
    The base class for network scores.
    
    A score is a class that takes an argument `data` on initialization and that
    implements a `score(network)` method that when given a network returns a
    real number.
    
    This base class also stubs the implementation for a decomposable score,
    where the `.score(network)` function can be obtained by summing the
    `.score_fam(node, parent)` method over the nodes in a network.
    '''
    def __init__(self, data) -> None:
        self.data = data

    def score_fam(self, node, parents: tuple):
        raise NotImplementedError

    def score(self, net):
        '''
        A default implementation for decomposable scores where the score of a
        network is the sum of the scores of each family (i.e. the set of a node
        and its parents).
        '''
        return np.sum(self.score_fam(node, tuple(net.predecessors(node))) for node in net.nodes())

class LogLikScore(BaseScore):
    '''
    The LogLik score of a network is the (natural) logarithm of the maximum
    likelihood of the data given the network as estimated by net.mle().

    This score is decomposable and thus takes advantage of caching.
    '''

    @lru_cache(maxsize=1024)
    def score_fam(self, node, parents):
        raise NotImplementedError


class BICScore(BaseScore):
    '''
    The Bayesian Information Criterion score of a network is the LogLikScore of
    that network minus a regularization penalty proportional to the size of the
    data and the complexity of the network.

    This score is decomposable and thus takes advantage of caching.
    '''
    def __init__(self, data):
        self.data = data
        self.loglik_score = LogLikScore(data)

    @lru_cache(maxsize=1024)
    def score_fam(self, node, parents: tuple):
        log_lik = self.loglik_score.score_fam(node, parents)
        return log_lik - .5 * np.log(len(self.data)) * (len(parents) + 2)

    
class AIC(BaseScore):
    '''
    The Akaike Information Criterio (AIC) score of a network is the LogLikScore
    of that network minus a regularization constant proportional to the
    complexity of the network.

    This score is decomposable and thus takes advantage of caching.
    '''

    def __init__(self, data):
        self.data = data
        self.loglik_score = LogLikScore(data)

    @lru_cache(maxsize=1024)
    def score_fam(self, node, parents: tuple):
        log_lik = self.loglik_score.score_fam(node, parents)
        return log_lik - .5 * (len(parents) + 2)
