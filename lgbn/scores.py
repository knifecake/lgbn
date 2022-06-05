from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from lgbn.models import BayesianNetwork, LinearGaussianCPD


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
    
    data: pd.DataFrame
    '''
    A Pandas DataFrame with one row per observation and one column per variable.
    '''

    def __init__(self, data: pd.DataFrame) -> None:
        '''
        Create a BaseScore
        
        Parameters
        ----------
        data : pandas.DataFrame
            A Pandas DataFrame with one row per observation and one column per variable.
        '''
        self.data = data

    def score_fam(self, node: Any, parents: tuple[Any]):
        raise NotImplementedError

    def score(self, net: BayesianNetwork):
        '''
        A default implementation for decomposable scores where the score of a
        network is the sum of the scores of each family (i.e. the set of a node
        and its parents).
        '''
        return sum(self.score_fam(node, tuple(net.predecessors(node))) for node in net.nodes())

class LogLikScore(BaseScore):
    '''
    The LogLik score of a network is the (natural) logarithm of the maximum
    likelihood of the data given the network as estimated by net.mle().

    This score is decomposable and thus takes advantage of caching.

    See section 18.3.1 of [1]_ for a more in-depth discussion of the log
    likelihood score.
    '''

    @lru_cache(maxsize=1024)
    def score_fam(self, node, parents):
        # estimate parameters by maximum likelihood
        fam = LinearGaussianCPD(node, parents=parents).mle(self.data)
        
        # calculate likelihood
        parent_data = self.data[list(fam.parents)]
        weights = pd.Series(fam.weights, index=fam.parents, dtype=float)
        mean = fam.mean + parent_data.mul(weights, axis='columns').sum(axis=1)
        return norm.logpdf(self.data[node], loc=mean, scale=np.sqrt(fam.var)).sum()


class BICScore(BaseScore):
    '''
    The Bayesian Information Criterion score of a network is the LogLikScore of
    that network minus a regularization penalty proportional to the size of the
    data and the complexity of the network.

    This score is decomposable and thus takes advantage of caching.

    For a complete description of the BIC score see [1]_ section 18.3.5.

    References
    ----------

    .. [1] D. Koller and N. Friedman, Probabilistic graphical models: principles
           and techniques. Cambridge, MA: MIT Press, 2009. 

    '''
    def __init__(self, data):
        self.data = data
        self.loglik_score = LogLikScore(data)

    @lru_cache(maxsize=1024)
    def score_fam(self, node, parents: tuple):
        log_lik = self.loglik_score.score_fam(node, parents)
        return log_lik - .5 * np.log(len(self.data)) * (len(parents) + 2)
