from typing import Any, Optional, Dict
import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal


class CPD:
    '''A conditional probability distribution for a node which also references
    the node's parents.'''
    node: Any
    parents: tuple[Any]

    def __init__(self, node: Any, parents: Optional[tuple[Any]]=None):
        self.node = node
        self.parents = parents

        if self.parents is None:
            self.parents = tuple()
        else:
            self.parents = tuple(self.parents)

    def to_dict(self):
        '''Serializes this conditional probability distribution into a dict.'''
        return {
            'node': self.node,
            'parents': self.parents
        }

    @classmethod
    def from_dict(cls, data):
        '''Loads this conditional probability distribution from a dict.'''
        return cls(**data)

class LinearGaussianCPD(CPD):
    '''
    A linear Gaussian conditional probability distribution where the mean is a
    linear combination of the means of the parent nodes plus a bias term, i.e.
    if this node (X) has `parents` :math:`U_1, \ldots, U_k` then
    
    .. math::

        p(X) = N(X \mid \mu + w_1 \mu_{U_1} + \ldots + w_k \mu_{U_k}, \sigma^2),
        
    where :math:`\mu` is specified via the `mean` parameter, :math:`\sigma^2`
    via the `var` parameter and :math:`w_1, \ldots, w_k` via the `weights`
    parameter.
    '''

    node: Any
    parents: tuple[Any]

    mean: float
    var: float
    weights: tuple[float]

    def __init__(self, node: Any, mean: Optional[float]=0, var: Optional[float]=1, parents:Optional[tuple[Any]]=None, weights: Optional[tuple[float]]=None):
        self.node = node
        self.mean = mean
        self.var = var
        self.parents = parents
        self.weights = weights

        # make sure parents and weights are tuples for caching to work
        if self.parents is None:
            self.parents = tuple()
        else:
            self.parents = tuple(self.parents)

        if self.weights is None:
            self.weights = tuple([1] * len(self.parents))
        else:
            self.weights = tuple(self.weights)
            if len(self.parents) != len(self.weights):
                raise ValueError('parents and weights must have the same length')

        if self.var <= 0:
            raise ValueError('variance must be positive')

    
    def mle(self, data):
        '''Find maximum likelihood estimate for `mean`, `variance` and `weights` given
        the `data` and the dependencies on the `parents`. Estimate is returned as a new LinearGaussianCPD.

        `data` must be a `pandas.DataFrame` with one row per observation and one column
        per variable.
        
        See section 17.2.4 of Koller & Friedman.'''
        M = len(data)
        k = len(self.parents)
        x_sum = data[self.node].sum()
        u_sums = data[list(self.parents)].sum().to_numpy()
        xu_sums = [(data[self.node] * data[p]).sum() for p in self.parents]
        uu_sums = [[(data[ui] * data[uj]).sum() for uj in self.parents]
                   for ui in self.parents]

        # solve A*beta = b
        A = np.block(
            [[np.reshape([M], (1, 1)),
              np.reshape(u_sums, (1, k))],
             [np.reshape(u_sums, (k, 1)),
              np.reshape(uu_sums, (k, k))]])
        b = [x_sum] + xu_sums
        beta = np.linalg.solve(A, b)

        # extract parameters
        mean, weights = beta[0], beta[1:]
        x_var = data[self.node].var()
        cov_d = data[list(self.parents)].cov()
        var = x_var - sum([
            sum([
                weights[i] * weights[j] * cov_d[pi][pj]
                for j, pj in enumerate(self.parents)
            ]) for i, pi in enumerate(self.parents)
        ])
        return LinearGaussianCPD(node=self.node,
                                 mean=mean,
                                 var=var,
                                 parents=self.parents,
                                 weights=weights)

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'mean': self.mean,
            'var': self.var,
            'weights': self.weights
        })
        return data

    def __str__(self):
        cond = ' + '.join(
            [f'{w:.3f}*{p}' for w, p in zip(self.weights, self.parents)])
        return f'{self.node} ~ N({self.mean:.3f} + {cond}, {self.var:.3f})'

class BayesianNetwork(nx.DiGraph):
    '''
    A Bayesian Network.
    '''

    cpds = Dict[Any, CPD]
    cpd_class = CPD

    def __init__(self):
        self.cpds = {}
        super().__init__()

    def add_cpd(self, cpd: CPD):
        '''Add a conditional probability distribution to the network. Also adds
        the node and edges to the underlying DiGraph.'''
        self.cpds[cpd.node] = cpd

        # add node and edges from parents (if not already present)
        self.add_node(cpd.node)
        self.add_edges_from([(p, cpd.node) for p in cpd.parents])

    def apply_op(self, op):
        action, edge = op
        if action == '+':
            self.add_edge(*edge)
        elif action == '-':
            self.remove_edge(*edge)
        elif action == 'F':
            self.remove_edge(*edge)
            self.add_edge(*reversed(edge))
        else:
            raise NotImplemented(f'Operation type {action} is not implemented')

    def update_cpds_from_structure(self):
        '''
        Updates the `parents` attribute in each CPD to match the current graph
        structure.
        '''
        for node in self.nodes:
            parents = tuple(self.predecessors(node))
            self.cpds[node] = self.cpd_class(node, parents=parents)

    def to_dict(self):
        '''
        Serializes the Bayesian network into a dictionary.
        '''
        return [self.cpds[n].to_dict() for n in nx.topological_sort(self)]

    @classmethod
    def from_dict(cls, data):
        '''
        Load Bayesian network from a dict.
        
        Note: this method expects the data to be sorted by node in topological
        order, so that a node always comes before its children. This is the way
        BayesianNetwork.to_dict() generates dictionaries.
        '''
        net = cls()
        for cpd_data in data:
            net.add_cpd(cls.cpd_class.from_dict(cpd_data))

        return net

    def __str__(self):
        return '\n'.join([f'{node} -> {list(children.keys())}' for node, children in self.adjacency()]) \
            + '\n' + '\n'.join([str(cpd) for cpd in self.cpds.values()])


class LinearGaussianBayesianNetwork(BayesianNetwork):
    '''
    A Bayesian network where every node has a Gaussian distribution, where the
    mean of each node is a linear combination of its parents plus a bias factor
    and the standard deviations of the nodes are independent.

    The joint distribution of these networks also a Gaussian distribution, the
    parameters of which can be obtained via the `to_joint_gaussian()` method.
    '''

    def to_joint_gaussian(self):
        '''
        Find the equivalent multivariate Gaussian distribution to this Bayesian
        network, useful for sampling.

        Returns a `scipy.stats.multivariate_normal` frozen random variable which
        has attributes `mean` (vector of means) and `cov` (covariance matrix).
        
        See Bishop PRML pp. 370-371. K&F p. 252.
        '''

        variables = list(nx.topological_sort(self))
        mean = np.zeros(len(variables))
        cov = np.zeros((len(variables), len(variables)))

        for i, x in enumerate(variables):
            cpd = self.cpds[x]

            # calculate the mean vector (Bishop eq. 8.15)
            # mean[variables.index(p)] is guaranteed to exists bc we
            # are iterating with the topological sort
            mean[i] = cpd.mean + sum([
                w * mean[variables.index(p)]
                for w, p in zip(cpd.weights, cpd.parents)
            ])

            # first step in calculating the cov matrix (KF 7.2, example 7.3)
            cov[i][i] = cpd.var + sum([
                w * w * cov[variables.index(p)][variables.index(p)]
                for w, p in zip(cpd.weights, cpd.parents)
            ])

        # second step to calculate the covariance matrix (KF 7.2, example 7.3)
        # adapted from pgmpy/models/LinearGaussianBayesianNetwork.to_joint_gaussian
        for i in range(len(variables)):
            for j, xj in enumerate(variables):
                if cov[j][i] != 0:
                    # make the cov matrix symmetric
                    cov[i][j] = cov[j][i]
                else:
                    cpd = self.cpds[xj]
                    cov[i][j] = sum([
                        w * cov[i][variables.index(p)]
                        for w, p in zip(cpd.weights, cpd.parents)
                    ])

        return multivariate_normal(mean=mean, cov=cov, allow_singular=True)
