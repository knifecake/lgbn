from typing import Any, Optional, Dict
import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd


class CPD:
    '''
    A conditional probability distribution for a node.which also references the
    node's parents.s

    In a `BayesianNetwork` the probability distributions of the nodes are
    specified via CPDs or Conditional Probability distributions. These classes
    represent the probability distributions of the random variables in the nodes
    of a Bayesian network, which in general are conditioned on the parent nodes.

    .. note::

        This is a base class which cannot actually be used.

    See Also
    --------

    LinearGaussianCPD
        A conditional probability distribution for linear Gaussian Bayesian
        networks.
    '''

    node: Any
    parents: tuple[Any]

    def __init__(self, node: Any, parents: Optional[tuple[Any]]=None) -> None:
        '''
        Create a conditional probability distribution
        
        Parameters
        ----------
        node : Any
            The identifier of the node which has this CPD.
        parents : tuple[Any]
            The identifiers of the parent nodes upon which this distribution
            depends.
        '''
        self.node = node
        self.parents = parents

        if self.parents is None:
            self.parents = tuple()
        else:
            self.parents = tuple(self.parents)

    def to_dict(self) -> Dict[str, Any]:
        '''Serializes this conditional probability distribution into a dict.'''
        return {
            'node': self.node,
            'parents': self.parents
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        '''Loads this conditional probability distribution from a dict.'''
        return cls(**data)

class LinearGaussianCPD(CPD):
    '''
    A linear Gaussian conditional probability distribution.


    A linear Gaussian conditional probability distribution is normal
    distribution where the mean is a linear combination of the means of the
    parent nodes plus a bias term, i.e. if this node (X) has `parents`
    :math:`U_1, \ldots, U_k` then
    
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
        '''
        Create a linear Gaussian conditional probability distribution.
        
        Parameters
        ----------
        node : Any
            The identifier of the node which has this CPD.
        mean : float
            The constant term or bias term in the mean of this distribution.
            Defaults to 0.
        var : float
            The variance of this distribution (must be strictly positive).
            Defaults to 1.
        parents : tuple[Any]
            The identifiers of the parent nodes upon which this distribution
            depends.
        weights : tuple[float]
            The weights of the dependencies of this distribution to the values
            taken by the parent nodes. Defaults to a tuple of 1s and of the same
            length as parents.

        Raises
        ------
        ValueError
            If the variance `var` is not positive or the `weights` and `parents`
            don't have the same length.

        Examples
        --------
        >>> LinearGaussianCPD('A', mean=0, var=1)
        <LinearGaussianCPD: A ~ N(0.000 + , 1.000)>

        Mean and variance default to 0 and 1 respectively:

        >>> LinearGaussianCPD('A', parents=('B', 'C'), weights=(1, 1))
        <LinearGaussianCPD: A ~ N(0.000 + 1.000*B + 1.000*C, 1.000)>

        Parents can be any iterable and weights default to a vector of 1s the
        same length as parents, so the following example is equivalent to the
        previous one:

        >>> LinearGaussianCPD('A', parents='BC')
        <LinearGaussianCPD: A ~ N(0.000 + 1.000*B + 1.000*C, 1.000)>
    
    '''
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

    
    def mle(self, data: pd.DataFrame):
        '''
        Find maximum likelihood estimate for `mean`, `variance` and `weights`
        given the `data` and the dependencies on the `parents`.


        Parameters
        ----------
        data : pandas.DataFrame
               A DataFrame with one row per observation and one column per
               variable.
        
        Returns
        -------
        LinearGaussianCPD
            A new conditional probability distribution where the parameters are
            set to the ML estimates.

        Notes
        -----
        Maximum likelihood estimation of parameters is computed using the
        *sufficient statistics*  approach described in section 17.2.4 of [1]_.
        
        References
        ----------

        .. [1] D. Koller and N. Friedman, Probabilistic graphical models:
               principles and techniques. Cambridge, MA: MIT Press, 2009.

        '''
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

    def __repr__(self) -> str:
        return f'<LinearGaussianCPD: {str(self)}>'

class BayesianNetwork(nx.DiGraph):
    '''
    A Bayesian Network.

    A Bayesian network is a directed acyclic graph where each node is a random
    variable with a distribution conditional on the parent nodes.
    '''

    
    cpds = Dict[Any, CPD]
    '''
    A dictionary mapping node identifiers to Conditional Probability
    Distributions.
    '''

    cpd_class = CPD
    '''
    The class used to instantiated CPDs when loading from a dict via
    `.from_dict(data)`. 
    '''

    def __init__(self):
        self.cpds = {}
        super().__init__()

    def add_cpd(self, cpd: CPD) -> None:
        '''
        Add a conditional probability distribution to the network.
        
        .. note:: 
            Also adds the node and edges to the underlying `networkx.DiGraph`. If
            there already is a conditional probability distribution it is replaced
            and new edges are added, but previous edges not present in the new
            distribution will not be removed.

        Parameters
        ----------
        cpd : CPD
            A conditional probability distribution

        '''
        self.cpds[cpd.node] = cpd

        # add node and edges from parents (if not already present)
        self.add_node(cpd.node)
        self.add_edges_from([(p, cpd.node) for p in cpd.parents])

    def apply_op(self, op):
        '''
        Modify the network according to the given operation.
        
        An operation is an edge together with an action (add edge, remove edge,
        flip edge).
        
        Parameters
        ----------
        op
            A tuple ``(action, (u, v))`` where action is one of +, - or F and (u,
            v) is an edge.

        Raises
        ------
        NotImplementedError
            If the given action is not supported.
        
        ValueError
            If the edge is not in the network and operation requires editing it.
            (This is actually raised by `networkx.DiGraph.remove_edge`.)
        '''
        action, edge = op
        if action == '+':
            self.add_edge(*edge)
        elif action == '-':
            self.remove_edge(*edge)
        elif action == 'F':
            self.remove_edge(*edge)
            self.add_edge(*reversed(edge))
        else:
            raise NotImplementedError(f'Operation type {action} is not implemented')

    def update_cpds_from_structure(self) -> None:
        '''
        Updates the `parents` attribute in each CPD to match the current graph
        structure.

        .. tip:: Use this method after updating the network structure (e.g. via learning).
        '''
        for node in self.nodes:
            parents = tuple(self.predecessors(node))
            self.cpds[node] = self.cpd_class(node, parents=parents)

    def to_dict(self):
        '''
        Serializes the Bayesian network into a list of dictionaries.

        Each dictionary is the result of serializing a CPD via `CPD.to_dict`.
        '''
        return [self.cpds[n].to_dict() for n in nx.topological_sort(self)]

    @classmethod
    def from_dict(cls, data):
        '''
        Load Bayesian network from a dict.

        Parameters
        ----------
        data : Sequence[Dict[str, Any]]
            A list of dictionaries corresponding to CPDs. See `CPD.to_dict` for
            more information on the format.

        See Also
        --------
        to_dict, CPD.from_dict
        
        Notes
        -----
        This method expects the data to be sorted by node in topological order,
        so that a node always comes before its children. This is the way
        `to_dict` generates dictionaries.

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
    parameters of which can be obtained via the `to_joint_gaussian` method.
    '''
    cpd_class = LinearGaussianCPD

    def to_joint_gaussian(self):
        '''
        Get the equivalent multivariate Gaussian distribution to this Bayesian
        network.

        Returns a `scipy.stats.multivariate_normal` frozen random variable which
        has attributes `mean` (vector of means) and `cov` (covariance matrix).
        
        Returns
        -------
        scipy.stats.multivariate_normal
            A frozen random variable.

        Notes
        -----

        Linear Gaussian Bayesian networks have a joint probability distribution
        that is also normal, and thus this method is well defined. See p. 252 of
        [1]_ and pp. 370-371 of [2]_.

        References
        ----------

        .. [2] C. M. Bishop, Pattern recognition and machine learning. New York:
                  Springer, 2006.
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

