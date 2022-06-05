from functools import cached_property
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lgbn.models import (BayesianNetwork, LinearGaussianBayesianNetwork,
                         LinearGaussianCPD)
from lgbn.scores import BaseScore, BICScore, LogLikScore


class ScoreSearchEstimator(BaseEstimator):
    '''
    A structure estimator using score-based search.
    
    .. note:: This class is just a general interface, it cannot actually be used.
    
    See Also
    --------
    K2Search, GreedyHillClimbing, GreedyEquivalentSearch
    '''


    model: BayesianNetwork = None
    '''
    The model resulting from the estimation.
    '''

    eps: float = 1e-9
    '''
    Tolerance for equality testing of numeric values. Two values a and b are
    equal if ``abs(a - b) < eps``.
    '''
    
    _score: BaseScore
    _data: pd.DataFrame = None

    def __init__(self, score=None, eps=1e-9):
        '''
        Create a ScoreSearchEstimator
        
        Parameters
        ----------
        
        score : str or BaseScore
            An instance of a Score, i.e. an object implementing a score(network)
            method.
            
            You can use `loglik` and `bic` as shortcuts for passing instances of
            LogLikScore and BICScore, respectively.

        eps : float

            Tolerance for number equality, i.e. a and b are considered equal if
            ``abs(a - b) < eps``. This is used to stop iteration in iterative
            search algorithms.
            
        Notes
        -----
        Some estimators require decomposable scores, i.e. those which also
        implement the score_fam method.
        '''
        self._set_score(score)

        self.eps = eps

        if self.eps <= 0:
            raise ValueError('epsilon must be positive')

    def _set_score(self, score):
        if isinstance(score, str):
            if score == 'loglik':
                self.score_class = LogLikScore
            elif score == 'bic':
                self.score_class = BICScore
            else:
                raise ValueError('Score %s unknown' % score)
        else:
            self._score = score

        if hasattr(self, 'score'):
            del self.score

    @cached_property
    def score(self):
        '''
        Score instance to use for scoring networks in the search procedure.
        '''
        if hasattr(self, 'score_class'):
            # if self._data is None:
            #     raise UserWarning('accessing the score before calling .fit(data) will yield an unusable score with no data.')
            self._score = self.score_class(self._data)

        return self._score


    def fit(self, data):
        '''
        Fit the model to the given data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame with one row per observation and one column per
            variable. Column names will be used for node identifiers in the
            resulting model.
        
        Returns
        -------
        A fitted estimator (self).
        
        '''
        self._data = data

        # invalidate score cache
        if hasattr(self, 'score'):
            del self.score

        self.model = self.search()
        self.model.update_cpds_from_structure()
        return self # must return an estimator  

    def search(self) -> BayesianNetwork:
        '''
        Search the space of posible models for the one that maximizes the score
        of this estimator.
        '''
        raise NotImplementedError
    
    def get_params(self, deep=True):
        return {'score': self.score, 'eps': self.eps}

    def set_params(self, **kwargs):
        if 'score' in kwargs:
            self._set_score(kwargs['score'])
        
        if 'eps' in kwargs:
            self.eps = kwargs['eps']

            if self.eps <= 0:
                raise ValueError('epsilon must be positive')

        return self # must return an estimator

class K2Search(ScoreSearchEstimator):
    '''
    K2 structure learning algorithm.
    
    The K2 algorithm learns the structure of a Bayesian network that maximizes
    the given score. The search procedure is guided by a given topological
    ordering of the network. In that ordering, if node x comes before node y,
    then node y can never be a parent of node x. This vastly reduces the search
    space resulting in a significant speedup, even without using caching.

    See [1]_ for a detailed description of the K2 algorithm.
    
    .. note::
        This implementation requires a decomposable score, although
        there exist other implementations that work with non-decomposable scores.

    References
    ----------

    .. [1] G. F. Cooper and E. Herskovits, “A Bayesian method for the induction
              of probabilistic networks from data,” Mach Learn, vol. 9, no. 4,
              pp. 309–347, Oct. 1992, doi: 10.1007/BF00994110. 

    '''

    def __init__(self, score=None, ordering=None, eps=1e-9):
        '''
        Create a K2Search estimator.
        
        Parameters
        ----------
        score : str or BaseScore
            An instance of a Score, i.e. an object implementing a score(network)
            method.
            
            You can use `loglik` and `bic` as shortcuts for passing instances of
            LogLikScore and BICScore, respectively.
        
        ordering : Sequence[Any]
            A list-like object of node identifiers to use as the topological
            ordering.

        eps : float

            Tolerance for number equality, i.e. a and b are considered equal if
            ``abs(a - b) < eps``. This is used to stop iteration in iterative
            search algorithms.
        
        '''
        super().__init__(score=score, eps=eps)

        self.ordering = ordering
        if self.ordering is None:
            raise ValueError('Ordering must be defined for the K2Search estimator')

    def search(self):
        # initialize disconnected DAG
        dag = LinearGaussianBayesianNetwork()
        for node in self.ordering:
            dag.add_cpd(LinearGaussianCPD(node))

        current_score = self.score.score(dag)
        for node in self.ordering:
            score = self.score.score_fam(node, tuple())
            best_parent, best_delta = self._best_add_parent(dag, node)
            while best_delta is not None and best_delta > score:
                score = best_delta

                current_score += best_delta
                dag.add_edge(best_parent, node)
                best_parent, best_delta = self._best_add_parent(dag, node)
        return dag

    def _best_add_parent(self, dag, node):
        idx = self.ordering.index(node)
        possible_parents = self.ordering[:idx]
        current_parents = list(dag.predecessors(node))
        possible_parents = list(
            filter(lambda n: n not in current_parents, possible_parents))
        parent_score_pairs = [
            (p, self.score.score_fam(node, tuple(current_parents + [p])))
            for p in possible_parents
        ]

        return max(parent_score_pairs,
                   key=lambda t: t[1],
                   default=(None, None))

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['ordering'] = self.ordering
        return params

    def set_params(self, **kwargs):
        if 'ordering' in kwargs:
            self.ordering = kwargs['ordering']
        
        return super().set_params(**kwargs)

    

class GreedyHillClimbing(ScoreSearchEstimator):
    '''
    Greedy Hill Climbing structure search algorithm.

    The Greedy Hill Climbing algorithm learns the structure of a Bayesian
    network that maximizes the given score. The search procedure starts with an
    initial network, which defaults to a fully disconnected network. Edges are
    added, removed or have their direction reversed one at a time until no more
    modifications increase the overall score of the network. This algorithm is
    reasonably fast when used with a decomposable score which can be cached.

    See p. 40 in [2]_ for a detailed description of the Greedy Hill Climbing
    algorithm. The source refers to Greedy Hill Climbing as Max-Min Hill
    Climbing.
    
    .. note::
        This implementation requires a decomposable score, although
        there exist other implementations that work with non-decomposable scores.

    References
    ----------

    .. [2] I. Tsamardinos, L. E. Brown, and C. F. Aliferis, “The max-min
           hill-climbing Bayesian network structure learning algorithm,” Mach
           Learn, vol. 65, no. 1, pp. 31–78, Oct. 2006, doi:
           10.1007/s10994-006-6889-7. 

    '''
    def __init__(self, score=None, start_net=None, max_iter=1_000_000, eps=1e-9, random_state=None):
        '''
        Create a GreedyHillClimbing estimator.
        
        Parameters
        ----------
        score : str or BaseScore
            An instance of a Score, i.e. an object implementing a score(network)
            method.
            
            You can use `loglik` and `bic` as shortcuts for passing instances of
            LogLikScore and BICScore, respectively.
        
        start_net : Optional[BayesianNetwork]
            A starting Bayesian network, defaults to a fully disconnected
            network.

        max_iter : int
            Maximum number of iterations to perform, defaults to 1 million.

        eps : float
            Tolerance for number equality, i.e. a and b are considered equal if
            ``abs(a - b) < eps``. This is used to stop iteration in iterative
            search algorithms.
        
        random_state : None, int or numpy.random.Generator
            Controls the source of randomness for the algorithm. If None or an
            int is passed, then a new source of randomness is obtained from
            ``numpy.random.default_rng(random_state)``. Otherwise you can pass a
            Generator that you have created yourself.
        '''
        super().__init__(score=score, eps=eps)

        self.start_net = start_net
        self.max_iter = max_iter
        self._set_random_state(random_state)
        

    def _set_random_state(self, random_state):
        if isinstance(random_state, np.random.Generator):
            self.random_state = random_state
        else:
            self.random_state = np.random.default_rng(random_state)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['start_net'] = self.start_net
        params['max_iter'] = self.max_iter
        params['random_state'] = self.random_state
        return params

    def set_params(self, **kwargs):
        if 'start_net' in kwargs:
            self.start_net = kwargs['start_net']
        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']
        if 'random_state' in kwargs:
            self._set_random_state(kwargs['random_state'])

        return super().set_params(**kwargs)

    def search(self):
        dag = self.start_net
        if dag is None:
            # create an empty dag
            dag = LinearGaussianBayesianNetwork()
            for v in self.score.data.columns:
                dag.add_cpd(LinearGaussianCPD(v))

        current_score = self.score.score(dag)
        for _ in range(self.max_iter):
            legal_ops = self._get_legal_operations(dag)
            self.random_state.shuffle(legal_ops)
            op_deltas = [self._get_op_delta(op, dag) for op in legal_ops]

            op_delta_pairs = list(zip(legal_ops, op_deltas))

            best_op, best_delta = max(op_delta_pairs,
                                      key=lambda t: t[1],
                                      default=(None, None))

            if best_delta > self.eps:
                current_score += best_delta
                dag.apply_op(best_op)
            else:
                break

        return dag

    def _get_legal_operations(self, dag):
        # adapted from pgmpy/estimators/HillClimbSearch

        # Step 1: adding edges
        possible_edges = permutations(dag.nodes(), 2)
        potential_new_edges = set(possible_edges) - set(dag.edges())

        # filter out edges that would introduce a cycle
        new_edges = list(
            filter(lambda e: not nx.has_path(dag, e[1], e[0]),
                   potential_new_edges))

        ops = [('+', edge) for edge in new_edges]

        # Step 2: removing edges
        ops += [('-', edge) for edge in dag.edges()]

        # Step 3: flipping edges
        possible_edges = dag.edges()
        edges_to_flip = list(
            filter(
                lambda e: not any(
                    map(lambda p: len(p) > 2,
                        nx.all_simple_paths(dag, e[0], e[1]))),
                possible_edges))
        ops += [('F', edge) for edge in edges_to_flip]

        return ops

    def _get_op_delta(self, op, net):
        # adapted from pgmpy/estimators/HillClimbSearch
        code, (X, Y) = op
        score_delta = 0
        if code == '+':
            old_parents = list(net.predecessors(Y))
            new_parents = old_parents + [X]
            score_delta = (self.score.score_fam(Y, tuple(new_parents)) -
                           self.score.score_fam(Y, tuple(old_parents)))

        elif code == '-':
            old_parents = list(net.predecessors(Y))
            new_parents = old_parents.copy()
            new_parents.remove(X)
            score_delta = (self.score.score_fam(Y, tuple(new_parents)) -
                           self.score.score_fam(Y, tuple(old_parents)))

        elif code == 'F':
            old_X_parents = list(net.predecessors(X))
            old_Y_parents = list(net.predecessors(Y))
            new_X_parents = old_X_parents + [Y]
            new_Y_parents = old_Y_parents.copy()
            new_Y_parents.remove(X)
            score_delta = (self.score.score_fam(X, tuple(new_X_parents)) +
                           self.score.score_fam(Y, tuple(new_Y_parents)) -
                           self.score.score_fam(X, tuple(old_X_parents)) -
                           self.score.score_fam(Y, tuple(old_Y_parents)))
        else:
            raise NotImplementedError

        return score_delta


class GreedyEquivalentSearch(ScoreSearchEstimator):
    '''
    Greedy Equivalent Search structure learning algorithm.

    The Greedy Equivalent algorithm learns the structure of a Bayesian network
    that maximizes the given score. The search procedure starts with an empty
    graph. Edges are added until no more increase the score and then removed
    until no further operation increases the score. Equality of operations and
    thus networks is defined by equivalence classes. An equivalence class
    contains all networks which have the same edges regardless of orientation.
    This algorithm is reasonably fast when used with a decomposable score which
    can be cached.

    See [3]_ for a detailed description of the Greedy Equivalent Search
    algorithm.
    
    .. note::
        This implementation requires a decomposable score, although
        there exist other implementations that work with non-decomposable scores.

    References
    ----------

    .. [3] D. M. Chickering, “Optimal Structure Identiﬁcation With Greedy
              Search,” Journal of Machine Learning Research, vol. 3, no. Nov
              2002, p. 48, Nov. 2002. 

    '''

    def __init__(self, score=None, max_iter=1_000_000, eps=1e-9):
        '''
        Create a GreedyEquivalentSearch estimator.
        
        Parameters
        ----------
        score : str or BaseScore
            An instance of a Score, i.e. an object implementing a score(network)
            method.
            
            You can use `loglik` and `bic` as shortcuts for passing instances of
            LogLikScore and BICScore, respectively.
        
        max_iter : int
            Maximum number of iterations to perform, defaults to 1 million.

        eps : float
            Tolerance for number equality, i.e. a and b are considered equal if
            ``abs(a - b) < eps``. This is used to stop iteration in iterative
            search algorithms.

        '''
        super().__init__(score=score, eps=eps)
        self.max_iter = max_iter
        
    def search(self):
        # create an empty dag
        dag = LinearGaussianBayesianNetwork()
        for v in self.score.data.columns:
            dag.add_cpd(LinearGaussianCPD(v))

        current_score = self.score.score(dag)

        while True:
            legal_ops = self._get_legal_add_operations(dag)
            op_deltas = [self._get_op_delta(op, dag) for op in legal_ops]
            op_delta_pairs = list(zip(legal_ops, op_deltas))

            best_op, best_delta = max(op_delta_pairs,
                                      key=lambda t: t[1],
                                      default=(None, None))

            if best_delta > self.eps:
                current_score += best_delta
                dag.apply_op(best_op)
            else:
                break # pragma: no cover

        while True:
            legal_ops = self._get_legal_remove_operations(dag)
            op_deltas = [self._get_op_delta(op, dag) for op in legal_ops]
            op_delta_pairs = list(zip(legal_ops, op_deltas))

            best_op, best_delta = max(op_delta_pairs,
                                      key=lambda t: t[1],
                                      default=(None, None))

            if best_delta > self.eps:
                current_score += best_delta
                dag.apply_op(best_op)
            else:
                break # pragma: no cover

        return dag

    def _get_legal_add_operations(self, dag):
        # adapted from pgmpy/estimators/HillClimbSearch

        # Step 1: adding edges
        possible_edges = permutations(dag.nodes(), 2)
        potential_new_edges = set(possible_edges) - set(dag.edges())

        # filter out edges that would introduce a cycle
        new_edges = list(
            filter(lambda e: not nx.has_path(dag, e[1], e[0]),
                   potential_new_edges))

        return [('+', edge) for edge in new_edges]

    def _get_legal_remove_operations(self, dag):
        # Step 2: removing edges
        return [('-', edge) for edge in dag.edges()]

    def _get_op_delta(self, op, net):
        # adapted from pgmpy/estimators/HillClimbSearch
        code, (X, Y) = op
        score_delta = 0
        if code == '+':
            old_parents = list(net.predecessors(Y))
            new_parents = old_parents + [X]
            score_delta = (self.score.score_fam(Y, tuple(new_parents)) -
                           self.score.score_fam(Y, tuple(old_parents)))

        elif code == '-':
            old_parents = list(net.predecessors(Y))
            new_parents = old_parents.copy()
            new_parents.remove(X)
            score_delta = (self.score.score_fam(Y, tuple(new_parents)) -
                           self.score.score_fam(Y, tuple(old_parents)))

        else:
            raise NotImplementedError

        return score_delta

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['max_iter'] = self.max_iter
        return params

    def set_params(self, **kwargs):
        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']

        return super().set_params(**kwargs)
