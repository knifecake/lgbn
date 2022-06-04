import random
from functools import cached_property
from itertools import permutations
from time import time

import networkx as nx
import pandas as pd
from sklearn.base import BaseEstimator

from lgbn.models import (BayesianNetwork, LinearGaussianBayesianNetwork,
                         LinearGaussianCPD)
from lgbn.scores import BaseScore, BICScore, LogLikScore


class ScoreSearchEstimator(BaseEstimator):
    model: BayesianNetwork = None
    eps: float = 1e-9
    
    _score: BaseScore
    _data: pd.DataFrame = None

    def __init__(self, score=None, eps=1e-9):
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
        '''Returns the score to use to fit the data to the model.'''
        if hasattr(self, 'score_class'):
            # if self._data is None:
            #     raise UserWarning('accessing the score before calling .fit(data) will yield an unusable score with no data.')
            self._score = self.score_class(self._data)

        return self._score


    def fit(self, data):
        self._data = data

        # invalidate score cache
        if hasattr(self, 'score'):
            del self.score

        self.model = self.search()
        self.model.update_cpds_from_structure()
        return self

    def search(self) -> BayesianNetwork:
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
    def __init__(self, score=None, ordering=None, eps=1e-9):
        super().__init__(score=score, eps=eps)

        self.ordering = ordering
        if self.ordering is None:
            raise ValueError('Ordering must be defined for the K2Search estimator')

    def search(self):
        # initialize empty DAG
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
    def __init__(self, score=None, start_net=None, max_iter=1_000_000, eps=1e-9):
        super().__init__(score=score, eps=eps)

        self.start_net = start_net
        self.max_iter = max_iter

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['start_net'] = self.start_net
        params['max_iter'] = self.max_iter

        return params

    def set_params(self, **kwargs):
        if 'start_net' in kwargs:
            self.start_net = kwargs['start_net']
        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']

        return super().set_params(**kwargs)

    def search(self):
        random.seed(time()) # TODO: handle randomness

        dag = self.start_net
        if dag is None:
            # create an empty dag
            dag = LinearGaussianBayesianNetwork()
            for v in self.score.data.columns:
                dag.add_cpd(LinearGaussianCPD(v))

        current_score = self.score.score(dag)
        for iter in range(self.max_iter):
            legal_ops = self._get_legal_operations(dag)
            random.shuffle(legal_ops)
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
            raise NotImplemented

        return score_delta


class GreedyEquivalentSearch(ScoreSearchEstimator):
    def __init__(self, score=None, max_iter=1_000_000, eps=1e-9):
        super().__init__(score=score, eps=eps)
        self.max_iter = max_iter

    def search(self):
        random.seed(time()) # TODO: handle randomness

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
                break

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
                break

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
            raise NotImplemented

        return score_delta
