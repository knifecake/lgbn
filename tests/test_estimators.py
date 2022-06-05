import os
import unittest

import pandas as pd
from lgbn.estimators import GreedyEquivalentSearch, GreedyHillClimbing, K2Search, ScoreSearchEstimator
from lgbn.scores import BaseScore, BICScore, LogLikScore


class TestSearchEstimator(unittest.TestCase):
    def test_estimator(self):
        score = BaseScore(None)
        with self.assertRaises(ValueError):
            ScoreSearchEstimator(eps=0)

        sse = ScoreSearchEstimator(score=score, eps=1)

        self.assertEqual(sse.get_params(), {'score': score, 'eps': 1})
        self.assertEqual(sse.get_params(deep=False), {'score': score, 'eps': 1})

        sse.set_params(eps=2)
        self.assertEqual(sse.eps, 2)

        with self.assertRaises(ValueError):
            sse.set_params(eps=-2)

        with self.assertRaises(NotImplementedError):
            sse.search()

    def test_score_strings(self):
        sse = ScoreSearchEstimator('loglik')
        self.assertEqual(sse.score_class, LogLikScore)
        self.assertIsInstance(sse.score, LogLikScore)

        sse = ScoreSearchEstimator('bic')
        self.assertEqual(sse.score_class, BICScore)
        self.assertIsInstance(sse.score, BICScore)

        with self.assertRaises(ValueError):
            ScoreSearchEstimator('unknown')

        sse.set_params(score='loglik')
        self.assertEqual(sse.score_class, LogLikScore)
        self.assertIsInstance(sse.score, LogLikScore)
        

class TestK2Search(unittest.TestCase):
    def test_fit(self):
        data = pd.read_csv(os.path.join('tests', 'test_data', 'chain.csv'))
        estimator = K2Search(score='bic', ordering='ABC')
        estimator.fit(data)

        self.assertListEqual(list(estimator.model.predecessors('A')), [])
        self.assertListEqual(list(estimator.model.predecessors('B')), ['A'])
        self.assertListEqual(list(estimator.model.predecessors('C')), ['B'])

    
    def test_params(self):
        estimator = K2Search(score='bic', ordering='BAC')
        params = estimator.get_params()
        params['ordering'] = 'ABC'
        estimator.set_params(**params)
        self.assertEqual(estimator.ordering, 'ABC')
    
    def test_ordering_required(self):
        with self.assertRaises(ValueError):
            K2Search(score='bic')

class TestGreedyHillClimbing(unittest.TestCase):
    def test_fit(self):
        data = pd.read_csv(os.path.join('tests', 'test_data', 'chain.csv'))
        estimator = GreedyHillClimbing(score='bic')
        estimator.fit(data)

        # self.assertListEqual(list(estimator.model.predecessors('A')), ['B'])
        # self.assertListEqual(list(estimator.model.predecessors('B')), [])
        # self.assertListEqual(list(estimator.model.predecessors('C')), ['B'])

    
    def test_get_op_delta(self):
        # tests an edge case to ensure full coverage
        estimator = GreedyHillClimbing(score='bic')
        with self.assertRaises(NotImplementedError):
            estimator._get_op_delta(('X', ('A', 'B')), None)

    def test_params(self):
        estimator = GreedyHillClimbing(score='bic')
        params = estimator.get_params()
        params['max_iter'] = 10
        estimator.set_params(**params)
        self.assertEqual(estimator.max_iter, 10)


class TestGreedyEquivalentSearch(unittest.TestCase):
    def test_fit(self):
        data = pd.read_csv(os.path.join('tests', 'test_data', 'chain.csv'))
        estimator = GreedyEquivalentSearch(score='bic')
        estimator.fit(data)
        print(estimator.model)
        # self.assertListEqual(list(estimator.model.predecessors('A')), ['B'])
        # self.assertListEqual(list(estimator.model.predecessors('B')), [])
        # self.assertListEqual(list(estimator.model.predecessors('C')), ['B'])

    def test_fit_bnlearn(self):
        data = pd.read_csv(os.path.join('tests', 'test_data', 'bnlearn-gaussian.csv'))
        estimator = GreedyEquivalentSearch(score='bic')
        estimator.fit(data)

    def test_get_op_delta(self):
        # tests an edge case to ensure full coverage
        estimator = GreedyEquivalentSearch(score='bic')
        with self.assertRaises(NotImplementedError):
            estimator._get_op_delta(('X', ('A', 'B')), None)

    
    def test_params(self):
        estimator = GreedyEquivalentSearch(score='bic')
        params = estimator.get_params()
        params['max_iter'] = 10
        estimator.set_params(**params)
        self.assertEqual(estimator.max_iter, 10)