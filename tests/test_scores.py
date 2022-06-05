import os
import unittest

import numpy as np
import pandas as pd
from lgbn.models import LinearGaussianBayesianNetwork, LinearGaussianCPD
from lgbn.scores import BaseScore, BICScore, LogLikScore
from scipy.stats import norm


class TestBaseScore(unittest.TestCase):

    def test_score_fam_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            score = BaseScore(None)
            score.score_fam('A', ('B', 'C'))


class TestLogLikScore(unittest.TestCase):

    def test_score_bnlearn(self):
        # bnlearn model string "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"
        net = LinearGaussianBayesianNetwork()
        A = LinearGaussianCPD('A')
        B = LinearGaussianCPD('B')
        E = LinearGaussianCPD('C')
        G = LinearGaussianCPD('G')
        C = LinearGaussianCPD('C', parents='AB')
        D = LinearGaussianCPD('D', parents='B')
        F = LinearGaussianCPD('F', parents='ADEG')

        [net.add_cpd(x) for x in (A, B, C, D, E, F, G)]
        data = pd.read_csv(
            os.path.join('tests', 'test_data', 'bnlearn-gaussian.csv')
        )
        score = LogLikScore(data)
        sc = score.score(net)
        BNLEARN_REFERENCE_VALUE = -53131.917260
        self.assertAlmostEqual(
            sc,
            BNLEARN_REFERENCE_VALUE,
            delta=abs(1e-6 * BNLEARN_REFERENCE_VALUE)
        )


class TestBICScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.chain_data = pd.read_csv(
            os.path.join('tests', 'test_data', 'chain.csv')
        )

    def test_score_single_node(self):
        G = LinearGaussianBayesianNetwork()
        cpd = LinearGaussianCPD('A', mean=1, var=2)
        G.add_cpd(cpd)

        data = pd.DataFrame([0, -1, 1], columns=('A', ))
        score = BICScore(data)

        self.assertEqual(score.score(G), self._expected_single_node(data))

    def test_score_single_node_vs_matlab(self):
        G = LinearGaussianBayesianNetwork()
        G.add_cpd(LinearGaussianCPD('A'))

        score = BICScore(self.chain_data)
        sc = score.score(G)

        # assert they agree within 1 part per million
        MATLAB_REFERENCE_VALUE = -3.163210236416053e+04
        self.assertAlmostEqual(
            sc,
            MATLAB_REFERENCE_VALUE,
            delta=abs(1e-6 * MATLAB_REFERENCE_VALUE)
        )

    def test_score_chain_vs_matlab(self):
        G = LinearGaussianBayesianNetwork()
        G.add_cpd(LinearGaussianCPD('A'))
        G.add_cpd(LinearGaussianCPD('B', parents=['A'], weights=[1]))
        G.add_cpd(LinearGaussianCPD('C', parents=['B'], weights=[1]))

        score = BICScore(self.chain_data)
        sc = score.score(G)

        MATLAB_REFERENCE_VALUE = -9.296057540613585e+04
        self.assertAlmostEqual(
            sc,
            MATLAB_REFERENCE_VALUE,
            delta=abs(1e-6 * MATLAB_REFERENCE_VALUE)
        )

    def test_score_bnlearn(self):
        net = LinearGaussianBayesianNetwork()
        # "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"
        A = LinearGaussianCPD('A')
        B = LinearGaussianCPD('B')
        E = LinearGaussianCPD('C')
        G = LinearGaussianCPD('G')
        C = LinearGaussianCPD('C', parents=('A', 'B'), weights=(1, 1))
        D = LinearGaussianCPD('D', parents=('B', ), weights=(1, ))
        F = LinearGaussianCPD(
            'F', parents=('A', 'D', 'E', 'G'), weights=(1, 1, 1, 1)
        )

        [net.add_cpd(x) for x in (A, B, C, D, E, F, G)]

        data = pd.read_csv(
            os.path.join('tests', 'test_data', 'bnlearn-gaussian.csv')
        )
        score = BICScore(data)
        sc = score.score(net)
        BNLEARN_REFERENCE_VALUE = -53221.35
        self.assertAlmostEqual(
            sc,
            BNLEARN_REFERENCE_VALUE,
            delta=abs(1e-6 * BNLEARN_REFERENCE_VALUE)
        )

    def _expected_single_node(self, data):
        # estimate parameters
        mean = data.mean()
        var = data.var()

        log_lik = np.sum(norm.logpdf(data, loc=mean, scale=np.sqrt(var)))
        penalty = .5 * 2 * np.log(len(data))  # 1/2 * # params * log(# cases)
        return log_lik - penalty
