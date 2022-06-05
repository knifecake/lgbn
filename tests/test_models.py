import unittest

from lgbn.models import (
    CPD, BayesianNetwork, LinearGaussianBayesianNetwork, LinearGaussianCPD
)


class TestCPD(unittest.TestCase):

    def test_casts_parents_to_tuple(self):
        cpd = CPD(node='A', parents='BCD')
        self.assertEqual(cpd.parents, ('B', 'C', 'D'))

    def test_serialization(self):
        cpd = CPD(node='A', parents=['B', 'C'])
        data = {'node': 'A', 'parents': ('B', 'C')}
        self.assertEqual(cpd.to_dict(), data)

        cpd = CPD.from_dict(data)
        self.assertEqual(cpd.node, 'A')
        self.assertEqual(cpd.parents, ('B', 'C'))

    def test_repr(self):
        cpd = CPD(node='A', parents='BCD')
        self.assertIsNotNone(repr(cpd))


class TestLinearGaussianCPD(unittest.TestCase):

    def test_casts_parents_to_tuple(self):
        cpd = LinearGaussianCPD(node='A', parents='BC', weights=[1, 2])
        self.assertEqual(cpd.weights, (1, 2))

    def test_checks_for_positive_variance(self):
        with self.assertRaises(ValueError):
            LinearGaussianCPD(node='A', var=-1)

    def test_checks_for_parent_weight_matching_length(self):
        with self.assertRaises(ValueError):
            LinearGaussianCPD(node='A', parents='AB', weights=(1, ))

    def test_serialization(self):
        cpd = LinearGaussianCPD(
            node='A', mean=-1, var=2, parents='BC', weights=[1, 2]
        )
        data = {
            'node': 'A',
            'mean': -1,
            'var': 2,
            'parents': ('B', 'C'),
            'weights': (1, 2)
        }

        self.assertEqual(cpd.to_dict(), data)

        cpd = LinearGaussianCPD.from_dict(data)
        self.assertEqual(cpd.node, 'A')
        self.assertEqual(cpd.mean, -1)
        self.assertEqual(cpd.var, 2)
        self.assertEqual(cpd.parents, ('B', 'C'))
        self.assertEqual(cpd.weights, (1, 2))

    def test_repr(self):
        cpd = LinearGaussianCPD(node='A', parents='BCD')
        self.assertIsNotNone(repr(cpd))


class TestBayesianNetwork(unittest.TestCase):

    def test_add_cpd(self):
        net = BayesianNetwork()
        net.add_cpd(CPD(node='A'))
        self.assertTrue(net.has_node('A'))

        net.add_cpd(CPD(node='B', parents='A'))
        self.assertTrue(net.has_node('B'))
        self.assertTrue(net.has_edge('A', 'B'))

    def test_apply_op(self):
        net = BayesianNetwork()
        net.add_cpd(CPD(node='A'))
        net.add_cpd(CPD(node='B'))

        net.apply_op(('+', ('A', 'B')))
        self.assertTrue(net.has_edge('A', 'B'))

        net.apply_op(('F', ('A', 'B')))
        self.assertFalse(net.has_edge('A', 'B'))
        self.assertTrue(net.has_edge('B', 'A'))

        net.apply_op(('-', ('B', 'A')))
        self.assertFalse(net.has_edge('B', 'A'))

        with self.assertRaises(NotImplementedError):
            net.apply_op(('X', ('B', 'C')))

    def test_update_cpd_structure(self):
        net = BayesianNetwork()
        net.add_cpd(CPD(node='A'))
        net.add_cpd(CPD(node='B'))
        net.apply_op(('+', ('A', 'B')))
        net.update_cpds_from_structure()

        self.assertEqual(net.cpds['B'].parents, ('A', ))

    def test_dict_serialization(self):
        net = BayesianNetwork()
        net.add_cpd(CPD(node='A'))
        net.add_cpd(CPD(node='B', parents=('A', )))
        data = [net.cpds['A'].to_dict(), net.cpds['B'].to_dict()]
        self.assertEqual(net.to_dict(), data)

        new_net = BayesianNetwork.from_dict(data)
        self.assertEqual(new_net.cpds['A'].node, 'A')
        self.assertEqual(new_net.cpds['B'].node, 'B')
        self.assertEqual(new_net.cpds['B'].parents, ('A', ))


class TestLinearGaussianBayesianNetwork(unittest.TestCase):

    def test_to_joint_gaussian(self):
        # Example from K&F p. 252
        net = LinearGaussianBayesianNetwork()
        net.add_cpd(LinearGaussianCPD('A', mean=1, var=4))
        net.add_cpd(
            LinearGaussianCPD(
                'B', mean=-3.5, var=4, parents=('A', ), weights=(.5, )
            )
        )
        net.add_cpd(
            LinearGaussianCPD(
                'C', mean=1, var=3, parents=('B', ), weights=(-1, )
            )
        )

        dist = net.to_joint_gaussian()
        self.assertListEqual(list(dist.mean), [1, -3, 4])

        # TODO: check manually, I think K&F is wrong in this case
        # self.assertListEqual([list(row) for row in dist.cov], [[4, 2, -2], [2, 5, -5], [-2, 5, 8]])

    def test_to_joint_gaussian2(self):
        net = LinearGaussianBayesianNetwork()
        net.add_cpd(LinearGaussianCPD('x1', mean=1, var=4))
        net.add_cpd(
            LinearGaussianCPD(
                'x2', mean=-5, var=4, parents=('x1', ), weights=(.5, )
            )
        )
        net.add_cpd(
            LinearGaussianCPD(
                'x3', mean=4, var=3, parents=('x2', ), weights=(-1, )
            )
        )

        dist = net.to_joint_gaussian()
        self.assertListEqual(list(dist.mean), [1, -4.5, 8.5])
        self.assertListEqual([list(row) for row in dist.cov],
                             [[4., 2., -2.], [2., 5., -5.], [-2., -5., 8.]])
