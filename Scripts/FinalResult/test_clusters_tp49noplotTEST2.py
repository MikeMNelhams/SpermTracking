import unittest.mock  # Python 3

import tracking_algorithm42 as tp

test_tp = '49'


class TestMIKEHTDBSCAN(unittest.TestCase):
    def test_HTDBSCAN0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='mike-htdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='mike-htdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='mike-htdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='mike-htdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='mike-htdbscan')
        self.assertTrue(True)


class TestMIKEHTHDBSCAN(unittest.TestCase):
    def test_HTDBSCAN0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='mike-hthdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='mike-hthdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='mike-hthdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='mike-hthdbscan')
        self.assertTrue(True)

    def test_HTDBSCAN13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='mike-hthdbscan')
        self.assertTrue(True)


class TestEvaluate(unittest.TestCase):
    def test_EvalKMEAN(self):
        tp.evaluate_success(algorithm='kmeans', tp='49', verbose=True, plot=False)
        tp.evaluate_success(algorithm='kmeans', tp='57', verbose=True, plot=False)
        self.assertTrue(True)

    def test_EvalDBSCAN(self):
        tp.evaluate_success(algorithm='dbscan', tp='49', verbose=True, plot=False)
        tp.evaluate_success(algorithm='dbscan', tp='57', verbose=True, plot=False)
        self.assertTrue(True)

    def test_EvalNone(self):
        tp.evaluate_success(algorithm='none', tp='49', verbose=True, plot=False)
        tp.evaluate_success(algorithm='none', tp='57', verbose=True, plot=False)
        self.assertTrue(True)

    def test_EvalMIKE(self):
        tp.evaluate_success(algorithm='mike', tp='49', verbose=True, plot=False)
        tp.evaluate_success(algorithm='mike', tp='57', verbose=True, plot=False)
        self.assertTrue(True)

    def test_EvalGMM(self):
        tp.evaluate_success(algorithm='gmm', tp='49', verbose=True, plot=False)
        tp.evaluate_success(algorithm='gmm', tp='57', verbose=True, plot=False)
        self.assertTrue(True)

    def test_EvalRICHARDDBSCAN(self):
        tp.evaluate_success(algorithm='richard-dbscan', tp='49', verbose=True, plot=False)
        tp.evaluate_success(algorithm='richard-dbscan', tp='57', verbose=True, plot=False)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
