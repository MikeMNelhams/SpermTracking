import unittest.mock  # Python 3

import tracking_algorithm40 as tp

test_tp = '49'


class TestNone(unittest.TestCase):
    def test_none0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='none', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_none1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='none', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_none5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='none', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_none10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='none', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_none13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='none', plot=True, plot_type='3d')
        self.assertTrue(True)


class TestDBSCAN(unittest.TestCase):
    def test_dbscan0(self):
        tp.run_main(tp=test_tp, cover='00', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_dbscan1(self):
        tp.run_main(tp=test_tp, cover='01', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_dbscan5(self):
        tp.run_main(tp=test_tp, cover='05', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_dbscan10(self):
        tp.run_main(tp=test_tp, cover='10', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_dbscan13(self):
        tp.run_main(tp=test_tp, cover='13', plot=True, plot_type='3d')
        self.assertTrue(True)


class TestGMM(unittest.TestCase):
    def test_GMM0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='gmm', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_GMM1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='gmm', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_GMM5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='gmm', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_GMM10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='gmm', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_GMM13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='gmm', plot=True, plot_type='3d')
        self.assertTrue(True)


class TestMike(unittest.TestCase):
    def test_Mike0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='mike', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Mike1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='mike', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Mike5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='mike', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Mike10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='mike', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Mike13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='mike', plot=True, plot_type='3d')
        self.assertTrue(True)


class TestHDBSCAN(unittest.TestCase):
    def test_HDBSCAN0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='hdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HDBSCAN1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='hdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HDBSCAN5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='hdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HDBSCAN10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='hdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HDBSCAN13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='hdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)


class TestRichard(unittest.TestCase):
    def test_Richard0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='richard-dbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Richard1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='richard-dbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Richard5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='richard-dbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Richard10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='richard-dbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_Richard13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='richard-dbscan', plot=True, plot_type='3d')
        self.assertTrue(True)


class TestHTDBSCAN(unittest.TestCase):
    def test_HTDBSCAN0(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='htdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HTDBSCAN1(self):
        tp.run_main(tp=test_tp, cover='01', algorithm='htdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HTDBSCAN5(self):
        tp.run_main(tp=test_tp, cover='05', algorithm='htdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HTDBSCAN10(self):
        tp.run_main(tp=test_tp, cover='10', algorithm='htdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)

    def test_HTDBSCAN13(self):
        tp.run_main(tp=test_tp, cover='13', algorithm='htdbscan', plot=True, plot_type='3d')
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
