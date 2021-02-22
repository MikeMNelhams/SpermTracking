import unittest.mock  # Python 3

import tracking_algorithm42 as tp

test_tp = '49'


class TestEDGECASES(unittest.TestCase):
    def test_FALSE_ALG(self):
        tp.run_main(tp=test_tp, cover='00', algorithm='1234')
        self.assertTrue(True)

    def test_NO_PARAM(self):
        tp.run_main()
        self.assertTrue(True)

    def test_UNKNOWN_COVER(self):
        with self.assertRaises(SystemExit) as cm:
            tp.run_main(tp=test_tp, cover='-200', algorithm='kmeans')
        self.assertEqual(cm.exception.code, 1)

    def test_UNKNOWN_TP(self):
        with self.assertRaises(SystemExit) as cm:
            tp.run_main(tp='-200', cover='00', algorithm='kmeans')
        self.assertEqual(cm.exception.code, 1)

    def test_UNKNOWN_BOTH(self):
        with self.assertRaises(SystemExit) as cm:
            tp.run_main(tp='-200', cover='-200', algorithm='kmeans')
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
