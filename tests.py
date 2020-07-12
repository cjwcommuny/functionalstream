from functionalstream import Stream
import unittest

class StreamTestCase(unittest.TestCase):
    def test_normal(self):
        stream = Stream(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).to_list()
        self.assertListEqual(stream, [0, 6, 12, 18, 24])

    def test_star(self):
        stream = Stream([(1,2), (3,4), (6,5)]).filter(lambda x, y: x < y, star=True).to_list()
        self.assertListEqual(stream, [(1, 2), (3, 4)])
