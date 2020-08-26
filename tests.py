import cv2
from datascience.opencv import VideoCaptureIter

from functionalstream import Stream
import unittest
import logging
import sys

class StreamTestCase(unittest.TestCase):
    def test_normal(self):
        stream = Stream(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).to_list()
        self.assertListEqual(stream, [0, 6, 12, 18, 24])

    def test_star(self):
        stream = Stream([(1,2), (3,4), (6,5)]).filter(lambda x, y: x < y, star=True).to_list()
        self.assertListEqual(stream, [(1, 2), (3, 4)])

    def test_find_first(self):
        x = Stream(range(1, 10)).find_first(lambda x: x % 2 == 0)
        self.assertEqual(x, 2)

    def test_flatten(self):
        stream = Stream([[0,1], [2,3,4],[5,6,7,8]]).flatten().to_list()
        self.assertListEqual(stream, list(range(9)))

    def test_lazy(self):
        Stream([[0,1], [2,3], [4,5]])\
            .enumerate()\
            .peek(lambda idx, x: print(f'1: {x[0]}'))\
            .filter(lambda idx, x: idx % 2 == 0)\
            .peek(lambda idx, x: print(f'2: {x[0]}'))\
            .consume()