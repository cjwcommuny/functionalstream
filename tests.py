from functionalstream import Stream
import unittest

class StreamTestCase(unittest.TestCase):
    def test(self):
        stream = Stream(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).to_list()
        self.assertListEqual(stream, [0, 6, 12, 18, 24])

