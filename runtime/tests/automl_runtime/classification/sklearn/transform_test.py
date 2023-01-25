import unittest


class DummyTest(unittest.TestCase):
    """
    Dummy test to check github action setup.
    """

    def test_new(self):
        import sklearn
        sklearn.__version__
