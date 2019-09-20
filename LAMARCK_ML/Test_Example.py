import unittest


@unittest.skip("showing class skipping")
class TestExample(unittest.TestCase):

  def setUp(self):
    # self.widget = Widget('The widget')
    pass

  def tearDown(self):
    # self.widget.dispose()
    pass

  @unittest.expectedFailure
  def test_return5(self):
    ex = Example()
    self.assertEqual(ex.return5(), 6)

  @unittest.skip("demonstrating skipping")
  def test_nothing(self):
    self.fail("shouldn't happen")

  @unittest.skipIf(mylib.__version__ < (1, 3),
                   "not supported in this library version")
  def test_format(self):
    # Tests that work for only a certain version of the library.
    pass

  @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
  def test_windows_support(self):
    # windows specific testing code
    pass

  def test_even(self):
    """
    Test that numbers between 0 and 5 are all even.
    """
    for i in range(0, 6):
      with self.subTest(i=i):
        self.assertEqual(i % 2, 0)
