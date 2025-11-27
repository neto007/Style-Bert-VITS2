import unittest


class TestPTLoader(unittest.TestCase):
    def test_parse_train_list_pt(self):
        line = "Data/pt_demo/wavs/utt001.wav|spk1|PT|ola mundo|_ o l a |0 0 0 0|1 1 2 1"
        self.assertEqual(len(line.split("|")), 7)


if __name__ == "__main__":
    unittest.main()
