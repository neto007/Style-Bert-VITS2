import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from style_bert_vits2.nlp.portuguese.g2p import __text_to_words as _words
    MODULES_AVAILABLE = True
except Exception:
    MODULES_AVAILABLE = False


class TestPTTokenizer(unittest.TestCase):
    def test_tokenizer_basic(self):
        if not MODULES_AVAILABLE:
            self.skipTest("Dependencies not available (numpy/phonemizer). Skipping PT tokenizer test.")
        tokens = _words("Olá, mundo!")
        self.assertGreater(len(tokens), 0)


if __name__ == "__main__":
    unittest.main()
