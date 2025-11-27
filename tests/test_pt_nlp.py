import unittest
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from style_bert_vits2.nlp.portuguese.normalizer import normalize_text
    MODULES_AVAILABLE = True
except Exception:
    MODULES_AVAILABLE = False


class TestPTNLP(unittest.TestCase):
    def test_normalize_text_pt(self):
        if not MODULES_AVAILABLE:
            self.skipTest("Dependencies not available (numpy). Skipping PT NLP test.")
        text = "Olá,mundo!Isso é um teste\nNovo parágrafo?"
        norm = normalize_text(text)
        self.assertIn("Olá, mundo!", norm)
        self.assertIn("Novo parágrafo?", norm)
        self.assertEqual(norm, norm.lower())


if __name__ == "__main__":
    unittest.main()
