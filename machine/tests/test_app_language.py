import unittest

from app import TEXT


class AppLanguageTests(unittest.TestCase):
    def test_thai_and_english_have_matching_translation_keys(self) -> None:
        thai, english = TEXT.values()

        self.assertEqual(set(thai), set(english))
        self.assertTrue(all(isinstance(value, str) for value in thai.values()))
        self.assertTrue(all(isinstance(value, str) for value in english.values()))


if __name__ == "__main__":
    unittest.main()
