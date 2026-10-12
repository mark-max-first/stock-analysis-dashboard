from pathlib import Path
import unittest


class RepositoryHygieneTests(unittest.TestCase):
    def test_removed_streamlit_width_api_is_not_reintroduced(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertNotIn("use_container_width", app_source)


if __name__ == "__main__":
    unittest.main()
