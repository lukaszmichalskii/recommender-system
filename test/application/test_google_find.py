import multiprocessing
import os
import threading
import unittest

from src.application.google_find import GoogleSearchError
from src.application import google_find


class TestGoogleFind(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_google_api_params(self):
        expected = {"query": "test", "limit": 1, "indent": True, "key": "dummy_key"}

        self.assertEqual(
            expected,
            google_find.google_api_params(
                query="test", limit=1, indent=True, api_key="dummy_key"
            ),
        )

    def test_build_url(self):
        params = {"query": "test", "limit": 1, "indent": True, "key": "dummy_key"}
        expected = "https://kgsearch.googleapis.com/v1/entities:search?query=test&limit=1&indent=True&key=dummy_key"
        self.assertEqual(expected, google_find.build_url(params))

    def test_search(self):
        expected = ["https://en.wikipedia.org/wiki/Eichmann_(film)"]
        try:
            actual = google_find.google_search(
                "Eichmann (2007), Movie",
                threading.BoundedSemaphore(multiprocessing.cpu_count()),
                os.environ.get("API_KEY"),
            )
            self.assertEqual(expected, actual)
        except google_find.GoogleSearchError:
            self.fail("GET request error response from API")

    def test_search_cannot_find_url(self):
        expected = []
        try:
            actual = google_find.google_search(
                "My Sassy Girl (Yeopgijeogin geunyeo) (2001), Movie",
                threading.BoundedSemaphore(multiprocessing.cpu_count()),
                os.environ.get("API_KEY"),
            )
            self.assertEqual(expected, actual)
        except google_find.GoogleSearchError:
            self.fail("GET request error response from API")

    def test_search_fails_on_missing_api_key(self):
        with self.assertRaises(GoogleSearchError):
            google_find.google_search(
                "Eichmann (2007), Movie",
                threading.BoundedSemaphore(multiprocessing.cpu_count()),
                None,
            )
