import sys
import unittest

from src.application.common import Environment


class TestCommon(unittest.TestCase):
    def test_get_os(self):
        environment = Environment.from_env({})
        self.assertEqual(
            environment.os,
            "linux" if sys.platform in {"linux", "linux2"} else "windows",
        )

    def test_recommendations_limit_set_via_env(self):
        environment = Environment.from_env({"RECOMMENDATIONS_LIMIT": "100"})
        self.assertEqual(100, environment.recommendations_limit)
