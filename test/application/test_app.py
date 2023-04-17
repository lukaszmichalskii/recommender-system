import logging
import os
import shutil
import tempfile
import unittest

from test import mock_logger
from src.application.app import main
from src.application.common import Environment, SUPPORTED_FORMAT
from src.config import ROOT


class TestApp(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = 16 * 1024
        self.temp = tempfile.mkdtemp()
        self.resources = ROOT.joinpath("test", "resources")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def main(self, args, env=None):
        cwd = os.getcwd()
        os.chdir(self.temp)
        try:
            logger = logging.getLogger("RSA")
            environment = Environment.from_env({} if env is None else env)
            return main(["app_runner.py"] + args, logger, environment)
        except SystemExit as e:
            return e.code
        finally:
            os.chdir(cwd)

    def test_csv_only_support(self):
        ratings = self.resources.joinpath("dummy.txt")
        with mock_logger.MockLogger() as logger:
            self.assertEqual(
                1, self.main(["--ratings", str(ratings), "--output", str(self.temp)])
            )
            self.assertIn(
                (
                    "ERROR",
                    f"Unsupported ratings file format, required {SUPPORTED_FORMAT} files only.",
                ),
                logger.messages,
            )

    def test_invalid_file_structure(self):
        ratings = self.resources.joinpath("invalid_structure.csv")
        with mock_logger.MockLogger() as logger:
            self.assertEqual(
                1, self.main(["--ratings", str(ratings), "--output", str(self.temp)])
            )
            self.assertIn(
                (
                    "ERROR",
                    "Error during user ratings reading. Details: File does not follow ratings convention: not enough "
                    "values to unpack (expected at least 2, got 1)",
                ),
                logger.messages,
            )

    def test_output_dir_not_empty(self):
        output = self.resources.joinpath("output")
        ratings = self.resources.joinpath("ratings.csv")
        with mock_logger.MockLogger() as logger:
            self.assertEqual(
                1, self.main(["--ratings", str(ratings), "--output", str(output)])
            )
            self.assertIn(
                (
                    "ERROR",
                    f"Output directory {str(output)} is not empty.",
                ),
                logger.messages,
            )
