import logging
import os
import shutil
import tempfile
import unittest

import mock_logger
from src.application.app import main
from src.application.common import Environment
from config import ROOT


class TestApp(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = 16 * 1024
        self.temp = tempfile.mkdtemp()

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

    def test_output_dir_not_empty(self):
        output = ROOT.joinpath("test", "resources", "output")
        with mock_logger.MockLogger() as logger:
            self.assertEqual(1, self.main(["--output", str(output)]))
            self.assertIn(
                (
                    "ERROR",
                    f"Output directory {str(output)} is not empty.",
                ),
                logger.messages,
            )
