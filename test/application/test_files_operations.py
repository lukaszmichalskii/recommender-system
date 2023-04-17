import unittest

from src.application.files_operations import (
    MalformedFileFormat,
    get_ratings,
)
from src.config import ROOT


class TestFilesOperations(unittest.TestCase):
    def setUp(self) -> None:
        self.resources = ROOT.joinpath("test", "resources")

    def test_invalid_file_structure(self):
        file = self.resources.joinpath("invalid_structure.csv")
        self.assertRaises(MalformedFileFormat, lambda: get_ratings(file))
