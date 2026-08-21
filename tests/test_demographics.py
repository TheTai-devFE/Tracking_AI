from __future__ import annotations

import unittest

from audience.demographics import map_age_group, normalize_gender


class DemographicsTest(unittest.TestCase):
    def test_age_boundaries(self) -> None:
        self.assertEqual(map_age_group(17.9), "0-17")
        self.assertEqual(map_age_group(18), "18-24")
        self.assertEqual(map_age_group(34.9), "25-34")
        self.assertEqual(map_age_group(55), "55+")

    def test_gender_normalization(self) -> None:
        self.assertEqual(normalize_gender("F"), "female")
        self.assertEqual(normalize_gender("Male"), "male")
        self.assertIsNone(normalize_gender("unsupported"))
