import unittest

from mechanistic.run_jlens_exclusion_layerwise import (
    SOURCE_LAYERS,
    _scenario_layers,
    _scenario_target,
    _scenarios,
)


class LayerwiseExclusionTests(unittest.TestCase):
    def test_scenarios_cover_each_causally_actionable_readout(self):
        scenarios = _scenarios()
        self.assertEqual(SOURCE_LAYERS, tuple(range(40, 63)))
        self.assertEqual(len(scenarios), 52)
        self.assertIn("exclude_neutral_into_game_L41", scenarios)
        self.assertIn("exclude_game_into_neutral_L63", scenarios)
        self.assertNotIn("exclude_neutral_into_game_L64", scenarios)

    def test_layer_and_target_resolution(self):
        self.assertEqual(_scenario_layers("exclude_neutral_into_game_L41"), (40,))
        self.assertEqual(_scenario_layers("exclude_game_into_neutral_L63"), (62,))
        self.assertEqual(
            _scenario_layers("exclude_neutral_into_game_L49_63"),
            tuple(range(48, 63)),
        )
        self.assertEqual(_scenario_target("exclude_neutral_into_game_L47"), "incorrect")
        self.assertEqual(_scenario_target("exclude_game_into_neutral_L47"), "neutral")


if __name__ == "__main__":
    unittest.main()
