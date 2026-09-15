import tempfile
import unittest

import torch

from tokengs.models.token_eru.query_metric_coupling import QueryMetricCoupling


class QueryMetricCheckpointTest(unittest.TestCase):
    def test_strict_state_roundtrip(self):
        torch.manual_seed(11)
        first = QueryMetricCoupling()
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(first.state_dict(), handle.name)
            second = QueryMetricCoupling()
            state = torch.load(handle.name, map_location="cpu", weights_only=True)
            result = second.load_state_dict(state, strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            for left, right in zip(first.parameters(), second.parameters()):
                self.assertTrue(torch.equal(left, right))
            self.assertNotIn("dino", " ".join(second.state_dict().keys()).lower())


if __name__ == "__main__":
    unittest.main()
