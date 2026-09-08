import unittest

import numpy as np

from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)


def _mask(rows: slice, cols: slice, size: int = 4) -> np.ndarray:
    output = np.zeros((size, size), dtype=bool)
    output[rows, cols] = True
    return output


class TestInstanceAP(unittest.TestCase):
    def test_different_images_cannot_match(self):
        prediction = _mask(slice(0, 2), slice(0, 2))
        gt_same_image = _mask(slice(2, 4), slice(2, 4))
        gt_other_image = prediction.copy()

        result = instance_ap(
            [prediction],
            [0.9],
            [gt_same_image, gt_other_image],
            thresholds=(0.5,),
            pred_image_ids=["view_a"],
            gt_image_ids=["view_a", "view_b"],
        )

        self.assertEqual(result["ap_50"], 0.0)

    def test_matching_follows_confidence_order(self):
        gt = _mask(slice(0, 1), slice(0, 2))
        high_score_half = _mask(slice(0, 1), slice(0, 1))
        low_score_perfect = gt.copy()

        result = instance_ap(
            [high_score_half, low_score_perfect],
            [0.9, 0.1],
            [gt],
            thresholds=(0.5,),
            pred_image_ids=["view", "view"],
            gt_image_ids=["view"],
        )

        self.assertAlmostEqual(result["ap_50"], 1.0, places=6)

    def test_image_ids_must_be_supplied_together(self):
        mask = _mask(slice(0, 1), slice(0, 1))
        with self.assertRaisesRegex(ValueError, "provided together"):
            instance_ap(
                [mask],
                [0.5],
                [mask],
                thresholds=(0.5,),
                pred_image_ids=["view"],
            )

        with self.assertRaisesRegex(ValueError, "provided together"):
            instance_ap(
                [],
                [],
                [mask],
                thresholds=(0.5,),
                gt_image_ids=["view"],
            )

    def test_minimum_mask_area_filters_predictions_and_gt(self):
        probabilities = np.zeros((3, 3, 3), dtype=np.float32)
        probabilities[2] = 0.1
        probabilities[0, 0, 0] = 0.9
        probabilities[1, 1:, 1:] = 0.9
        pred_masks, _ = masks_from_group_probs(
            probabilities, void_channel=2, min_mask_area=2
        )
        self.assertEqual(len(pred_masks), 1)
        self.assertEqual(int(pred_masks[0].sum()), 4)

        instance_map = np.zeros((3, 3), dtype=np.int64)
        instance_map[0, 0] = 1
        instance_map[1:, 1:] = 2
        gt_masks = gt_masks_from_instance_map(instance_map, min_mask_area=2)
        self.assertEqual(len(gt_masks), 1)
        self.assertEqual(int(gt_masks[0].sum()), 4)


if __name__ == "__main__":
    unittest.main()
