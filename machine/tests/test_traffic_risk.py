import tempfile
import unittest
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from traffic_risk import (
    TrackPoint,
    build_density_heatmap,
    draw_motion_overlays,
    estimate_pair_risk,
    process_video,
)


def history(x_positions: list[float], y: float = 0) -> list[TrackPoint]:
    return [
        TrackPoint(
            frame=index,
            center_x=x,
            center_y=y,
            width=20,
            height=20,
        )
        for index, x in enumerate(x_positions)
    ]


class EstimatePairRiskTests(unittest.TestCase):
    def test_detects_predicted_contact(self) -> None:
        risk = estimate_pair_risk(
            1,
            history([0, 5, 10]),
            2,
            history([100, 95, 90]),
            fps=10,
            horizon_seconds=3,
        )
        self.assertIsNotNone(risk)
        assert risk is not None
        self.assertAlmostEqual(risk.ttc_seconds, 0.8)
        self.assertEqual(risk.separation_pixels, 0)
        self.assertGreater(risk.score, 75)

    def test_motion_overlays_draw_heatmap_and_vehicle_trail(self) -> None:
        centers = [(25, 50)] * 12 + [(75, 50)]
        heatmap = build_density_heatmap(centers, width=100, height=100)
        self.assertEqual(heatmap.shape, (100, 100, 3))
        self.assertGreater(int(heatmap[50, 25, 2]), int(heatmap[50, 25, 0]))

        density = np.zeros((100, 100), dtype=np.float32)
        for x, y in centers:
            density[int(round(y)), int(round(x))] += 1
        expected_density = cv2.GaussianBlur(
            density, (0, 0), sigmaX=25, sigmaY=25
        )
        expected_density = cv2.normalize(
            expected_density, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        expected_heatmap = cv2.applyColorMap(
            expected_density, cv2.COLORMAP_JET
        )
        expected_heatmap = cv2.GaussianBlur(expected_heatmap, (0, 0), 3)
        np.testing.assert_array_equal(heatmap, expected_heatmap)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        current_tracks = {1: ((19, 40, 30, 60), "car")}
        trail_points = {1: deque((x, 50) for x in range(10, 20))}
        draw_motion_overlays(
            frame,
            current_tracks,
            trail_points,
            heatmap,
            persistent_track_ids={1},
        )

        self.assertGreater(int(frame[50, 15].min()), 200)

    def test_ignores_objects_moving_apart(self) -> None:
        risk = estimate_pair_risk(
            1,
            history([0, 5, 10]),
            2,
            history([30, 35, 40]),
            fps=10,
            horizon_seconds=3,
        )

        self.assertIsNone(risk)

    def test_ignores_future_paths_that_do_not_intersect(self) -> None:
        risk = estimate_pair_risk(
            1,
            history([0, 5, 10], y=0),
            2,
            history([100, 95, 90], y=100),
            fps=10,
            horizon_seconds=3,
        )

        self.assertIsNone(risk)

    def test_rejects_invalid_time_parameters(self) -> None:
        with self.assertRaises(ValueError):
            estimate_pair_risk(1, [], 2, [], fps=0, horizon_seconds=3)

    def test_missing_model_fails_with_actionable_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()
            with self.assertRaisesRegex(FileNotFoundError, "ไม่พบไฟล์โมเดล"):
                process_video(
                    input_path,
                    Path(temp_dir) / "output.mp4",
                    Path(temp_dir) / "missing.pt",
                )


if __name__ == "__main__":
    unittest.main()
