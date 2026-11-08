from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from math import hypot
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class PairRisk:
    first_id: int
    second_id: int
    ttc_seconds: float
    separation_pixels: float
    score: float


@dataclass
class RiskEvent:
    first_id: int
    second_id: int
    first_seen_seconds: float
    last_seen_seconds: float
    peak_score: float
    minimum_ttc_seconds: float
    minimum_separation_pixels: float

    @property
    def level(self) -> str:
        if self.peak_score >= 75:
            return "สูง"
        if self.peak_score >= 50:
            return "ปานกลาง"
        return "เฝ้าระวัง"


@dataclass(frozen=True)
class AnalysisResult:
    output_path: Path
    fps: float
    frames_processed: int
    duration_seconds: float
    tracked_objects: int
    trajectory_records: int
    tracks_at_least_10_frames: int
    events: list[RiskEvent]


def estimate_pair_risk(
    first_id: int,
    first_history: list[TrackPoint] | deque[TrackPoint],
    second_id: int,
    second_history: list[TrackPoint] | deque[TrackPoint],
    fps: float,
    horizon_seconds: float,
) -> PairRisk | None:
    """Estimate future contact from image-space track motion (not calibrated world TTC)."""
    if fps <= 0 or horizon_seconds <= 0:
        raise ValueError("fps and horizon_seconds must be positive")
    if len(first_history) < 3 or len(second_history) < 3:
        return None

    first_start, first_end = first_history[0], first_history[-1]
    second_start, second_end = second_history[0], second_history[-1]
    elapsed = (first_end.frame - first_start.frame) / fps
    second_elapsed = (second_end.frame - second_start.frame) / fps
    if elapsed <= 0 or second_elapsed <= 0:
        return None

    first_velocity = (
        (first_end.center_x - first_start.center_x) / elapsed,
        (first_end.center_y - first_start.center_y) / elapsed,
    )
    second_velocity = (
        (second_end.center_x - second_start.center_x) / second_elapsed,
        (second_end.center_y - second_start.center_y) / second_elapsed,
    )
    relative_position = (
        second_end.center_x - first_end.center_x,
        second_end.center_y - first_end.center_y,
    )
    relative_velocity = (
        second_velocity[0] - first_velocity[0],
        second_velocity[1] - first_velocity[1],
    )
    velocity_squared = (
        relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    )
    if velocity_squared < 1:
        return None

    ttc = -(
        relative_position[0] * relative_velocity[0]
        + relative_position[1] * relative_velocity[1]
    ) / velocity_squared
    if ttc <= 0 or ttc > horizon_seconds:
        return None

    closest_x = relative_position[0] + relative_velocity[0] * ttc
    closest_y = relative_position[1] + relative_velocity[1] * ttc
    separation = hypot(closest_x, closest_y)
    contact_radius = (
        hypot(first_end.width, first_end.height)
        + hypot(second_end.width, second_end.height)
    ) * 0.25
    if contact_radius <= 0 or separation >= contact_radius:
        return None

    proximity_score = 1 - separation / contact_radius
    urgency_score = 1 - ttc / horizon_seconds
    score = round(100 * (0.6 * proximity_score + 0.4 * urgency_score), 1)
    return PairRisk(
        first_id=first_id,
        second_id=second_id,
        ttc_seconds=ttc,
        separation_pixels=separation,
        score=score,
    )


def build_density_heatmap(
    trajectory_centers: list[tuple[float, float]],
    width: int,
    height: int,
):
    """Build the whole-video centroid density heatmap used by the notebook."""
    import cv2
    import numpy as np

    density = np.zeros((height, width), dtype=np.float32)
    for center_x, center_y in trajectory_centers:
        x = int(round(center_x))
        y = int(round(center_y))
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1

    density_blur = cv2.GaussianBlur(density, (0, 0), sigmaX=25, sigmaY=25)
    if density_blur.max() > 0:
        density_norm = cv2.normalize(
            density_blur, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_blur, dtype=np.uint8)

    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    return cv2.GaussianBlur(heatmap, (0, 0), 3)


def draw_motion_overlays(
    frame,
    current_tracks: dict[int, tuple[tuple[int, int, int, int], str]],
    trail_points: dict[int, deque[tuple[int, int]]],
    heatmap,
    persistent_track_ids: set[int] | None = None,
) -> None:
    """Apply the notebook's static density overlay and 30-point white trails."""
    import cv2

    frame[:] = cv2.addWeighted(frame, 0.78, heatmap, 0.22, 0)
    for track_id in current_tracks:
        points = list(trail_points.get(track_id, ()))
        persistent = (
            track_id in persistent_track_ids
            if persistent_track_ids is not None
            else len(points) >= 10
        )
        if not persistent:
            continue
        for start, end in zip(points[-30:-1], points[-29:]):
            cv2.line(
                frame,
                start,
                end,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    confidence: float = 0.5,
    horizon_seconds: float = 3.0,
    progress_callback: Callable[[float], None] | None = None,
) -> AnalysisResult:
    """Run YOLO tracking, render detections, and collect image-space near-miss events."""
    source_path = Path(input_path)
    weights_path = Path(model_path)
    destination_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"ไม่พบวิดีโอต้นฉบับ: {source_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล: {weights_path}. "
            "วางไฟล์ .pt ใน machine/models หรือกำหนด TRAFFIC_MODEL_PATH"
        )
    if not 0 < confidence < 1:
        raise ValueError("confidence ต้องอยู่ระหว่าง 0 ถึง 1")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds ต้องมากกว่า 0")

    import cv2
    import imageio_ffmpeg
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError("เปิดวิดีโอไม่ได้ กรุณาตรวจสอบชนิดไฟล์และความเสียหายของวิดีโอ")

    tracks: dict[int, deque[TrackPoint]] = {}
    seen_track_ids: set[int] = set()
    track_observation_counts: dict[int, int] = {}
    trajectory_centers: list[tuple[float, float]] = []
    frame_tracks: list[dict[int, tuple[tuple[int, int, int, int], str]]] = []
    frame_risks_by_frame: list[dict[tuple[int, int], PairRisk]] = []
    events: list[RiskEvent] = []
    active_events: dict[tuple[int, int], RiskEvent] = {}
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    try:
        model = YOLO(str(weights_path))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number == 0:
                height, width = frame.shape[:2]

            prediction = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                verbose=False,
            )[0]
            current_tracks: dict[int, tuple[tuple[int, int, int, int], str]] = {}
            boxes = prediction.boxes
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                names = prediction.names
                for coordinates_box, track_id, class_id in zip(
                    coordinates, track_ids, class_ids
                ):
                    x1, y1, x2, y2 = (int(value) for value in coordinates_box)
                    track_id = int(track_id)
                    seen_track_ids.add(track_id)
                    track_observation_counts[track_id] = (
                        track_observation_counts.get(track_id, 0) + 1
                    )
                    tracks.setdefault(track_id, deque(maxlen=8)).append(
                        TrackPoint(
                            frame=frame_number,
                            center_x=(x1 + x2) / 2,
                            center_y=(y1 + y2) / 2,
                            width=max(0, x2 - x1),
                            height=max(0, y2 - y1),
                        )
                    )
                    trajectory_centers.append(
                        ((x1 + x2) / 2, (y1 + y2) / 2)
                    )
                    class_name = str(names.get(class_id, class_id))
                    current_tracks[track_id] = (
                        (x1, y1, x2, y2),
                        class_name,
                    )

            frame_risks: dict[tuple[int, int], PairRisk] = {}
            for first_id, second_id in combinations(current_tracks, 2):
                risk = estimate_pair_risk(
                    first_id,
                    tracks[first_id],
                    second_id,
                    tracks[second_id],
                    fps,
                    horizon_seconds,
                )
                if risk is None:
                    continue
                pair = tuple(sorted((first_id, second_id)))
                frame_risks[pair] = risk
                event = active_events.get(pair)
                if event is None or frame_number / fps - event.last_seen_seconds > 1:
                    event = RiskEvent(
                        first_id=pair[0],
                        second_id=pair[1],
                        first_seen_seconds=frame_number / fps,
                        last_seen_seconds=frame_number / fps,
                        peak_score=risk.score,
                        minimum_ttc_seconds=risk.ttc_seconds,
                        minimum_separation_pixels=risk.separation_pixels,
                    )
                    active_events[pair] = event
                    events.append(event)
                event.last_seen_seconds = frame_number / fps
                event.peak_score = max(event.peak_score, risk.score)
                event.minimum_ttc_seconds = min(
                    event.minimum_ttc_seconds, risk.ttc_seconds
                )
                event.minimum_separation_pixels = min(
                    event.minimum_separation_pixels, risk.separation_pixels
                )

            frame_tracks.append(current_tracks)
            frame_risks_by_frame.append(frame_risks)
            frame_number += 1
            if progress_callback is not None and total_frames > 0:
                progress_callback(0.5 * min(frame_number / total_frames, 1.0))

        if frame_number == 0:
            raise ValueError("วิดีโอไม่มีเฟรมที่อ่านได้")
    finally:
        capture.release()

    heatmap = build_density_heatmap(trajectory_centers, width, height)
    persistent_track_ids = {
        track_id
        for track_id, count in track_observation_counts.items()
        if count >= 10
    }
    trail_points: dict[int, deque[tuple[int, int]]] = {}
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError("วิดีโอเปิดซ้ำเพื่อสร้างผลลัพธ์ไม่ได้")
    writer = imageio_ffmpeg.write_frames(
        str(destination_path),
        (width, height),
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        fps=fps,
        quality=8,
        codec="libx264",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )

    try:
        writer.send(None)
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_number >= len(frame_tracks):
                raise ValueError("จำนวนเฟรมวิดีโอเปลี่ยนระหว่างการประมวลผล")
            current_tracks = frame_tracks[frame_number]
            frame_risks = frame_risks_by_frame[frame_number]
            for track_id, (coordinates_box, _) in current_tracks.items():
                x1, y1, x2, y2 = coordinates_box
                trail_points.setdefault(track_id, deque(maxlen=30)).append(
                    ((x1 + x2) // 2, (y1 + y2) // 2)
                )

            draw_motion_overlays(
                frame,
                current_tracks,
                trail_points,
                heatmap,
                persistent_track_ids,
            )
            risky_ids = {
                track_id
                for first_id, second_id in frame_risks
                for track_id in (first_id, second_id)
            }
            for track_id, (coordinates_box, class_name) in current_tracks.items():
                x1, y1, x2, y2 = coordinates_box
                color = (0, 0, 255) if track_id in risky_ids else (40, 190, 80)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} #{track_id}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
            for pair, risk in frame_risks.items():
                first_box = current_tracks[pair[0]][0]
                cv2.putText(
                    frame,
                    f"NEAR-MISS {risk.score:.0f} | TTC {risk.ttc_seconds:.1f}s",
                    (first_box[0], min(frame.shape[0] - 10, first_box[3] + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            writer.send(frame.tobytes())
            frame_number += 1
            if progress_callback is not None and total_frames > 0:
                progress_callback(
                    0.5 + 0.5 * min(frame_number / total_frames, 1.0)
                )

        if frame_number != len(frame_tracks):
            raise ValueError("จำนวนเฟรมวิดีโอเปลี่ยนระหว่างการประมวลผล")
    finally:
        capture.release()
        writer.close()

    return AnalysisResult(
        output_path=destination_path,
        fps=fps,
        frames_processed=frame_number,
        duration_seconds=frame_number / fps,
        tracked_objects=len(seen_track_ids),
        trajectory_records=sum(track_observation_counts.values()),
        tracks_at_least_10_frames=sum(
            count >= 10 for count in track_observation_counts.values()
        ),
        events=events,
    )
