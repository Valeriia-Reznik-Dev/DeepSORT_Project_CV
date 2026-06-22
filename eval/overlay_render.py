"""Headless tracking overlay video renderer (no GUI)."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

import deep_sort_app
from application_util.visualization import create_unique_color_uchar

DEFAULT_FPS = 20.0


def _fps_from_seqinfo(seq_info: dict) -> float:
    update_ms = seq_info.get("update_ms")
    if update_ms:
        return max(1.0, 1000.0 / float(update_ms))
    return DEFAULT_FPS


def _draw_tracks(image: np.ndarray, track_ids: np.ndarray, boxes: np.ndarray) -> None:
    thickness = 2
    for track_id, box in zip(track_ids, boxes):
        color = create_unique_color_uchar(int(track_id))
        x, y, w, h = box.astype(int)
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(image, pt1, pt2, color, thickness)
        label = str(int(track_id))
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)[0]
        center = pt1[0] + 5, pt1[1] + 5 + text_size[1]
        pt2_bg = pt1[0] + 10 + text_size[0], pt1[1] + 10 + text_size[1]
        cv2.rectangle(image, pt1, pt2_bg, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness)


def render_sequence_overlay(
    sequence_dir: str | os.PathLike,
    result_file: str | os.PathLike,
    output_file: str | os.PathLike,
    *,
    fourcc: str = "MJPG",
) -> None:
    seq_info = deep_sort_app.gather_sequence_info(str(sequence_dir), None)
    results = np.loadtxt(str(result_file), delimiter=",")
    image_size = seq_info["image_size"]
    if image_size is None:
        raise ValueError(f"No images found in {sequence_dir}")

    height, width = image_size
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*fourcc),
        _fps_from_seqinfo(seq_info),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_file}")

    try:
        for frame_idx in range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1):
            img_path = seq_info["image_filenames"].get(frame_idx)
            if img_path is None:
                continue
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            mask = results[:, 0].astype(int) == frame_idx
            track_ids = results[mask, 1].astype(int)
            boxes = results[mask, 2:6]
            _draw_tracks(image, track_ids, boxes)
            writer.write(image)
    finally:
        writer.release()


def render_overlays_for_mot_dir(
    mot_dir: str | os.PathLike,
    result_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    fourcc: str = "MJPG",
) -> list[Path]:
    mot_dir = Path(mot_dir)
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    if not mot_dir.is_dir():
        print(f"SKIP overlays (missing): {mot_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    print(f"Rendering overlays from {result_dir} -> {output_dir}")
    for result_file in sorted(result_dir.glob("*.txt")):
        sequence = result_file.stem
        sequence_dir = mot_dir / sequence
        if not sequence_dir.is_dir():
            continue
        output_file = output_dir / f"{sequence}.avi"
        print(f"Saving {sequence} -> {output_file}")
        render_sequence_overlay(sequence_dir, result_file, output_file, fourcc=fourcc)
        written.append(output_file)
    return written
