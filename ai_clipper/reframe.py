"""
Active-speaker / subject reframing helpers.

Computes where the main subject (face) sits horizontally so a 16:9 source can
be cropped to a vertical 9:16 frame that keeps the speaker in view instead of a
blind center crop. Uses OpenCV's YuNet DNN face detector (bundled model), and
degrades gracefully to a center crop when detection is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compute_subject_center_x(
    video_path: Path,
    start: float,
    duration: float,
    samples: int = 12,
) -> Optional[float]:
    """
    Sample frames across a clip and return the median horizontal center of the
    largest detected face as a fraction in [0, 1]. Returns None when no faces
    are found or OpenCV/the model is unavailable (caller should center-crop).
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available; falling back to center crop.")
        return None

    if not _MODEL_PATH.exists() or not hasattr(cv2, "FaceDetectorYN"):
        logger.warning("YuNet model/detector unavailable; falling back to center crop.")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        detector = cv2.FaceDetectorYN.create(
            str(_MODEL_PATH), "", (320, 320), score_threshold=0.6
        )
    except Exception as e:  # noqa: BLE001 - detector construction can vary by build
        logger.warning(f"Could not create face detector: {e}")
        cap.release()
        return None

    centers: List[float] = []
    step = duration / max(1, samples)
    try:
        for i in range(samples):
            t = start + step * (i + 0.5)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue
            # Largest face (by area) is the most likely on-screen speaker.
            largest = max(faces, key=lambda f: float(f[2]) * float(f[3]))
            cx = (float(largest[0]) + float(largest[2]) / 2.0) / w
            centers.append(min(1.0, max(0.0, cx)))
    finally:
        cap.release()

    if not centers:
        return None
    return _median(centers)
