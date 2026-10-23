#!/usr/bin/env python3
"""
Mall People Tracker — real-time people counting for a mall entrance / section.

Counts people crossing a virtual line in both directions (IN / OUT) and keeps
a live occupancy estimate (IN - OUT), the way a footfall-analytics system at a
shopping-mall entrance would.

Two modes share the exact same tracking, counting and overlay pipeline:

  * Real mode       -- YOLOv8 detects people in a video file or webcam feed.
                       IDs come from ByteTrack (``model.track``) by default, or
                       from the built-in centroid tracker if you prefer.
  * Simulation mode -- a synthetic mall-entrance scene is rendered with
                       OpenCV/NumPy and the simulated detector feeds ground-
                       truth boxes into the same pipeline. Needs only
                       opencv-python + numpy, so it runs anywhere.

Usage:
    python people_counter.py --simulate
    python people_counter.py --simulate --output demo.mp4 --duration 30
    python people_counter.py --source mall.mp4
    python people_counter.py --source 0                 # webcam
    python people_counter.py --source mall.mp4 --output counted.mp4

Direction convention: crossing the line left -> right counts as IN (entering),
right -> left counts as OUT (leaving). Put the line at a doorway and that maps
directly onto a real entrance.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

PERSON_CLASS_ID = 0  # COCO class id for "person"

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
COL_LINE = (0, 215, 255)       # yellow counting line
COL_BOX = (80, 220, 80)        # green detection box
COL_IN = (80, 220, 80)         # green
COL_OUT = (70, 70, 235)        # red
COL_TEXT = (255, 255, 255)
COL_PANEL = (0, 0, 0)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One detected person in one frame (pixel coordinates)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    track_id: int | None = None  # filled in by a tracker or by YOLO/ByteTrack

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Trackers — assign persistent IDs to detections.
#
# The pipeline accepts detections that already carry an ID (YOLO + ByteTrack)
# or raw boxes, in which case the CentroidTracker below assigns IDs. Either
# way the counting/overlay code downstream is identical.
# ---------------------------------------------------------------------------
class CentroidTracker:
    """Simple centroid tracker: matches detections between frames by distance.

    Good enough for a counting line in a corridor/entrance view, and it works
    for any detection source — including the simulation, which has no YOLO.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 70.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 1
        self.objects: dict[int, Detection] = {}
        self.disappeared: dict[int, int] = {}

    def _register(self, det: Detection) -> None:
        det.track_id = self.next_id
        self.objects[self.next_id] = det
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, track_id: int) -> None:
        del self.objects[track_id]
        del self.disappeared[track_id]

    def update(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        """Match `detections` to known tracks; returns [(track_id, detection)]."""
        # No detections: age every track and drop the stale ones.
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
            return []

        # First frame(s): everything is new.
        if not self.objects:
            for det in detections:
                self._register(det)
            return [(d.track_id, d) for d in detections]

        existing_ids = list(self.objects)
        existing_centroids = [self.objects[t].centroid for t in existing_ids]
        new_centroids = [d.centroid for d in detections]

        # Greedy nearest-neighbour matching over all (existing, new) pairs.
        pairs = []
        for ei, (ex, ey) in enumerate(existing_centroids):
            for ni, (nx, ny) in enumerate(new_centroids):
                dist = math.hypot(ex - nx, ey - ny)
                pairs.append((dist, ei, ni))
        pairs.sort(key=lambda p: p[0])

        used_existing: set[int] = set()
        used_new: set[int] = set()
        matched: list[tuple[int, Detection]] = []
        for dist, ei, ni in pairs:
            if dist > self.max_distance:
                break
            if ei in used_existing or ni in used_new:
                continue
            tid = existing_ids[ei]
            det = detections[ni]
            det.track_id = tid
            self.objects[tid] = det
            self.disappeared[tid] = 0
            used_existing.add(ei)
            used_new.add(ni)
            matched.append((tid, det))

        # Tracks that were not matched age; drop if gone too long.
        for ei, tid in enumerate(existing_ids):
            if ei not in used_existing:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        # Unmatched detections are new people.
        for ni, det in enumerate(detections):
            if ni not in used_new:
                self._register(det)
                matched.append((det.track_id, det))

        return matched


# ---------------------------------------------------------------------------
# Line-crossing counter
# ---------------------------------------------------------------------------
class LineCounter:
    """Counts direction-aware crossings of a vertical line.

    A track is counted only when its centroid moves from one side of the line
    to the other, outside a small deadband around the line — so a person
    standing on the line cannot be counted repeatedly.
    """

    def __init__(self, line_x: int, margin: int = 6):
        self.line_x = line_x
        self.margin = margin
        self.count_in = 0
        self.count_out = 0
        self._side: dict[int, int] = {}       # track_id -> -1 left / +1 right
        self._last_seen: dict[int, int] = {}   # track_id -> frame index

    @property
    def occupancy(self) -> int:
        """People currently inside: IN - OUT (never negative)."""
        return max(0, self.count_in - self.count_out)

    def update(self, track_id: int, cx: int, frame_index: int) -> str | None:
        """Feed one centroid; returns "IN"/"OUT" if this caused a count."""
        if cx < self.line_x - self.margin:
            side = -1
        elif cx > self.line_x + self.margin:
            side = +1
        else:
            return None  # inside the deadband — no side change

        self._last_seen[track_id] = frame_index
        previous = self._side.get(track_id)
        self._side[track_id] = side
        if previous is None or previous == side:
            return None
        if previous == -1 and side == +1:
            self.count_in += 1
            return "IN"
        self.count_out += 1
        return "OUT"

    def retain_recent(self, frame_index: int, keep_frames: int = 600) -> None:
        """Forget tracks not seen for a while (keeps dicts small on long runs)."""
        stale = [tid for tid, seen in self._last_seen.items()
                 if frame_index - seen > keep_frames]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._side.pop(tid, None)


# ---------------------------------------------------------------------------
# Overlay — shared by real and simulation modes
# ---------------------------------------------------------------------------
def _draw_dashed_line(frame: np.ndarray, x: int, y0: int, y1: int,
                      color: tuple[int, int, int]) -> None:
    dash, gap = 18, 12
    y = y0
    while y < y1:
        cv2.line(frame, (x, y), (x, min(y + dash, y1)), color, 3)
        y += dash + gap


def draw_overlay(frame: np.ndarray,
                 tracked: list[tuple[int, Detection]],
                 counter: LineCounter,
                 fps: float,
                 last_event: tuple[str, float] | None = None) -> np.ndarray:
    """Draw counting line, boxes+IDs, HUD counters and FPS onto `frame`."""
    h, w = frame.shape[:2]
    line_x = counter.line_x

    # Counting line + label.
    _draw_dashed_line(frame, line_x, 0, h, COL_LINE)
    cv2.putText(frame, "ENTRANCE LINE", (line_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_LINE, 2)

    # Direction hints.
    cv2.putText(frame, "IN >>", (line_x + 12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_IN, 2)
    in_label = "<< OUT"
    (tw, _), _ = cv2.getTextSize(in_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, in_label, (line_x - 12 - tw, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_OUT, 2)

    # Boxes + IDs.
    for tid, det in tracked:
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), COL_BOX, 2)
        cx, cy = det.centroid
        cv2.circle(frame, (cx, cy), 4, COL_LINE, -1)
        label = f"ID {tid}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(det.y1, lh + 6)
        cv2.rectangle(frame, (det.x1, ly - lh - 6), (det.x1 + lw + 6, ly),
                      COL_BOX, -1)
        cv2.putText(frame, label, (det.x1 + 3, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # HUD panel (top-left).
    panel_w, panel_h = 296, 148
    panel = frame[0:panel_h, 0:panel_w].copy()
    frame[0:panel_h, 0:panel_w] = cv2.addWeighted(panel, 0.35,
                                                  np.zeros_like(panel), 0.65, 0)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), COL_LINE, 1)
    cv2.putText(frame, "MALL FOOTFALL COUNTER", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL_TEXT, 2)
    cv2.putText(frame, f"IN:        {counter.count_in}", (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_IN, 2)
    cv2.putText(frame, f"OUT:       {counter.count_out}", (12, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_OUT, 2)
    cv2.putText(frame, f"INSIDE NOW: {counter.occupancy}", (12, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)

    # FPS (top-right).
    fps_text = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, fps_text, (w - fw - 14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)

    # Brief flash when someone is counted.
    if last_event is not None:
        kind, age = last_event
        if age < 0.8:
            text = ">> IN +1" if kind == "IN" else "OUT +1 <<"
            color = COL_IN if kind == "IN" else COL_OUT
            (ew, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(frame, text, ((w - ew) // 2, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    return frame


# ---------------------------------------------------------------------------
# Real-mode detector — YOLOv8 (imported lazily; only needed for --source)
# ---------------------------------------------------------------------------
class YOLOPersonDetector:
    """Detects people with YOLOv8.

    With ``use_bytetrack=True`` (default) it calls ``model.track`` so each
    detection arrives with a ByteTrack ID. With ``False`` it uses plain
    ``model.predict`` and the shared CentroidTracker assigns the IDs.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4,
                 use_bytetrack: bool = True):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Real mode needs the 'ultralytics' package. Install it with "
                "'pip install ultralytics', or run simulation mode "
                "(--simulate), which only needs opencv-python + numpy."
            ) from exc
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.use_bytetrack = use_bytetrack
        self._fallback = CentroidTracker()  # if ByteTrack returns no IDs

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.use_bytetrack:
            results = self.model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                classes=[PERSON_CLASS_ID], conf=self.confidence, verbose=False)
        else:
            results = self.model.predict(
                frame, classes=[PERSON_CLASS_ID], conf=self.confidence,
                verbose=False)

        boxes = results[0].boxes
        detections: list[Detection] = []
        if boxes is None or len(boxes) == 0:
            if self.use_bytetrack:
                self._fallback.update([])
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = (boxes.conf.cpu().numpy() if boxes.conf is not None
                 else np.ones(len(xyxy)))
        ids = None
        if self.use_bytetrack and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            det = Detection(int(x1), int(y1), int(x2), int(y2),
                            confidence=float(confs[i]))
            if ids is not None:
                det.track_id = int(ids[i])
            detections.append(det)

        # ByteTrack occasionally returns boxes without IDs (e.g. first frame):
        # fall back to the centroid tracker for those frames.
        if self.use_bytetrack and ids is None:
            self._fallback.update(detections)
        return detections


# ---------------------------------------------------------------------------
# Simulation — a synthetic mall entrance rendered with OpenCV/NumPy
# ---------------------------------------------------------------------------
_SHIRTS = [(180, 90, 60), (60, 140, 200), (90, 180, 90), (200, 120, 160),
           (120, 120, 220), (70, 180, 220), (160, 90, 200), (100, 170, 140)]
_PANTS = [(60, 50, 40), (50, 50, 60), (70, 60, 50), (45, 45, 45)]
_SKIN = [(180, 200, 235), (150, 180, 220), (120, 150, 200), (200, 215, 240)]
_HAIR = [(40, 35, 30), (60, 45, 35), (25, 25, 25), (80, 60, 45)]


@dataclass
class SimPerson:
    """One simulated shopper walking through the entrance."""

    x: float            # horizontal position of the body's centre (px)
    baseline_y: int     # y of the feet (their "lane" in the corridor)
    direction: int      # +1 walking right (entering) / -1 walking left (leaving)
    speed: float        # px per frame
    scale: float        # size scale (lanes nearer the camera are bigger)
    shirt: tuple[int, int, int]
    pants: tuple[int, int, int]
    skin: tuple[int, int, int]
    hair: tuple[int, int, int]
    phase: float = 0.0  # walk-cycle phase

    @property
    def height(self) -> float:
        return 122 * self.scale

    @property
    def width(self) -> float:
        return 46 * self.scale

    def detection(self) -> Detection:
        return Detection(
            x1=int(self.x - self.width / 2), y1=int(self.baseline_y - self.height),
            x2=int(self.x + self.width / 2), y2=int(self.baseline_y),
            confidence=1.0)


class MallScene:
    """Renders a stylised side-view mall entrance and its shoppers.

    The scene is the simulation's "camera + detector" in one: each step moves
    the shoppers, draws the frame, and returns the ground-truth bounding boxes
    as Detections for the shared pipeline.
    """

    LANES = (516, 586, 656)  # feet-y of the three walking lanes

    def __init__(self, width: int = 1280, height: int = 720, seed: int = 7,
                 max_people: int = 7):
        self.width = width
        self.height = height
        self.max_people = max_people
        self.rng = random.Random(seed)
        self.people: list[SimPerson] = []
        self._spawn_cooldown = 0
        self._background = self._render_background()
        # Start with a few shoppers already mid-scene so the demo is alive
        # from the first second (two entering, one leaving).
        self._add_person(x=width * 0.16, direction=+1, lane=1)
        self._add_person(x=width * 0.36, direction=+1, lane=2)
        self._add_person(x=width * 0.82, direction=-1, lane=0)

    # -- people management --------------------------------------------------
    def _add_person(self, x: float, direction: int, lane: int) -> None:
        baseline = self.LANES[lane] + self.rng.randint(-8, 8)
        scale = 0.85 + (baseline - 500) * 0.0016
        self.people.append(SimPerson(
            x=x, baseline_y=baseline, direction=direction,
            speed=self.rng.uniform(2.3, 3.6) * scale,
            scale=scale,
            shirt=self.rng.choice(_SHIRTS), pants=self.rng.choice(_PANTS),
            skin=self.rng.choice(_SKIN), hair=self.rng.choice(_HAIR),
            phase=self.rng.uniform(0, math.tau)))

    def _maybe_spawn(self) -> None:
        if self._spawn_cooldown > 0:
            self._spawn_cooldown -= 1
            return
        if len(self.people) >= self.max_people:
            return
        direction = +1 if self.rng.random() < 0.55 else -1
        lane = self.rng.randrange(len(self.LANES))
        x = -50.0 if direction == +1 else self.width + 50.0
        # Avoid spawning on top of someone already at the edge in this lane.
        for p in self.people:
            if p.baseline_y - self.LANES[lane] in range(-16, 17) \
                    and abs(p.x - x) < 110:
                return
        self._add_person(x=x, direction=direction, lane=lane)
        self._spawn_cooldown = self.rng.randint(30, 85)

    # -- rendering ------------------------------------------------------------
    def _render_background(self) -> np.ndarray:
        w, h = self.width, self.height
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Wall (top) and floor (bottom).
        img[0:440, :] = (196, 188, 178)
        img[440:h, :] = (150, 140, 128)
        for y in range(440, h, 46):  # floor tile lines
            cv2.line(img, (0, y), (w, y), (128, 118, 106), 2)
        for x in range(0, w, 160):
            cv2.line(img, (x, 440), (x - 60, h), (128, 118, 106), 2)

        # Storefronts along the wall.
        shops = [("CAFE", (96, 130, 170)), ("BOOKS", (120, 100, 150)),
                 ("FASHION", (90, 150, 130)), ("TECH", (140, 120, 90)),
                 ("SPORTS", (110, 140, 160))]
        shop_w = w // len(shops)
        for i, (name, glass) in enumerate(shops):
            x0 = i * shop_w + 14
            x1 = (i + 1) * shop_w - 14
            cv2.rectangle(img, (x0, 150), (x1, 440), glass, -1)
            cv2.rectangle(img, (x0, 150), (x1, 440), (90, 84, 78), 3)
            cv2.rectangle(img, (x0, 96), (x1, 150), (70, 64, 60), -1)
            cv2.putText(img, name, (x0 + 16, 134), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (240, 240, 240), 2)
            cv2.line(img, (x0 + (x1 - x0) // 2, 150),
                     (x0 + (x1 - x0) // 2, 440), (90, 84, 78), 3)

        # Zone labels on the floor.
        cv2.putText(img, "OUTSIDE", (36, h - 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (105, 96, 86), 2)
        cv2.putText(img, "INSIDE THE MALL", (w - 300, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (105, 96, 86), 2)
        return img

    def _draw_doorway(self, frame: np.ndarray, line_x: int) -> None:
        """Door frame around the counting line (drawn above the background)."""
        cv2.rectangle(frame, (line_x - 30, 128), (line_x - 14, 668),
                      (60, 55, 52), -1)
        cv2.rectangle(frame, (line_x + 14, 128), (line_x + 30, 668),
                      (60, 55, 52), -1)
        cv2.rectangle(frame, (line_x - 120, 92), (line_x + 120, 128),
                      (60, 55, 52), -1)
        cv2.putText(frame, "ENTRANCE", (line_x - 82, 119),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)

    def _draw_person(self, frame: np.ndarray, p: SimPerson) -> None:
        s = p.scale
        cx = int(p.x)
        feet_y = p.baseline_y
        bob = abs(math.sin(p.phase)) * 3.0 * s
        head_r = 11 * s
        head_cy = feet_y - p.height + head_r + bob
        hip_y = feet_y - p.height * 0.48
        shoulder_y = head_cy + head_r + 4 * s

        # Soft shadow.
        cv2.ellipse(frame, (cx, feet_y + 5), (int(24 * s), int(6 * s)),
                    0, 0, 360, (110, 100, 92), -1)

        # Legs (walk cycle).
        stride = 16 * s
        lift = 5 * s
        for phase_off, sign in ((0.0, 1), (math.pi, -1)):
            f = math.sin(p.phase + phase_off)
            foot = (int(cx + f * stride),
                    int(feet_y - max(0.0, math.cos(p.phase + phase_off)) * lift))
            knee = (int(cx + f * stride * 0.5 + p.direction * 3),
                    int((hip_y + foot[1]) / 2))
            cv2.line(frame, (cx, int(hip_y)), knee, p.pants, max(2, int(5 * s)))
            cv2.line(frame, knee, foot, p.pants, max(2, int(5 * s)))

        # Torso.
        cv2.rectangle(frame, (int(cx - 9 * s), int(shoulder_y)),
                      (int(cx + 9 * s), int(hip_y)), p.shirt, -1)

        # Arms (swing opposite to the legs).
        for phase_off in (math.pi, 0.0):
            f = math.sin(p.phase + phase_off)
            hand = (int(cx + f * stride * 0.7), int(hip_y - 4 * s))
            cv2.line(frame, (cx, int(shoulder_y + 2)), hand, p.shirt,
                     max(2, int(4 * s)))

        # Head + hair.
        cv2.circle(frame, (cx, int(head_cy - 2 * s)), int(head_r), p.hair, -1)
        cv2.circle(frame, (cx, int(head_cy + 1 * s)), int(head_r * 0.85),
                   p.skin, -1)

    # -- main step --------------------------------------------------------------
    def step(self, line_x: int) -> tuple[np.ndarray, list[Detection]]:
        """Advance the scene one frame; returns (frame, detections)."""
        self._maybe_spawn()
        for p in self.people:
            p.x += p.direction * p.speed
            p.phase += p.speed * 0.085
        self.people = [p for p in self.people
                       if -80 < p.x < self.width + 80]

        frame = self._background.copy()
        self._draw_doorway(frame, line_x)
        for p in sorted(self.people, key=lambda q: q.baseline_y):
            self._draw_person(frame, p)
        detections = [p.detection() for p in self.people]
        return frame, detections


# ---------------------------------------------------------------------------
# Shared pipeline: track -> count -> annotate
# ---------------------------------------------------------------------------
class CountingPipeline:
    """Runs tracker + counter + overlay identically for both modes."""

    def __init__(self, line_x: int):
        self.centroid_tracker = CentroidTracker()
        self.counter = LineCounter(line_x)
        self._fps = 0.0
        self._last_event: tuple[str, float] | None = None

    def process(self, frame: np.ndarray, detections: list[Detection],
                frame_index: int, now: float, dt: float) -> np.ndarray:
        # Smooth FPS estimate.
        if dt > 0:
            instant = 1.0 / dt
            self._fps = instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant

        # Detections that already carry an ID (ByteTrack) pass straight
        # through; everything else goes through the shared centroid tracker.
        if detections and all(d.track_id is not None for d in detections):
            tracked = [(d.track_id, d) for d in detections]
            self.centroid_tracker.update([])  # keep its state ageing cleanly
        else:
            tracked = self.centroid_tracker.update(detections)

        for tid, det in tracked:
            event = self.counter.update(tid, det.centroid[0], frame_index)
            if event:
                self._last_event = (event, now)
        self.counter.retain_recent(frame_index)

        last = None
        if self._last_event is not None:
            kind, when = self._last_event
            last = (kind, now - when)
        return draw_overlay(frame, tracked, self.counter, self._fps, last)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def make_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output video '{path}' for writing "
            "(the mp4v codec may be unavailable in this OpenCV build).")
    return writer


def _destroy_windows() -> None:
    """Close preview windows; a no-op on headless OpenCV builds."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def show_frame(frame: np.ndarray, title: str) -> bool:
    """Show a frame; returns False when the user pressed q / Esc.

    Silently disables itself on headless OpenCV builds (no GUI support).
    """
    try:
        cv2.imshow(title, frame)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)
    except cv2.error:
        return True


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_simulation(args: argparse.Namespace) -> int:
    scene = MallScene(width=args.width, height=args.height, seed=args.seed,
                      max_people=args.max_people)
    line_x = int(args.line_x * args.width)
    pipeline = CountingPipeline(line_x)

    writer = None
    if args.output:
        writer = make_writer(args.output, args.fps, (args.width, args.height))

    total_frames = int(args.duration * args.fps) if args.duration > 0 else None
    print(f"[simulate] rendering {args.width}x{args.height} @ {args.fps} fps"
          + (f", {args.duration:.0f}s" if total_frames else ", until 'q'"))
    frame_index = 0
    prev = time.perf_counter()
    try:
        while total_frames is None or frame_index < total_frames:
            tick = time.perf_counter()
            frame, detections = scene.step(line_x)
            now = time.perf_counter()
            annotated = pipeline.process(frame, detections, frame_index,
                                         now, now - prev)
            prev = now
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                if not show_frame(annotated, "Mall People Tracker (simulation)"):
                    break
            frame_index += 1
            # Pace the loop to the target frame rate (real-time playback).
            elapsed = time.perf_counter() - tick
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        _destroy_windows()

    c = pipeline.counter
    print(f"[simulate] done: {frame_index} frames | "
          f"IN={c.count_in} OUT={c.count_out} INSIDE={c.occupancy}")
    if args.output:
        print(f"[simulate] video saved to {args.output}")
    return 0


def run_real(args: argparse.Namespace) -> int:
    source: int | str = 0 if str(args.source) == "0" else args.source
    if isinstance(source, str):
        import os
        if not os.path.exists(source):
            print(f"error: video file not found: {source}", file=sys.stderr)
            return 2

    try:
        detector = YOLOPersonDetector(
            model_name=args.model, confidence=args.conf,
            use_bytetrack=(args.tracker == "bytetrack"))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"error: could not open source: {args.source}", file=sys.stderr)
        return 2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    line_x = int(args.line_x * width)
    pipeline = CountingPipeline(line_x)

    writer = None
    if args.output:
        writer = make_writer(args.output, src_fps, (width, height))

    print(f"[real] source={args.source} {width}x{height} — press 'q' to stop")
    frame_index = 0
    prev = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            now = time.perf_counter()
            annotated = pipeline.process(frame, detections, frame_index,
                                         now, now - prev)
            prev = now
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                if not show_frame(annotated, "Mall People Tracker"):
                    break
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        _destroy_windows()

    c = pipeline.counter
    print(f"[real] done: {frame_index} frames | "
          f"IN={c.count_in} OUT={c.count_out} INSIDE={c.occupancy}")
    if args.output:
        print(f"[real] video saved to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mall People Tracker — count people entering/leaving "
                    "with YOLOv8 + OpenCV, or run the built-in simulation.")
    parser.add_argument("--source", default=None,
                        help="video file path, or 0 for the webcam (real mode)")
    parser.add_argument("--simulate", action="store_true",
                        help="run the synthetic mall-entrance simulation "
                             "(needs only opencv-python + numpy)")
    parser.add_argument("--output", default=None,
                        help="path to save the annotated output video (.mp4)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 model for real mode (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="detection confidence threshold (default: 0.4)")
    parser.add_argument("--tracker", choices=("bytetrack", "centroid"),
                        default="bytetrack",
                        help="real-mode ID source: ByteTrack via model.track "
                             "(default) or the built-in centroid tracker")
    parser.add_argument("--line-x", type=float, default=0.5,
                        help="counting-line position as a fraction of the "
                             "frame width, 0-1 (default: 0.5)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="simulation length in seconds (default: 30; "
                             "0 = run until 'q' when a display is available)")
    parser.add_argument("--fps", type=int, default=30,
                        help="simulation frame rate (default: 30)")
    parser.add_argument("--width", type=int, default=1280,
                        help="simulation frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="simulation frame height (default: 720)")
    parser.add_argument("--max-people", type=int, default=7,
                        help="maximum simultaneous shoppers in the simulation")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed for the simulation (default: 7)")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a preview window (headless use)")
    args = parser.parse_args(argv)

    if not 0.0 < args.line_x < 1.0:
        parser.error("--line-x must be between 0 and 1 (exclusive)")
    if not args.simulate and args.source is None:
        parser.error("choose a mode: --simulate, or --source <video path|0>")
    if args.simulate and args.source is not None:
        parser.error("--simulate and --source cannot be used together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.simulate:
        return run_simulation(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
