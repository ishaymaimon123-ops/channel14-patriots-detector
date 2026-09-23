"""Visual Channel 14 / Patriots break detection and separate ad segmentation.

The supplied screenshot is the sole positive calibration example. Scores are
heuristics; use real program and ad videos to tune thresholds before production.
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "assets/reference/patriots_commercial_reference.png"
LEFT_TEMPLATE = ROOT / "assets/detectors/patriots_break_box.png"
RIGHT_TEMPLATE = ROOT / "assets/detectors/channel14_commercial_marker.png"
CONFIG = ROOT / "config.json"
LOCAL_BIN = ROOT / "work/bin"
if LOCAL_BIN.is_dir():
    os.environ["PATH"] = str(LOCAL_BIN) + os.pathsep + os.environ.get("PATH", "")
TIMER_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[:：]\s*([0-5]\d)(?!\d)")
HEBREW_RE = re.compile(r"[^\u0590-\u05ff]")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def blue_mask(a: np.ndarray) -> np.ndarray:
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (b > 0.32) & (b > r * 1.37) & (b > g * 1.20) & ((b - r) > 0.17)


def white_mask(a: np.ndarray) -> np.ndarray:
    hi = a.max(axis=2)
    lo = a.min(axis=2)
    lum = a.mean(axis=2)
    # Local brightness and low chroma keep the spiral visible over changing ads.
    floor = max(0.62, float(np.percentile(lum, 25)) + 0.26)
    return (lum > floor) & ((hi - lo) < 0.22)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    n = int(a.sum()) + int(b.sum())
    return 2.0 * float(np.logical_and(a, b).sum()) / n if n else 0.0


def edge_map(a: np.ndarray) -> np.ndarray:
    lum = a.mean(axis=2)
    dx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    dy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    return np.clip((dx + dy) * 4.0, 0, 1)


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return clamp((float(np.dot(a, b)) / den + 1) / 2) if den > 1e-7 else 0.0


@dataclass
class FrameResult:
    timestamp: float
    confidence: float
    upper_left_box_score: float
    upper_right_symbol_score: float
    patriots_text_score: float
    return_soon_text_score: float
    timer_score: float
    ocr_text: str
    detected_timer: str | None
    classification: str = "PROGRAM"
    countdown_consistent: bool = False
    thumbnail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class VisualDetector:
    def __init__(self, config: dict[str, Any], *, ocr: bool = True):
        self.config = config
        self.left = rgb(Image.open(LEFT_TEMPLATE), (160, 84))
        self.right = rgb(Image.open(RIGHT_TEMPLATE), (80, 84))
        self.left_blue = blue_mask(self.left)
        self.left_white = white_mask(self.left)
        self.right_white = white_mask(self.right)
        self.left_edge = edge_map(self.left)
        self.right_edge = edge_map(self.right)
        self.ocr_enabled = ocr and shutil.which("tesseract") is not None

    def _left_score(self, candidate: Image.Image) -> float:
        a = rgb(candidate, (160, 84))
        blue = blue_mask(a)
        # The timer changes every second: blue layout is compared across the
        # whole graphic, while edge detail is taken only from its fixed bands.
        blue_similarity = dice(blue, self.left_blue)
        coverage = 1 - abs(float(blue.mean()) - float(self.left_blue.mean())) / max(float(self.left_blue.mean()), 0.01)
        static = np.zeros((84, 160), dtype=bool)
        static[:23] = True
        static[65:] = True
        edge_similarity = correlation(edge_map(a), self.left_edge, static)
        # A blue television backdrop can match the colour mask by itself.
        # Require some of the white lettering/countdown layout as well.
        white = white_mask(a)
        white_presence = clamp(float(white.mean()) / max(float(self.left_white.mean()), .01))
        white_layout = dice(white, self.left_white)
        return clamp(0.52 * blue_similarity + 0.10 * clamp(coverage) +
                     0.12 * edge_similarity + 0.13 * white_presence +
                     0.13 * white_layout)

    def _right_score(self, candidate: Image.Image) -> float:
        a = rgb(candidate, (80, 84))
        white = white_mask(a)
        if white.mean() < 0.035:
            return 0.0
        shape = dice(white, self.right_white)
        edge = correlation(edge_map(a), self.right_edge)
        fill = 1 - abs(float(white.mean()) - float(self.right_white.mean())) / max(float(self.right_white.mean()), 0.01)
        return clamp(0.72 * shape + 0.18 * edge + 0.10 * clamp(fill))

    def _search(self, image: Image.Image, key: str, score_fn) -> tuple[float, Image.Image]:
        w, h = image.size
        x1, y1, x2, y2 = self.config[key]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        ox = float(self.config["search_offset_x"])
        oy = float(self.config["search_offset_y"])
        # Exact reference position is included. Every coordinate scales with
        # the input frame, so this works for different stream resolutions.
        shifts = (0, -1, 1, -0.5, 0.5)
        best, best_crop = -1.0, None
        for scale in self.config["scales"]:
            for sx in shifts:
                for sy in shifts:
                    px, py = cx + sx * ox, cy + sy * oy
                    box = (round((px - bw * scale / 2) * w), round((py - bh * scale / 2) * h),
                           round((px + bw * scale / 2) * w), round((py + bh * scale / 2) * h))
                    if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h or box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    crop = image.crop(box)
                    score = score_fn(crop)
                    if score > best:
                        best, best_crop = score, crop
                    if best > 0.995:
                        return best, best_crop
        assert best_crop is not None
        return max(0.0, best), best_crop

    def _ocr(self, crop: Image.Image) -> tuple[str, float, float, float, str | None]:
        if not self.ocr_enabled:
            return "", 0.0, 0.0, 0.0, None
        # The labels are static, but the timer changes. Reading three isolated
        # lines is more reliable than OCR over the full tiny broadcast graphic.
        try:
            with tempfile.TemporaryDirectory() as d:
                w, h = crop.size
                regions = {
                    "title": ((.05, 0, .95, .27), "heb", False),
                    "timer": ((.07, .25, .93, .78), "eng", False),
                    "return": ((.19, .75, .82, 1), "heb", True),
                }
                lines = {}
                for name, (fractions, language, threshold) in regions.items():
                    x1, y1, x2, y2 = fractions
                    part = crop.crop((round(x1*w), round(y1*h), round(x2*w), round(y2*h)))
                    part = part.resize((part.width*5, part.height*5), Image.Resampling.BILINEAR)
                    if threshold:
                        part = ImageOps.grayscale(part).point(lambda p: 255 if p > 150 else 0)
                    p = Path(d) / f"{name}.png"
                    part.save(p)
                    proc = subprocess.run(["tesseract", str(p), "stdout", "-l", language, "--psm", "7"],
                                          capture_output=True, text=True, timeout=10)
                    lines[name] = proc.stdout.strip() if proc.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            return "", 0.0, 0.0, 0.0, None
        title = HEBREW_RE.sub("", lines["title"])
        return_soon = HEBREW_RE.sub("", lines["return"])
        patriot = (1.0 if "הפטריוטים" in title else difflib.SequenceMatcher(None, title, "הפטריוטים").ratio()) if len(title) >= 4 else 0.0
        soon = (1.0 if "מידנשוב" in return_soon else difflib.SequenceMatcher(None, return_soon, "מידנשוב").ratio()) if len(return_soon) >= 3 else 0.0
        match = TIMER_RE.search(lines["timer"])
        timer = f"{int(match[1]):02d}:{int(match[2]):02d}" if match else None
        return " | ".join(v for v in lines.values() if v), clamp(patriot), clamp(soon), 1.0 if timer else 0.0, timer

    def _fixed_crop(self, image: Image.Image, key: str) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = self.config[key]
        return image.crop((round(x1*w), round(y1*h), round(x2*w), round(y2*h)))

    def score_frame(self, image: Image.Image, timestamp: float = 0.0, *,
                    do_ocr: bool = True, search: bool = True) -> FrameResult:
        image = image.convert("RGB")
        if search:
            left, left_crop = self._search(image, "left_box", self._left_score)
            right, _ = self._search(image, "right_marker", self._right_score)
        else:
            left_crop = self._fixed_crop(image, "left_box")
            left = self._left_score(left_crop)
            right = self._right_score(self._fixed_crop(image, "right_marker"))
        # Video analysis schedules OCR periodically. Visual matches are checked
        # on every sampled frame, while OCR remains supporting evidence.
        ocr_text, patriot, soon, timer_score, timer = self._ocr(left_crop) if left > 0.32 and do_ocr else ("", 0., 0., 0., None)
        weights = self.config["weights"]
        support = max(timer_score, patriot, soon)
        both = weights["both_left"] * left + weights["both_right"] * right + weights["both_support"] * support
        left_only = weights["left_only_visual"] * left + weights["left_only_timer"] * timer_score + weights["left_only_text"] * max(patriot, soon)
        right_only = weights["right_only"] * right
        ocr_bundle = weights["ocr_patriots"] * patriot + weights["ocr_return_soon"] * soon + weights["ocr_timer"] * timer_score
        confidence = clamp(max(both, left_only, right_only, ocr_bundle))
        frame_class = "COMMERCIAL" if confidence >= self.config["start_threshold"] else "PROGRAM"
        return FrameResult(timestamp, confidence, left, right, patriot, soon, timer_score, ocr_text, timer,
                           classification=frame_class)


@dataclass
class Break:
    start: float
    end: float
    ads: list[dict[str, Any]] = field(default_factory=list)


class TemporalTracker:
    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        self.active = False
        self.good: list[FrameResult] = []
        self.bad: list[FrameResult] = []
        self.previous: FrameResult | None = None
        self.last_timer: FrameResult | None = None
        self.open_start: float | None = None
        self.breaks: list[Break] = []

    def add(self, result: FrameResult) -> None:
        if result.detected_timer and self.last_timer:
            prev = self._seconds(self.last_timer.detected_timer)
            now = self._seconds(result.detected_timer)
            dt = result.timestamp - self.last_timer.timestamp
            delta = prev - now
            if dt > 0 and -0.5 <= delta <= dt + 1.5 and abs(delta - dt) <= 1.5:
                result.countdown_consistent = True
                result.confidence = clamp(result.confidence + float(self.cfg["countdown_bonus"]))
        if result.detected_timer:
            self.last_timer = result
        self.previous = result
        if not self.active:
            if result.confidence >= self.cfg["start_threshold"]:
                self.good.append(result)
                if len(self.good) >= self.cfg["start_positive_samples"]:
                    self.open_start = self.good[0].timestamp
                    self.active = True
                    self.bad.clear()
                    result.classification = "COMMERCIAL"
            else:
                self.good.clear()
        else:
            result.classification = "COMMERCIAL"
            if result.confidence < self.cfg["end_threshold"]:
                self.bad.append(result)
                if len(self.bad) >= self.cfg["end_negative_samples"]:
                    self.breaks.append(Break(self.open_start or 0, self.bad[0].timestamp))
                    self.active = False
                    self.open_start = None
                    self.good.clear()
                    result.classification = "PROGRAM"
            else:
                self.bad.clear()

    def finish(self, duration: float) -> list[Break]:
        if self.active and self.open_start is not None:
            self.breaks.append(Break(self.open_start, duration))
            self.active = False
        return self.breaks

    @staticmethod
    def _seconds(timer: str) -> int:
        mm, ss = timer.split(":")
        return int(mm) * 60 + int(ss)


def detect_breaks(results: list[FrameResult], duration: float, cfg: dict[str, Any]) -> list[Break]:
    """Group persistent corner graphics into whole breaks, bridging short animation gaps."""
    if not results:
        return []
    ordered = sorted(results, key=lambda result: result.timestamp)
    joint_left = float(cfg.get("visual_joint_left", .65))
    joint_right = float(cfg.get("visual_joint_right", .69))
    strong_left = float(cfg.get("visual_strong_left", .78))
    strong_right = float(cfg.get("visual_strong_right", .88))
    positive = [result for result in ordered if
                (result.upper_left_box_score >= joint_left and result.upper_right_symbol_score >= joint_right)
                or result.upper_left_box_score >= strong_left
                or result.upper_right_symbol_score >= strong_right
                or (result.detected_timer and result.upper_left_box_score >= .55)]
    if not positive:
        return []
    max_gap = float(cfg.get("max_positive_gap_seconds", 25))
    groups: list[list[FrameResult]] = []
    current = [positive[0]]
    for result in positive[1:]:
        gap = result.timestamp - current[-1].timestamp
        # A large temporary loss of the spiral is common in an ad, while a
        # genuine return to the programme also loses the blue layout.
        between = [frame.upper_left_box_score for frame in ordered
                   if current[-1].timestamp < frame.timestamp < result.timestamp] if gap > 12 else []
        bridge = gap <= max_gap and (gap <= 12 or (between and float(np.median(between)) >= .62))
        if not bridge:
            groups.append(current)
            current = []
        current.append(result)
    groups.append(current)

    minimum_duration = min(float(cfg.get("minimum_break_seconds", 45)), duration * .4)
    minimum_density = float(cfg.get("minimum_positive_density", .18))
    lead = float(cfg.get("boundary_lead_seconds", 7.5))
    trail = float(cfg.get("boundary_trail_seconds", 3))
    if duration < 30:
        trail = min(trail, .75)
    breaks: list[Break] = []
    for group in groups:
        local_frames = [frame for frame in ordered
                        if group[0].timestamp <= frame.timestamp <= group[-1].timestamp]
        local_times = [frame.timestamp for frame in local_frames]
        local_steps = [right - left for left, right in zip(local_times, local_times[1:])
                       if 0 < right - left <= 2]
        step = float(np.median(local_steps)) if local_steps else float(cfg.get("sample_interval_seconds", .75))
        span = group[-1].timestamp - group[0].timestamp + step
        density = len(group) / max(span / step, 1)
        support = (any(frame.upper_left_box_score >= strong_left or
                       frame.upper_right_symbol_score >= strong_right or frame.detected_timer
                       for frame in group)
                   or sum(frame.upper_left_box_score >= joint_left and
                          frame.upper_right_symbol_score >= joint_right
                          for frame in local_frames) / max(len(local_frames), 1) >= .4)
        if span < minimum_duration or density < minimum_density or len(group) < 3 or not support:
            continue
        # Ignore an isolated logo-like match ahead of the sustained overlay.
        needed = min(len(group), max(5, int(np.ceil(6 / step))))
        anchor = next((result.timestamp for result in group
                       if sum(result.timestamp <= other.timestamp <= result.timestamp + 12
                              for other in group) >= needed), group[0].timestamp)
        start = max(0., anchor - lead)
        end = min(duration, group[-1].timestamp + trail)
        if breaks and start <= breaks[-1].end:
            breaks[-1].end = max(breaks[-1].end, end)
        else:
            breaks.append(Break(start, end))
    return breaks


def ffprobe_duration(video: str) -> float:
    if shutil.which("ffprobe"):
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=noprint_wrappers=1:nokey=1", video], capture_output=True, text=True, check=True)
        return float(proc.stdout.strip())
    # The bundled imageio-ffmpeg executable has ffmpeg but not ffprobe.
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-i", video], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not match:
        raise RuntimeError("Could not determine video duration")
    return int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])


def extract_frames(video: str, out: Path, fps: float, start: float = 0,
                   end: float | None = None, *, coarse: bool = False) -> list[tuple[float, Path]]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if coarse:
        # The first pass only locates likely breaks. HLS keyframes retain the
        # corner overlays while avoiding a full decode of every source frame.
        cmd += ["-skip_frame", "nokey"]
    cmd += ["-i", video]
    if end is not None:
        cmd += ["-t", str(max(0, end - start))]
    filters = f"fps={fps:.8f}" + (",scale=640:-2" if coarse else "")
    cmd += ["-vf", filters, "-q:v", "5" if coarse else "3",
            "-start_number", "0", str(out / "%06d.jpg")]
    subprocess.run(cmd, check=True)
    return [(start + i / fps, p) for i, p in enumerate(sorted(out.glob("*.jpg")))]


def content_features(image: Image.Image) -> tuple[np.ndarray, np.ndarray, bool]:
    # Omit the top 23%, so persistent break graphics never create ad cuts.
    w, h = image.size
    a = np.asarray(image.crop((int(.20*w), int(.23*h), int(.85*w), int(.90*h))).convert("RGB").resize((96, 54)), dtype=np.float32) / 255
    gray = a.mean(axis=2)
    hist = np.concatenate([np.histogram(a[:, :, c], bins=16, range=(0, 1), density=True)[0] for c in range(3)])
    hist /= max(hist.sum(), 1e-6)
    return gray, hist, bool(gray.mean() < .075)


def silence_midpoints(video: str, start: float, end: float) -> list[float]:
    cmd = ["ffmpeg", "-hide_banner", "-ss", str(start), "-i", video, "-t", str(end-start),
           "-af", "silencedetect=noise=-35dB:d=0.15", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(v) for v in re.findall(r"silence_start:\s*([\d.]+)", proc.stderr)]
    ends = [float(v) for v in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]
    return [start + (a+b)/2 for a, b in zip(starts, ends) if 0 <= a <= b]


def split_ads(video: str, a_break: Break, frames_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    frames = extract_frames(video, frames_dir, float(cfg["ad_analysis_fps"]), a_break.start, a_break.end)
    if len(frames) < 2:
        return [{"start": a_break.start, "end": a_break.end, "boundary_confidence": None}]
    silences = silence_midpoints(video, a_break.start, a_break.end)
    candidates: list[tuple[float, float, dict[str, float]]] = []
    boundary_threshold = min(float(cfg["ad_boundary_threshold"]), .72) if a_break.end - a_break.start < 30 else float(cfg["ad_boundary_threshold"])
    old = None
    for t, path in frames:
        with Image.open(path) as im:
            gray, hist, black = content_features(im)
        if old:
            old_gray, old_hist, old_black = old
            pixel = float(np.mean(np.abs(gray - old_gray)))
            histogram = float(np.abs(hist-old_hist).sum()) / 2
            silence = any(abs(t-s) < .65 for s in silences)
            strength = clamp(.50 * min(pixel/.26, 1) + .40 * min(histogram/.38, 1) + .10 * (black or old_black) + .12 * silence)
            if strength >= boundary_threshold:
                candidates.append((t, strength, {"pixel_change": round(pixel, 3), "histogram_change": round(histogram, 3), "black_frame": float(black or old_black), "silence": float(silence)}))
        old = (gray, hist, black)
    min_gap = min(float(cfg["minimum_ad_seconds"]), max(3., (a_break.end - a_break.start) / 4))
    boundaries: list[tuple[float, float, dict[str, float]]] = []
    for candidate in sorted(candidates, key=lambda x: x[1], reverse=True):
        t = candidate[0]
        if t-a_break.start >= min_gap and a_break.end-t >= min_gap and all(abs(t-b[0]) >= min_gap for b in boundaries):
            boundaries.append(candidate)
    boundaries.sort()
    points = [(a_break.start, 0., {})] + boundaries + [(a_break.end, 0., {})]
    return [{"start": points[i][0], "end": points[i+1][0],
             "boundary_confidence": round(points[i][1], 3) if i else None,
             "boundary_evidence": points[i][2] if i else {}} for i in range(len(points)-1)]


def render_debug(results: list[FrameResult], destination: Path) -> None:
    cards = []
    for r in results:
        image_tag = f'<img src="{html.escape(r.thumbnail)}" alt="frame at {r.timestamp:.2f}s">' if r.thumbnail else ""
        cards.append(f'''<article class="{'commercial' if r.classification == 'COMMERCIAL' else 'program'}">
          {image_tag}<div class="data"><h3>{r.timestamp:.2f}s · {r.classification} · {r.confidence:.0%}</h3>
          <p><b>Upper left</b><br>Blue break graphic: {r.upper_left_box_score:.2f}<br>
          OCR: <span dir="rtl">{html.escape(r.ocr_text or '—')}</span><br>
          Timer: {html.escape(r.detected_timer or '—')} {'✓ countdown' if r.countdown_consistent else ''}<br>
          הפטריוטים: {r.patriots_text_score:.2f} · מיד נשוב: {r.return_soon_text_score:.2f}</p>
          <p><b>Upper right</b><br>Commercial marker: {r.upper_right_symbol_score:.2f}</p></div></article>''')
    destination.write_text('''<!doctype html><meta charset="utf-8"><title>Patriots detector debug</title>
    <style>body{font:15px system-ui;background:#101827;color:#e9eef7;margin:25px}h1{margin-bottom:25px}
    article{display:flex;gap:20px;background:#1c2940;border-left:5px solid #68788f;margin:12px 0;padding:12px}
    article.commercial{border-color:#4bd5a2}img{width:320px;object-fit:contain}.data{line-height:1.45}
    h3{margin:0}p{margin:8px 0}b{color:#80bfff}</style><h1>Channel 14 / Patriots debug</h1>'''+"\n".join(cards), encoding="utf-8")


def analyse_video(video: str, out: Path, cfg: dict[str, Any], debug: bool, ocr: bool,
                  progress: Callable[[str, float], None] | None = None) -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Video analysis requires ffmpeg on PATH")
    def update(stage: str, fraction: float) -> None:
        if progress:
            progress(stage, clamp(fraction))
    out.mkdir(parents=True, exist_ok=True)
    update("קורא את הסרטון", .03)
    duration = ffprobe_duration(video)
    coarse_cfg = dict(cfg)
    coarse_cfg["scales"] = cfg.get("coarse_scales", cfg["scales"])
    coarse_detector = VisualDetector(coarse_cfg, ocr=False)
    detector = VisualDetector(cfg, ocr=ocr)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        update("דוגם תמונות לסריקה מהירה", .08)
        sampled = extract_frames(video, Path(temp) / "coarse_frames",
                                 1/float(cfg["coarse_interval_seconds"]), coarse=True)
        samples: dict[float, tuple[FrameResult, Path]] = {}
        for index, (t, path) in enumerate(sampled):
            with Image.open(path) as frame:
                # Every fourth frame gets the full position/scale search. The
                # others use calibrated ROIs; temporal grouping bridges gaps.
                result = coarse_detector.score_frame(frame, t, do_ocr=False,
                                                     search=index % 4 == 0)
            samples[round(t, 3)] = result, path
            if index % 10 == 0:
                update("סורק את הסרטון", .14 + .34 * (index + 1) / max(len(sampled), 1))

        candidates = detect_breaks([item[0] for item in samples.values()], duration, cfg)
        for break_index, candidate in enumerate(candidates):
            update("מדייק את גבולות ההפסקות", .49 + .19 * break_index / max(len(candidates), 1))
            padding = float(cfg.get("refine_padding_seconds", 15))
            start = max(0., candidate.start - padding)
            end = min(duration, candidate.end + padding)
            detailed = extract_frames(video, Path(temp) / f"refine_frames_{break_index}",
                                      1/float(cfg["sample_interval_seconds"]), start, end)
            for index, (t, path) in enumerate(detailed):
                with Image.open(path) as frame:
                    result = detector.score_frame(frame, t, do_ocr=False)
                samples[round(t, 3)] = result, path
                if index % 10 == 0:
                    update("מדייק את גבולות ההפסקות", .49 + .19 *
                           (break_index + (index + 1) / max(len(detailed), 1)) / max(len(candidates), 1))

        ordered = [samples[key] for key in sorted(samples)]
        results = [item[0] for item in ordered]
        breaks = detect_breaks(results, duration, cfg)
        update("קורא כותרות וטיימרים", .69)
        for index, (result, path) in enumerate(ordered):
            if ocr and index % 4 == 0 and any(item.start <= result.timestamp <= item.end for item in breaks):
                with Image.open(path) as frame:
                    refreshed = detector.score_frame(frame, result.timestamp, do_ocr=True)
                ordered[index] = refreshed, path
        results = [item[0] for item in ordered]
        tracker = TemporalTracker(cfg)
        for result in results:
            tracker.add(result)
        thumbnails = out / "debug_frames"
        if debug:
            thumbnails.mkdir(exist_ok=True)
            for index, (result, path) in enumerate(ordered):
                with Image.open(path) as frame:
                    frame.thumbnail((640, 360))
                    name = f"{index:06d}.jpg"
                    frame.save(thumbnails / name)
                    result.thumbnail = "debug_frames/" + name
                if index % 25 == 0:
                    update("שומר תמונות לבדיקה", .71 + .05 * (index + 1) / max(len(ordered), 1))
        for i, item in enumerate(breaks):
            update("מפריד בין פרסומות", .77 + .18 * i / max(len(breaks), 1))
            item.ads = split_ads(video, item, Path(temp) / f"ad_frames_{i}", cfg)
    # Backfill provisional start and end samples after confirmation.
    for r in results:
        r.classification = "COMMERCIAL" if any(b.start <= r.timestamp < b.end for b in breaks) else "PROGRAM"
    payload = {"video": video, "duration": duration, "breaks": [b.__dict__ for b in breaks],
               "frames": [r.as_dict() for r in results], "note": "Ad boundaries are visual/audio candidates; validate on real ad samples."}
    (out / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if debug:
        render_debug(results, out / "debug.html")
    update("הניתוח הושלם", 1.0)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Detect Patriots commercial breaks, then segment ads")
    p.add_argument("video", help="Local video file or ffmpeg-readable URL")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/detection")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-ocr", action="store_true")
    args = p.parse_args()
    result = analyse_video(args.video, args.output, load_config(args.config), args.debug, not args.no_ocr)
    print(json.dumps({"breaks": result["breaks"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
