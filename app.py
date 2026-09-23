"""Browser UI for the Patriots commercial detector.

Run with: python3 app.py. By default the server is local-only; a deployment may
set PATRIOTS_HOST to 0.0.0.0 and choose its port with PATRIOTS_PORT.
"""
from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import threading
import traceback
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image

from patriots_detector import REFERENCE, VisualDetector, analyse_video, ffprobe_duration, load_config

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
WORK = ROOT / "work"
UPLOADS = WORK / "uploads"
RUNS = WORK / "runs"
ARCHIVE_CACHE = WORK / "archive_cache"
RESULT_CACHE = WORK / "result_cache"
ANALYSIS_VERSION = 2
DEMO = ROOT / "assets/demo/patriots_demo.mp4"
PORT = int(os.environ.get("PATRIOTS_PORT", "8765"))
HOST = os.environ.get("PATRIOTS_HOST", "127.0.0.1")
MAX_BYTES = 3 * 1024 * 1024 * 1024
MAX_LINK_SECONDS = 2 * 60 * 60
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
MIMES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".png": "image/png", ".json": "application/json; charset=utf-8"}


def set_job(job_id: str, **values) -> None:
    with LOCK:
        JOBS[job_id].update(values)
        save_job(job_id)


def save_job(job_id: str) -> None:
    path = RUNS / job_id / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(JOBS[job_id], ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def register_job(job_id: str, filename: str, stage: str) -> None:
    with LOCK:
        JOBS[job_id] = {"id": job_id, "filename": filename,
                        "status": "queued", "stage": stage, "progress": 0.0}
        save_job(job_id)


def get_job(job_id: str) -> dict | None:
    with LOCK:
        if job_id in JOBS:
            return JOBS[job_id].copy()
    path = RUNS / job_id / "job.json"
    if not path.is_file():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("status") in {"queued", "processing"}:
        job.update(status="error", stage="הניתוח הופסק", error="השרת הופעל מחדש לפני סיום הניתוח")
    return job


def archive_cache_path(url: str) -> Path:
    return ARCHIVE_CACHE / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".mkv")


def result_cache_path(url: str, mode: str) -> Path:
    signature = json.dumps({"url": url, "mode": mode, "config": load_config(),
                            "version": ANALYSIS_VERSION}, ensure_ascii=False, sort_keys=True)
    return RESULT_CACHE / (hashlib.sha256(signature.encode("utf-8")).hexdigest() + ".json")


def cached_job_id(url: str, mode: str) -> str | None:
    path = result_cache_path(url, mode)
    if not path.is_file():
        return None
    try:
        job_id = json.loads(path.read_text(encoding="utf-8"))["job_id"]
        job = get_job(job_id)
        if job and job.get("status") == "done":
            return job_id
    except (OSError, ValueError, KeyError):
        pass
    return None


def remember_result(url: str, mode: str, job_id: str) -> None:
    path = result_cache_path(url, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
    os.replace(temporary, path)


def render_clip(source: Path, start: float, end: float, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, end - start)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", str(max(0, start)), "-i", str(source), "-t", str(duration),
               "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "ultrafast",
               "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", str(destination)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode or not destination.is_file() or destination.stat().st_size < 1024:
        destination.unlink(missing_ok=True)
        raise RuntimeError("ייצוא הווידאו נכשל: " + proc.stderr[-400:])


def export_clips(job_id: str, source: Path, breaks: list[dict], base: float) -> str | None:
    output = RUNS / job_id / "clips"
    count = sum(1 + len(item.get("ads", [])) for item in breaks)
    done = 0
    for break_index, item in enumerate(breaks, 1):
        entries = [(f"break_{break_index:03d}.mp4", item)]
        entries += [(f"ad_{break_index:03d}_{ad_index:03d}.mp4", ad)
                    for ad_index, ad in enumerate(item.get("ads", []), 1)]
        for filename, segment in entries:
            set_job(job_id, stage=f"מייצא קטעי וידאו · {done + 1} מתוך {count}",
                    progress=round(base + (1-base) * done / max(count, 1), 3))
            render_clip(source, float(segment["start"]), float(segment["end"]), output / filename)
            segment["clip_url"] = f"/api/jobs/{job_id}/clips/{filename}"
            done += 1
    ads = sorted(output.glob("ad_*.mp4"))
    if not ads:
        return None
    with zipfile.ZipFile(output / "all_ads.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for clip in ads:
            archive.write(clip, clip.name)
    return f"/api/jobs/{job_id}/all_ads.zip"


def run_job(job_id: str, video: Path) -> None:
    try:
        set_job(job_id, status="processing", stage="מכין את הסרטון", progress=.01)
        output = RUNS / job_id
        if video.parent.resolve() == UPLOADS.resolve():
            saved = output / ("source" + video.suffix.lower())
            saved.parent.mkdir(parents=True, exist_ok=True)
            os.replace(video, saved)
            video = saved
        def progress(stage: str, fraction: float) -> None:
            set_job(job_id, stage=stage, progress=round(.02 + .78 * fraction, 3))
        result = analyse_video(str(video), output, load_config(), True, True, progress)
        all_ads_url = export_clips(job_id, video, result["breaks"], .80)
        set_job(job_id, status="done", stage="הניתוח הושלם", progress=1.0,
                result={"duration": result["duration"], "breaks": result["breaks"],
                        "sample_count": len(result["frames"]), "all_ads_url": all_ads_url})
    except Exception as exc:
        traceback.print_exc()
        set_job(job_id, status="error", stage="הניתוח נכשל", error=str(exc))



def create_job(video: Path, filename: str) -> str:
    job_id = uuid.uuid4().hex
    register_job(job_id, filename, "הקובץ מוכן")
    threading.Thread(target=run_job, args=(job_id, video), daemon=True).start()
    return job_id


def validate_video_url(value: str) -> str:
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("יש להדביק קישור וידאו תקין")
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("נדרש קישור HTTP או HTTPS ישיר לווידאו או ל־HLS")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".internal")):
        raise ValueError("קישורים לכתובות מקומיות אינם נתמכים")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("לא ניתן למצוא את שרת הווידאו") from exc
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("קישור הווידאו חייב להפנות לשרת ציבורי")
    return value


def capture_url(job_id: str, url: str, mode: str, destination: Path) -> tuple[float, float | None]:
    try:
        source_duration = ffprobe_duration(url)
    except (RuntimeError, ValueError, subprocess.SubprocessError):
        source_duration = None
    if mode == "full":
        if source_duration is None:
            raise RuntimeError("זהו כנראה שידור חי; בחר משך הקלטה של 3 או 15 דקות")
        if source_duration > MAX_LINK_SECONDS:
            raise RuntimeError("הקישור ארוך משעתיים. בחר קטע קצר יותר")
        seconds = source_duration
    else:
        seconds = min(float(mode), source_duration) if source_duration else float(mode)
    set_job(job_id, stage=f"קולט וידאו מהקישור · עד {round(seconds/60)} דקות", progress=.03)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = RUNS / job_id / "capture.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-progress", "pipe:1",
               "-rw_timeout", "15000000", "-i", url, "-t", str(seconds),
               "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-f", "matroska", "-y", str(destination)]
    with log_path.open("w", encoding="utf-8") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith("out_time="):
                match = re.match(r"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)", line)
                if match:
                    position = int(match[1])*3600 + int(match[2])*60 + float(match[3])
                    set_job(job_id, progress=round(min(.27, .03 + .24 * position / max(seconds, 1)), 3),
                            stage=f"קולט וידאו מהקישור · {int(position)} מתוך {int(seconds)} שניות")
        code = process.wait()
    if code or not destination.exists() or destination.stat().st_size < 1000:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-500:].replace(url, "[הקישור]")
        raise RuntimeError("לא ניתן לקלוט את הווידאו מהקישור. " + detail)
    return seconds, source_duration


def run_url_job(job_id: str, url: str, mode: str) -> None:
    video = RUNS / job_id / "source.mkv"
    try:
        set_job(job_id, status="processing", stage="בודק את הקישור", progress=.01)
        cache = archive_cache_path(url)
        if mode == "full" and cache.is_file():
            video = cache
            source_duration = ffprobe_duration(str(video))
            requested = source_duration
            set_job(job_id, stage="ההקלטה המלאה כבר נקלטה", progress=.27)
        else:
            requested, source_duration = capture_url(job_id, url, mode, video)
            if mode == "full":
                cache.parent.mkdir(parents=True, exist_ok=True)
                os.replace(video, cache)
                video = cache
        output = RUNS / job_id
        def progress(stage: str, fraction: float) -> None:
            set_job(job_id, stage=stage, progress=round(.28 + .52*fraction, 3))
        result = analyse_video(str(video), output, load_config(), True, True, progress)
        all_ads_url = export_clips(job_id, video, result["breaks"], .80)
        set_job(job_id, status="done", stage="הניתוח הושלם", progress=1.0,
                result={"duration": result["duration"], "breaks": result["breaks"],
                        "sample_count": len(result["frames"]), "source_duration": source_duration,
                        "mode": mode, "requested_seconds": requested,
                        "all_ads_url": all_ads_url})
        remember_result(url, mode, job_id)
    except Exception as exc:
        traceback.print_exc()
        set_job(job_id, status="error", stage="הניתוח נכשל", error=str(exc))


def create_url_job(url: str, mode: str, force: bool = False) -> tuple[str, bool]:
    if not force:
        previous = cached_job_id(url, mode)
        if previous:
            return previous, True
    job_id = uuid.uuid4().hex
    register_job(job_id, urlsplit(url).hostname or "קישור וידאו", "הקישור התקבל")
    threading.Thread(target=run_url_job, args=(job_id, url, mode), daemon=True).start()
    return job_id, False


class Handler(BaseHTTPRequestHandler):
    server_version = "PatriotsDetector/1.0"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, status: int = 200) -> None:
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def send_file(self, path: Path, root: Path) -> None:
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            self.send_error(404)
            return
        self.send_bytes(path.read_bytes(), MIMES.get(path.suffix.lower(), "application/octet-stream"))

    def send_video_file(self, path: Path, download: bool = False,
                        content_type: str = "video/mp4") -> None:
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        requested_range = self.headers.get("Range", "")
        if requested_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested_range.strip())
            if not match or (not match[1] and not match[2]):
                self.send_error(416)
                return
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else size - 1
            else:
                start = max(0, size - int(match[2]))
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if requested_range else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "private, no-store")
        if requested_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self.send_json({"ok": True, "ffmpeg": bool(shutil.which("ffmpeg")),
                            "ocr": bool(shutil.which("tesseract"))})
            return
        if path == "/api/reference-test":
            with Image.open(REFERENCE) as image:
                result = VisualDetector(load_config(), ocr=True).score_frame(image)
            self.send_json(result.as_dict())
            return
        if path == "/reference.png":
            self.send_file(REFERENCE, ROOT)
            return
        job_match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path)
        if job_match:
            job = get_job(job_match[1])
            if not job:
                self.send_error(404)
            else:
                self.send_json(job)
            return
        clip_match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/clips/((?:break_\d{3}|ad_\d{3}_\d{3})\.mp4)", path)
        if clip_match:
            self.send_video_file(RUNS / clip_match[1] / "clips" / clip_match[2],
                                 download="download=1" in urlsplit(self.path).query)
            return
        archive_match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/all_ads\.zip", path)
        if archive_match:
            self.send_video_file(RUNS / archive_match[1] / "clips" / "all_ads.zip",
                                 download=True, content_type="application/zip")
            return
        file_match = re.fullmatch(r"/jobs/([0-9a-f]{32})/(.+)", path)
        if file_match:
            self.send_file(RUNS / file_match[1] / unquote(file_match[2]), RUNS / file_match[1])
            return
        requested = "index.html" if path == "/" else unquote(path.lstrip("/"))
        self.send_file(WEB / requested, WEB)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/url-jobs":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 10000:
                    raise ValueError("בקשת הקישור ארוכה מדי")
                data = json.loads(self.rfile.read(size))
                url = validate_video_url(data.get("url", ""))
                mode = str(data.get("mode", "180"))
                if mode not in {"180", "900", "full"}:
                    raise ValueError("יש לבחור 3 דקות, 15 דקות או את כל הקישור")
                force = data.get("force", False)
                if not isinstance(force, bool):
                    raise ValueError("ערך סריקה מחדש אינו תקין")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            job_id, reused = create_url_job(url, mode, force)
            self.send_json({"id": job_id, "reused": reused}, 200 if reused else 202)
            return
        if path == "/api/demo":
            if not DEMO.is_file():
                self.send_json({"error": "סרטון ההדגמה חסר"}, 500)
                return
            self.send_json({"id": create_job(DEMO, "סרטון הדגמה סינתטי")}, 202)
            return
        if path != "/api/jobs":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= MAX_BYTES:
            self.send_json({"error": "יש להעלות קובץ וידאו בגודל עד 3GB"}, 413)
            return
        filename = unquote(self.headers.get("X-Filename", "video.mp4"))
        ext = Path(filename).suffix.lower()
        if ext not in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".ts"}:
            self.send_json({"error": "סוג הקובץ אינו נתמך. יש לבחור MP4, MOV, MKV, WEBM או TS."}, 415)
            return
        job_id = uuid.uuid4().hex
        UPLOADS.mkdir(parents=True, exist_ok=True)
        RUNS.mkdir(parents=True, exist_ok=True)
        target = UPLOADS / (job_id + ext)
        remaining = size
        with target.open("wb") as stream:
            while remaining:
                part = self.rfile.read(min(1024 * 1024, remaining))
                if not part:
                    break
                stream.write(part)
                remaining -= len(part)
        if remaining:
            target.unlink(missing_ok=True)
            self.send_json({"error": "ההעלאה לא הושלמה"}, 400)
            return
        register_job(job_id, Path(filename).name, "הקובץ הועלה")
        threading.Thread(target=run_job, args=(job_id, target), daemon=True).start()
        self.send_json({"id": job_id}, 202)


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    local_bin = WORK / "bin"
    os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
    print(f"Patriots detector: http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
