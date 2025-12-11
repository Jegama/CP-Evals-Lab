"""Audio file handling utilities for sermon evaluation.

Provides:
- Duration extraction (ffprobe + mutagen fallback)
- Gemini Files API upload with caching
- UX helpers (file size formatting, upload progress indicators)
"""

from __future__ import annotations

import json
import os
import threading
import time
import itertools
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class AudioFileManager:
    """Manages audio file operations: duration extraction, upload caching."""

    def __init__(self, cache_dir: Path = Path(".cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.audio_cache_path = self.cache_dir / "gemini_files_cache.json"
        if not self.audio_cache_path.exists():
            self.audio_cache_path.write_text("{}", encoding="utf-8")

    @staticmethod
    def get_audio_duration(file_path: str) -> Optional[float]:
        """Extract audio duration in seconds using ffprobe (preferred) or mutagen.

        Returns None if file cannot be parsed.
        """
        # 1. Try ffprobe first (most accurate for VBR/containers)
        try:
            import subprocess

            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                val = result.stdout.strip()
                if val:
                    return float(val)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
            # ffprobe not installed, timed out, invalid output, or file access error
            pass

        # 2. Fallback to mutagen
        try:
            from mutagen._file import File
            from mutagen.mp3 import MP3
            from mutagen.mp4 import MP4
            from mutagen.wave import WAVE
            from mutagen.flac import FLAC
            from mutagen.oggvorbis import OggVorbis

            audio = File(file_path)
            if audio is not None and audio.info:
                return float(audio.info.length)

            ext = Path(file_path).suffix.lower()
            if ext == ".mp3":
                audio = MP3(file_path)
            elif ext in [".m4a", ".mp4"]:
                audio = MP4(file_path)
            elif ext == ".wav":
                audio = WAVE(file_path)
            elif ext == ".flac":
                audio = FLAC(file_path)
            elif ext == ".ogg":
                audio = OggVorbis(file_path)

            if audio is not None and audio.info:
                return float(audio.info.length)

        except (ImportError, OSError, AttributeError, ValueError) as e:
            # mutagen not installed, file access error, missing attributes, or invalid data
            print(
                f"[sermons] Warning: could not extract audio duration from {file_path}: {e}"
            )
        return None

    @staticmethod
    def format_file_size(bytes_count: int) -> str:
        """Format byte count as human-readable size."""
        try:
            mb = bytes_count / (1024 * 1024)
            if mb >= 1024:
                gb = mb / 1024
                return f"{gb:.2f} GB"
            return f"{mb:.2f} MB"
        except (TypeError, ZeroDivisionError):
            # Invalid input type or division error
            return f"{bytes_count} B"

    @staticmethod
    @contextmanager
    def upload_indicator(message: str = "Working"):
        """ASCII spinner shown while a blocking operation runs."""
        stop = threading.Event()

        def spin():
            for ch in itertools.cycle("|/-\\"):
                if stop.is_set():
                    break
                print(f"\r{message} {ch}", end="", flush=True)
                time.sleep(0.1)
            print("\r", end="")

        t = threading.Thread(target=spin, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            # Increased timeout and added cleanup check
            if not t.join(timeout=1.0):
                # Thread didn't terminate cleanly, but it's a daemon so it will be killed
                print(f"\r{message} done (cleanup delayed).")
            else:
                print(f"{message} done.")

    def load_cache(self) -> Dict[str, Any]:
        """Load audio file upload cache."""
        try:
            return json.loads(self.audio_cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # Cache file missing, corrupted, or unreadable
            return {}

    def save_cache(self, cache: Dict[str, Any]) -> None:
        """Save audio file upload cache."""
        self.audio_cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def upload_or_get_gemini_file(
        self, local_path: str, provider: Any
    ) -> Tuple[str, Any]:
        """Upload audio file to Gemini Files API or retrieve from cache.

        Args:
            local_path: Path to local audio file
            provider: ParrotAIGemini instance with upload_file/get_file methods

        Returns:
            Tuple of (remote_id, file_obj)
        """
        abs_path = str(Path(local_path).expanduser().resolve())
        cache = self.load_cache()

        if abs_path in cache:
            remote_id = cache[abs_path]
            print(
                f"[sermons] [cache] Using previously uploaded file:\n  {abs_path} -> {remote_id}"
            )
            file_obj = None
            try:
                if hasattr(provider, "get_file"):
                    file_obj = provider.get_file(remote_id)
            except Exception as e:
                print(
                    f"[sermons] [cache] Could not fetch file object for {remote_id}: {e}"
                )
                try:
                    with self.upload_indicator(message="Re-uploading to Gemini"):
                        file_obj = provider.upload_file(abs_path)
                    cache[abs_path] = (
                        getattr(file_obj, "name", None)
                        or getattr(file_obj, "uri", None)
                        or getattr(file_obj, "id", None)
                    )
                    remote_id = cache[abs_path]
                    self.save_cache(cache)
                    print(f"[sermons] [upload] Re-uploaded successfully -> {remote_id}")
                except Exception as ee:
                    print(f"[sermons] [upload] Re-upload failed: {ee}")
            return remote_id, file_obj

        # Upload new file
        try:
            size_bytes = os.path.getsize(abs_path)
        except (FileNotFoundError, OSError):
            # File missing or inaccessible
            size_bytes = -1
        size_str = (
            self.format_file_size(size_bytes) if size_bytes >= 0 else "unknown size"
        )
        print(
            f"[sermons] [upload] Uploading to Gemini Files API:\n  {abs_path} ({size_str})"
        )

        with self.upload_indicator(message="Uploading to Gemini"):
            file_obj = provider.upload_file(abs_path)

        cache[abs_path] = (
            getattr(file_obj, "name", None)
            or getattr(file_obj, "uri", None)
            or getattr(file_obj, "id", None)
        )
        self.save_cache(cache)
        print(f"[sermons] [upload] Uploaded successfully -> {cache[abs_path]}")
        return cache[abs_path], file_obj


__all__ = ["AudioFileManager"]
