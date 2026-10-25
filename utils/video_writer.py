"""
Small streaming video writer with high-quality ffmpeg encoding and cv2 fallback.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class StreamingVideoWriter:
    """
    Write RGB uint8 frames to MP4 without keeping them in memory.
    """

    def __init__(
        self,
        video_path: str,
        fps: int,
        width: int,
        height: int,
        crf: int = 16,
        preset: str = "slow",
        pix_fmt: str = "yuv420p",
    ) -> None:
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.cv2_writer: cv2.VideoWriter | None = None
        self.process: subprocess.Popen[bytes] | None = None
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            self.cv2_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (self.width, self.height),
            )
            if not self.cv2_writer.isOpened():
                raise RuntimeError("Failed to open video writer for {}.".format(video_path))
            return

        command = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "{}x{}".format(self.width, self.height),
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            str(pix_fmt),
            "-movflags",
            "+faststart",
            video_path,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                "Frame size {}x{} does not match writer size {}x{}.".format(
                    frame_rgb.shape[1],
                    frame_rgb.shape[0],
                    self.width,
                    self.height,
                )
            )
        frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        if self.process is not None:
            if self.process.stdin is None:
                raise RuntimeError("ffmpeg stdin is closed for {}.".format(self.video_path))
            self.process.stdin.write(frame_rgb.tobytes())
            return
        if self.cv2_writer is None:
            raise RuntimeError("Video writer is not initialized for {}.".format(self.video_path))
        self.cv2_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            return_code = self.process.wait()
            self.process = None
            if return_code != 0:
                raise RuntimeError("ffmpeg failed with code {} for {}.".format(return_code, self.video_path))
        if self.cv2_writer is not None:
            self.cv2_writer.release()
            self.cv2_writer = None
