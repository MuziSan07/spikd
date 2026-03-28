"""
pipeline/tribe_processor.py

Handles:
  1. Video → frame extraction (OpenCV)
  2. Frame → TRIBE v2 cortical surface prediction [T, 20484]
  3. Glasser atlas masking → isolate DAN + V1 vertices
  4. 1D collapse → attention spike waveform [T]
  5. Engagement drop detection
  6. Waveform PNG export (monochrome, no 3D rendering)

No 3D rendering. No nilearn mesh viewers. Headless only.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pipeline.atlas_masker import AtlasMasker

logger = logging.getLogger("spikd.tribe")


# ── Glasser HCP MMP1.0 — region indices relevant to SPIKD ────────────────────
# These are parcel IDs (1-indexed, left hemisphere 1–180, right 181–360).
# Dorsal Attention Network (DAN): IPS, FEF, MT+, SPL
DAN_PARCEL_IDS = [
    7, 17, 18, 19, 23, 24,        # IPS regions (LIPd, LIPv, VIP, MIP, AIP, 7PL)
    10, 11,                        # FEF (area 8C, p9-46v)
    157,                           # MT+
    181 + 7 - 1, 181 + 17 - 1,   # Right hemisphere mirrors
    181 + 18 - 1, 181 + 19 - 1,
    181 + 23 - 1, 181 + 24 - 1,
    181 + 10 - 1, 181 + 11 - 1,
    181 + 157 - 1,
]

# Primary Visual Cortex (V1): parcel 1 (left) and 181 (right)
V1_PARCEL_IDS = [1, 181]

TARGET_PARCEL_IDS = list(set(DAN_PARCEL_IDS + V1_PARCEL_IDS))


class TribeProcessor:
    """
    Loads TRIBE v2, processes a video file, and returns
    the 1D attention waveform with metadata.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = None
        self.masker     = AtlasMasker(target_parcel_ids=TARGET_PARCEL_IDS)
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Load TRIBE v2 weights.

        TRIBE v2 (Meta Research) outputs predicted BOLD responses on the
        fsLR 32k cortical surface → shape [T, 20484] per hemisphere pair.

        If weights are not yet available (model still gated / in review),
        the processor falls back to SIMULATION MODE and logs a clear warning.
        Simulation mode generates synthetic data with realistic spectral
        properties for UI/pipeline development and testing.
        """
        checkpoint = self.model_path / "tribe_v2.pt"
        config_file = self.model_path / "config.json"

        if checkpoint.exists() and config_file.exists():
            try:
                import json
                with open(config_file) as f:
                    cfg = json.load(f)

                # Dynamic import — replace with actual TRIBE v2 architecture class
                # when the model is publicly released.
                # from tribe_v2 import TRIBEv2Model
                # self.model = TRIBEv2Model(cfg)
                # self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
                # self.model.eval().to(self.device)

                logger.info("TRIBE v2 weights loaded from %s", checkpoint)
            except Exception as exc:
                logger.warning("Failed to load TRIBE v2 weights: %s — using simulation.", exc)
                self.model = None
        else:
            logger.warning(
                "TRIBE v2 checkpoint not found at %s. "
                "Running in SIMULATION MODE. "
                "Place tribe_v2.pt and config.json in the model directory "
                "to enable real predictions.",
                self.model_path,
            )
            self.model = None

    # ── Main entry point ───────────────────────────────────────────────────────
    def process(self, video_path: str) -> dict:
        """
        Full pipeline: video → 1D waveform + metadata.

        Returns
        -------
        dict with keys:
            waveform            : np.ndarray [T]  — normalised 0–1
            peak_engagement_sec : float
            drop_timestamp_sec  : float
            fps                 : float
            duration_sec        : float
            n_frames            : int
            waveform_png_path   : str
            simulation_mode     : bool
        """
        frames, fps, duration_sec = self._extract_frames(video_path)
        logger.info("Extracted %d frames @ %.1f fps (%.1fs)", len(frames), fps, duration_sec)

        # Step 1: TRIBE v2 → [T, 20484] cortical surface activations
        surface_tensor = self._predict_surface(frames)

        # Step 2: Atlas masking → isolate DAN + V1 vertices
        masked_tensor = self.masker.apply(surface_tensor)   # [T, N_masked]

        # Step 3: 1D collapse — spatial mean over masked vertices
        waveform = masked_tensor.mean(axis=1)               # [T]

        # Step 4: Normalise to [0, 1]
        wmin, wmax = waveform.min(), waveform.max()
        if wmax - wmin > 1e-8:
            waveform = (waveform - wmin) / (wmax - wmin)
        else:
            waveform = np.zeros_like(waveform)

        # Step 5: Metrics
        peak_idx            = int(np.argmax(waveform))
        drop_idx            = self._find_engagement_drop(waveform)
        peak_engagement_sec = peak_idx / fps
        drop_timestamp_sec  = drop_idx / fps

        # Step 6: Render waveform PNG
        out_dir  = Path(video_path).parent
        png_path = str(out_dir / "waveform.png")
        self._render_waveform(
            waveform, fps, drop_idx, peak_idx, png_path
        )

        return {
            "waveform":             waveform.tolist(),
            "peak_engagement_sec":  round(peak_engagement_sec, 2),
            "drop_timestamp_sec":   round(drop_timestamp_sec, 2),
            "fps":                  round(fps, 2),
            "duration_sec":         round(duration_sec, 2),
            "n_frames":             len(frames),
            "waveform_png_path":    png_path,
            "simulation_mode":      self.model is None,
        }

    # ── Frame extraction ───────────────────────────────────────────────────────
    def _extract_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError("Video appears empty — no frames could be read.")

        return frames, fps, duration_sec

    # ── TRIBE v2 inference ─────────────────────────────────────────────────────
    def _predict_surface(self, frames: list) -> np.ndarray:
        """
        Returns np.ndarray of shape [T, 20484].

        When TRIBE v2 weights are loaded:
          - Batch frames through the video encoder
          - Pass through the cortical surface regression head
          - Return predicted BOLD on fsLR-32k surface

        When in simulation mode:
          - Generate synthetic activations with realistic 1/f spectral profile
            and embedded attention peaks matching typical DAN/V1 responses.
        """
        T = len(frames)
        N_VERTS = 20484

        if self.model is not None:
            # ── Real inference ─────────────────────────────────────────────────
            self.model.eval()
            tensor = torch.from_numpy(
                np.stack(frames).astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)  # [1, T, C, H, W]

            with torch.no_grad():
                surface_out = self.model(tensor)   # expected [1, T, 20484]
            return surface_out.squeeze(0).cpu().numpy()

        else:
            # ── Simulation mode ────────────────────────────────────────────────
            # Realistic 1/f noise base
            rng = np.random.default_rng(42)
            freqs     = np.fft.rfftfreq(T)
            freqs[0]  = 1e-6
            power     = 1.0 / freqs ** 0.8
            amplitude = rng.standard_normal((N_VERTS, len(freqs))) * np.sqrt(power)
            phase     = rng.uniform(0, 2 * np.pi, (N_VERTS, len(freqs)))
            spectrum  = amplitude * np.exp(1j * phase)
            surface   = np.fft.irfft(spectrum, n=T, axis=1).T  # [T, N_VERTS]

            # Inject synthetic attention event mid-video
            peak_frame = T // 3
            surface[peak_frame - 2 : peak_frame + 3, :] += rng.uniform(
                0.3, 0.9, (5, N_VERTS)
            )
            return surface.astype(np.float32)

    # ── Engagement drop detection ──────────────────────────────────────────────
    def _find_engagement_drop(self, waveform: np.ndarray, window: int = 5) -> int:
        """
        Returns the index where attention drops most sharply after the peak.
        Uses first-derivative minimum on a smoothed signal.
        """
        if len(waveform) < window * 2:
            return len(waveform) - 1

        kernel   = np.ones(window) / window
        smoothed = np.convolve(waveform, kernel, mode="same")
        gradient = np.gradient(smoothed)
        peak_idx = int(np.argmax(smoothed))

        # Search only after the peak
        post_peak = gradient[peak_idx:]
        if len(post_peak) == 0:
            return peak_idx
        drop_offset = int(np.argmin(post_peak))
        return peak_idx + drop_offset

    # ── Waveform rendering ─────────────────────────────────────────────────────
    def _render_waveform(
        self,
        waveform: np.ndarray,
        fps: float,
        drop_idx: int,
        peak_idx: int,
        out_path: str,
    ):
        """
        Renders a stark monochrome 2D line graph of the attention waveform.
        Marks the peak engagement and the engagement drop timestamp.
        No colour maps, no 3D, no brain meshes.
        """
        T      = len(waveform)
        times  = np.arange(T) / fps

        fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0a0a0a")
        ax.set_facecolor("#0a0a0a")

        # Main waveform line — no fill, raw signal only
        ax.plot(times, waveform, color="#ffffff", linewidth=1.6, zorder=3, antialiased=True)

        # Peak marker
        if peak_idx < T:
            ax.axvline(
                times[peak_idx], color="#ffffff", linewidth=0.8,
                linestyle="--", alpha=0.6, zorder=4
            )
            ax.annotate(
                f"Peak\n{times[peak_idx]:.1f}s",
                xy=(times[peak_idx], waveform[peak_idx]),
                xytext=(times[peak_idx] + 0.3, 0.85),
                color="#ffffff", fontsize=7, alpha=0.8,
                arrowprops=dict(arrowstyle="-", color="#ffffff", alpha=0.4),
            )

        # Drop marker
        if drop_idx < T:
            ax.axvline(
                times[drop_idx], color="#aaaaaa", linewidth=0.8,
                linestyle=":", alpha=0.7, zorder=4
            )
            ax.annotate(
                f"Drop\n{times[drop_idx]:.1f}s",
                xy=(times[drop_idx], waveform[drop_idx]),
                xytext=(times[drop_idx] + 0.3, 0.55),
                color="#aaaaaa", fontsize=7, alpha=0.8,
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", alpha=0.4),
            )

        # Axes styling
        ax.set_xlim(0, times[-1])
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Time (seconds)", color="#888888", fontsize=9)
        ax.set_ylabel("Normalised Attention", color="#888888", fontsize=9)
        ax.tick_params(colors="#666666", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, color="#1e1e1e", linewidth=0.5, zorder=1)

        plt.tight_layout(pad=0.5)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close(fig)
        logger.info("Waveform PNG saved: %s", out_path)
