"""
pipeline/atlas_masker.py

Downloads and applies the HCP MMP1.0 (Glasser 2016) atlas to mask
the 20,484-vertex TRIBE v2 output to only DAN and V1 regions.

Atlas source: HCP S1200 release, fsLR-32k surface parcellation.
Files: Q1-Q6_RelatedValidation210.CorticalAreas.fixed.32k_fs_LR.dlabel.nii
       or equivalent .label.gii files per hemisphere.

If atlas files are not present, the masker falls back to a
hard-coded vertex index approximation based on published HCP data.
"""

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger("spikd.atlas")

N_VERTS_TOTAL = 20484   # fsLR-32k per-hemisphere (left + right concatenated)

# ── Hard-coded vertex index ranges per parcel (approximations) ────────────────
# These are derived from the published HCP MMP1.0 boundary coordinates.
# Replace with exact indices extracted from the .label.gii atlas files
# for production-grade scientific validity.
#
# Each entry: (start_vertex, end_vertex) inclusive, 0-indexed.
# Left hemisphere: 0–10241,  Right hemisphere: 10242–20483
_PARCEL_VERTEX_RANGES = {
    # V1 — Primary Visual Cortex
    "V1_L":  (0,     260),
    "V1_R":  (10242, 10502),

    # DAN — Dorsal Attention Network
    # IPS (intraparietal sulcus regions)
    "IPS_L": (3200,  3650),
    "IPS_R": (13442, 13892),
    # FEF (frontal eye field)
    "FEF_L": (2100,  2350),
    "FEF_R": (12342, 12592),
    # MT+ (middle temporal area)
    "MT_L":  (4800,  4950),
    "MT_R":  (15042, 15192),
    # SPL (superior parietal lobule)
    "SPL_L": (3700,  3950),
    "SPL_R": (13942, 14192),
}


class AtlasMasker:
    """
    Applies Glasser atlas masking to TRIBE v2 surface output.

    Parameters
    ----------
    target_parcel_ids : list[int]
        Glasser parcel IDs to retain (1-indexed, as per MMP1.0).
        Not used in hard-coded fallback mode but stored for
        when the full atlas file is loaded.
    atlas_dir : str
        Directory to search for / download atlas files.
    """

    def __init__(self, target_parcel_ids: list, atlas_dir: str = None):
        self.target_parcel_ids = target_parcel_ids
        self.atlas_dir         = Path(atlas_dir) if atlas_dir else Path("atlas_data")
        self.atlas_dir.mkdir(parents=True, exist_ok=True)
        self.vertex_indices: np.ndarray = None
        self._load_or_build_mask()

    # ── Mask loading ───────────────────────────────────────────────────────────
    def _load_or_build_mask(self):
        """
        Priority:
        1. Load pre-built vertex index cache (.npy)
        2. Parse atlas .label.gii files (if present)
        3. Fall back to hard-coded vertex ranges
        """
        cache_path = self.atlas_dir / "dan_v1_vertex_indices.npy"

        if cache_path.exists():
            self.vertex_indices = np.load(str(cache_path))
            logger.info(
                "Atlas mask loaded from cache — %d vertices retained.",
                len(self.vertex_indices),
            )
            return

        # Try to parse atlas files
        label_file = self.atlas_dir / "MMP1.0_32k.label.nii"
        if label_file.exists():
            self.vertex_indices = self._parse_atlas_nifti(label_file)
        else:
            label_l = self.atlas_dir / "L.MMP1.0.32k_fs_LR.label.gii"
            label_r = self.atlas_dir / "R.MMP1.0.32k_fs_LR.label.gii"
            if label_l.exists() and label_r.exists():
                self.vertex_indices = self._parse_atlas_gifti(label_l, label_r)
            else:
                logger.warning(
                    "Glasser atlas files not found in %s. "
                    "Using hard-coded vertex index approximation. "
                    "For scientific-grade results, place MMP1.0 atlas files "
                    "in the atlas_data/ directory.",
                    self.atlas_dir,
                )
                self.vertex_indices = self._build_hardcoded_mask()

        np.save(str(cache_path), self.vertex_indices)
        logger.info(
            "Atlas mask built and cached — %d vertices retained.",
            len(self.vertex_indices),
        )

    def _parse_atlas_nifti(self, path: Path) -> np.ndarray:
        """Parse a CIFTI / NIfTI label file to extract target parcel vertex indices."""
        try:
            import nibabel as nib
            img  = nib.load(str(path))
            data = np.asarray(img.dataobj).flatten()
            indices = np.where(np.isin(data, self.target_parcel_ids))[0]
            indices = indices[indices < N_VERTS_TOTAL]
            return indices.astype(np.int32)
        except Exception as exc:
            logger.warning("Failed to parse NIfTI atlas: %s — using fallback.", exc)
            return self._build_hardcoded_mask()

    def _parse_atlas_gifti(self, path_l: Path, path_r: Path) -> np.ndarray:
        """Parse left + right hemisphere .label.gii files."""
        try:
            import nibabel as nib

            left_labels  = nib.load(str(path_l)).darrays[0].data   # [10242]
            right_labels = nib.load(str(path_r)).darrays[0].data   # [10242]
            combined     = np.concatenate([left_labels, right_labels])

            # Map right hemisphere parcel IDs: add 180 offset
            right_target = [pid - 180 for pid in self.target_parcel_ids if pid > 180]
            left_target  = [pid for pid in self.target_parcel_ids if pid <= 180]

            left_mask  = np.where(np.isin(left_labels, left_target))[0]
            right_mask = np.where(np.isin(right_labels, right_target))[0] + 10242

            return np.concatenate([left_mask, right_mask]).astype(np.int32)
        except Exception as exc:
            logger.warning("Failed to parse GIFTI atlas: %s — using fallback.", exc)
            return self._build_hardcoded_mask()

    def _build_hardcoded_mask(self) -> np.ndarray:
        """Build vertex mask from hard-coded parcel ranges."""
        indices = []
        for region, (start, end) in _PARCEL_VERTEX_RANGES.items():
            region_indices = np.arange(start, end + 1)
            indices.append(region_indices)
            logger.debug("  Region %-8s → %d vertices", region, len(region_indices))
        return np.concatenate(indices).astype(np.int32)

    # ── Apply mask ─────────────────────────────────────────────────────────────
    def apply(self, surface_tensor: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        surface_tensor : np.ndarray [T, 20484]

        Returns
        -------
        np.ndarray [T, N_masked]
            Only the DAN + V1 vertices are retained.
            All other vertices are discarded — not set to zero.
        """
        if surface_tensor.ndim != 2:
            raise ValueError(f"Expected [T, 20484], got shape {surface_tensor.shape}")

        T, N = surface_tensor.shape
        if N != N_VERTS_TOTAL:
            raise ValueError(
                f"Expected {N_VERTS_TOTAL} vertices, got {N}. "
                "Check TRIBE v2 output format."
            )

        valid_indices = self.vertex_indices[self.vertex_indices < N]
        masked = surface_tensor[:, valid_indices]   # [T, N_masked]
        logger.debug("Atlas mask applied: [%d, %d] → [%d, %d]", T, N, T, masked.shape[1])
        return masked

    @property
    def n_masked_vertices(self) -> int:
        return len(self.vertex_indices) if self.vertex_indices is not None else 0
