"""
ERAASR (Estimation and Removal of Array Artifacts via Sequential principal components Regression)
Python implementation adapted from the algorithm described in:

  O'Shea & Shenoy (2018), J. Neural Engineering 15(2):026020
  Open-access manuscript: https://pmc.ncbi.nlm.nih.gov/articles/PMC5833982/

Key ideas (see §3.1–3.5):
- Build a 4D tensor X with shape [C (channels), T (samples per pulse), P (pulses), R (trials)].
- Clean in three sequential stages using PCA regression while *excluding* a small neighborhood around the target item:
  1) Across channels (exclude the channel ±λ_C) [§3.2]
  2) Across pulses (exclude the pulse ±λ_P) [§3.3]
  3) Across trials (exclude the trial ±λ_R), per channel [§3.4]
- Optionally clean post-stimulation transients over trials [§3.5].

This file provides:
- eraasr_tensor_clean(): apply the three-stage ERAASR cleaning on a prepared tensor X
- apply_eraasr_si(): convenience wrapper to integrate with SpikeInterface recordings

Notes & assumptions:
- Recording **must not** saturate during stimulation (additive artifact assumption).
- You must provide stimulation metadata (trial starts, pulses per train, pulse rate).
- For large Neuropixels sessions, this reference implementation loads data into memory for simplicity.
  For production use, consider chunking/writing to Zarr. The math stays the same.

Author: Reza Asri
Cite O'Shea & Shenoy (2018) if you use this.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import spikeinterface as si  # type: ignore
except Exception:  # allow use without SI for tensor-only API
    si = None  # noqa: N816


# -----------------------------
# Small linear algebra helpers
# -----------------------------

def _pca_components(
    M: np.ndarray,
    n_components: int,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA via SVD on matrix M (n_samples x n_features).

    Returns (components, mean, M_centered), where
      - components: (n_features x n_components), column k is w_k
      - mean: (n_features,), column means used for centering
      - M_centered: centered data used for SVD (n_samples x n_features)
    """
    assert M.ndim == 2, "M must be 2D"
    if center:
        mu = M.mean(axis=0, keepdims=True)
        X = M - mu
    else:
        mu = np.zeros((1, M.shape[1]), dtype=M.dtype)
        X = M

    # economical SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    if n_components > V.shape[1]:
        n_components = V.shape[1]
    components = V[:, :n_components]  # (n_features x K)
    return components, mu.ravel(), X


def _regress_and_subtract(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return residual y - A @ beta from least-squares fit.
    y: (N,)
    A: (N, K)
    """
    if A.ndim != 2:
        A = A.reshape(A.shape[0], -1)
    # Handle degenerate K=0 (no PCs)
    if A.size == 0 or A.shape[1] == 0:
        return y.copy()
    # Least squares
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


# -------------------------------------------------
#  Core ERAASR stages on a tensor X: C x T x P x R
# -------------------------------------------------

@dataclass
class ERAASRParams:
    Kc: int = 4          # PCs across channels
    Kp: int = 2          # PCs across pulses
    Kr: int = 4          # PCs across trials
    lambda_c: int = 1    # exclude ±lambda_c adjacent channels
    lambda_p: int = 0    # exclude ±lambda_p adjacent pulses
    lambda_r: int = 0    # exclude ±lambda_r adjacent trials
    center: bool = True  # center columns before PCA


def _clean_over_channels(X: np.ndarray, params: ERAASRParams) -> np.ndarray:
    """Stage 1: shared structure over channels (PCA regression), §3.2.

    X: (C, T, P, R)
    Returns X1 with same shape.
    """
    C, T, P, R = X.shape
    # Reshape to M^c: (R*T*P) x C
    M = X.transpose(1, 2, 3, 0).reshape(T * P * R, C)
    W, mu, Xc = _pca_components(M, params.Kc, center=params.center)
    # Precompute A for each target channel with row-exclusions in W
    M_clean = M.copy()
    for c in range(C):
        W_mod = W.copy()
        # zero out loadings for c ± lambda_c
        lo = max(0, c - params.lambda_c)
        hi = min(C, c + params.lambda_c + 1)
        W_mod[lo:hi, :] = 0.0
        A = Xc @ W_mod  # (RTP x Kc)
        y = Xc[:, c]
        y_clean = _regress_and_subtract(y, A)
        # add mean back (if centered)
        M_clean[:, c] = y_clean + (mu[c] if params.center else 0.0)
    X1 = M_clean.reshape(T, P, R, C).transpose(3, 0, 1, 2)
    return X1


def _clean_over_pulses(X: np.ndarray, params: ERAASRParams) -> np.ndarray:
    """Stage 2: shared structure over pulses, §3.3.

    X: (C, T, P, R)
    Returns X2 with same shape.
    """
    C, T, P, R = X.shape
    # Reshape to M^p: (T*R*C) x P
    M = X.transpose(1, 3, 0, 2).reshape(T * R * C, P)
    W, mu, Xc = _pca_components(M, params.Kp, center=params.center)
    M_clean = M.copy()
    for p in range(P):
        W_mod = W.copy()
        lo = max(0, p - params.lambda_p)
        hi = min(P, p + params.lambda_p + 1)
        W_mod[lo:hi, :] = 0.0
        A = Xc @ W_mod  # (TRC x Kp)
        y = Xc[:, p]
        y_clean = _regress_and_subtract(y, A)
        M_clean[:, p] = y_clean + (mu[p] if params.center else 0.0)
    X2 = M_clean.reshape(T, R, C, P).transpose(2, 0, 3, 1)
    return X2


def _clean_over_trials(X: np.ndarray, params: ERAASRParams) -> np.ndarray:
    """Stage 3: shared structure over trials (per channel), §3.4.

    X: (C, T, P, R)
    Returns X3 with same shape.
    """
    C, T, P, R = X.shape
    X_clean = X.copy()
    TP = T * P
    for c in range(C):
        # M_c: (TP) x R
        M = X_clean[c].reshape(TP, R)
        W, mu, Xc = _pca_components(M, params.Kr, center=params.center)
        M_out = M.copy()
        for r in range(R):
            W_mod = W.copy()
            lo = max(0, r - params.lambda_r)
            hi = min(R, r + params.lambda_r + 1)
            W_mod[lo:hi, :] = 0.0
            A = Xc @ W_mod  # (TP x Kr)
            y = Xc[:, r]
            y_clean = _regress_and_subtract(y, A)
            M_out[:, r] = y_clean + (mu[r] if params.center else 0.0)
        X_clean[c] = M_out.reshape(T, P, R)
    return X_clean


def eraasr_tensor_clean(X: np.ndarray, params: ERAASRParams | None = None) -> np.ndarray:
    """Apply the 3-stage ERAASR cleaning to a prepared tensor X.

    Parameters
    ----------
    X : ndarray
        4D tensor with shape (C, T, P, R).
    params : ERAASRParams
        Algorithm parameters. Defaults match O'Shea & Shenoy (2018) examples.

    Returns
    -------
    X_clean : ndarray
        Cleaned tensor with the same shape as X.
    """
    if params is None:
        params = ERAASRParams()
    assert X.ndim == 4, "X must be C x T x P x R"
    X1 = _clean_over_channels(X, params)
    X2 = _clean_over_pulses(X1, params)
    X3 = _clean_over_trials(X2, params)
    return X3


# ---------------------------------------------------------
#  SpikeInterface integration (simple in-memory reference)
# ---------------------------------------------------------

@dataclass
class StimConfig:
    sampling_frequency: float           # Hz
    stim_starts_samples: Sequence[int]  # len R, onset sample of each train
    n_pulses: int                       # P
    pulse_rate_hz: float                # Hz
    samples_per_pulse: Optional[int] = None  # default: round(fs/fstim)
    channel_ids: Optional[Sequence[int]] = None


def _build_tensor_from_recording(
    recording,
    cfg: StimConfig,
) -> Tuple[np.ndarray, List[int], int, int, int]:
    """Extract C x T x P x R tensor around the stimulation pulses from a RecordingExtractor.

    For simplicity and to avoid overlapping windows, this uses T == period (one period per pulse),
    and tiles windows contiguously across the train.
    """
    fs = cfg.sampling_frequency
    period = int(round(fs / cfg.pulse_rate_hz))
    T = cfg.samples_per_pulse or period
    P = int(cfg.n_pulses)
    starts = np.asarray(cfg.stim_starts_samples, dtype=int)
    R = len(starts)

    if cfg.channel_ids is None:
        try:
            ch_ids = list(recording.channel_ids)
        except Exception:  # SpikeInterface >= 0.103
            ch_ids = list(recording.get_channel_ids())
    else:
        ch_ids = list(cfg.channel_ids)
    C = len(ch_ids)

    X = np.zeros((C, T, P, R), dtype=np.float32)
    for r, t0 in enumerate(starts):
        for p in range(P):
            s0 = t0 + p * period
            s1 = s0 + T
            traces = recording.get_traces(start_frame=s0, end_frame=s1, channel_ids=ch_ids)
            X[:, :, p, r] = traces.T  # traces: (T x C) -> (C x T)
    return X, ch_ids, T, P, R

def _copy_recording_metadata(src, dst):
    try:
        locs = src.get_channel_locations()
        dst.set_channel_locations(locs)
    except Exception:
        pass
    try:
        for prop in src.get_property_keys():          # includes 'inter_sample_shift'
            try:
                dst.set_property(prop, src.get_property(prop))
            except Exception:
                pass
    except Exception:
        pass
    for get_name, set_name in [("get_channel_gains","set_channel_gains"),
                               ("get_channel_offsets","set_channel_offsets")]:
        try:
            vals = getattr(src, get_name)()
            getattr(dst, set_name)(vals)
        except Exception:
            pass
    try:
        if hasattr(src, "get_annotation_keys") and hasattr(dst, "set_annotation"):
            for k in src.get_annotation_keys():
                try:
                    dst.set_annotation(k, src.get_annotation(k))
                except Exception:
                    pass
        elif hasattr(dst, "annotate"):
            # fallback for older SI
            pass
    except Exception:
        pass


def _reinsert_tensor_into_recording(recording, X_clean: np.ndarray, cfg: StimConfig, ch_ids, T: int):
    """Create a new in-memory SpikeInterface recording with cleaned segments spliced back in.

    Fixes 'assignment destination is read-only' by copying the full trace to a writable array.
    Handles minor end-of-file clipping and dtype alignment.
    """
    assert si is not None, "SpikeInterface is required for this function"

    fs = cfg.sampling_frequency
    period = int(round(fs / cfg.pulse_rate_hz))
    P = int(cfg.n_pulses)
    starts = np.asarray(cfg.stim_starts_samples, dtype=int)

    # Get full data as a **writable, contiguous** array (time x channels)
    full = np.ascontiguousarray(np.array(recording.get_traces(), copy=True))
    if full.ndim != 2:
        raise ValueError("recording.get_traces() must return 2D array [time x channels]")
    T_total, C_total = full.shape

    # Map channel order
    try:
        rec_ch_ids = list(recording.get_channel_ids())
    except Exception:
        rec_ch_ids = list(recording.channel_ids)
    idx = np.array([rec_ch_ids.index(cid) for cid in ch_ids], dtype=int)

    # Ensure dtype compatibility
    if X_clean.dtype != full.dtype:
        Xc_dtype = X_clean.astype(full.dtype, copy=False)
    else:
        Xc_dtype = X_clean

    # Splice cleaned data
    for r, t0 in enumerate(starts):
        for p in range(P):
            s0 = t0 + p * period
            s1 = s0 + T
            if s0 >= T_total:
                continue
            s1 = min(s1, T_total)
            seg = Xc_dtype[:, : (s1 - s0), p, r].T  # (window x Csel)
            full[s0:s1, idx] = seg

    # Build a new NumpyRecording (SI >= 0.100 uses list of segments)
    cleaned = si.NumpyRecording([full], sampling_frequency=fs, channel_ids=rec_ch_ids)
    _copy_recording_metadata(recording, cleaned)
    try:
        locs = recording.get_channel_locations()
        cleaned.set_channel_locations(locs)
    except Exception:
        pass
    return cleaned


def apply_eraasr_si(
    recording,
    cfg: StimConfig,
    params: ERAASRParams | None = None,
    return_tensor: bool = False,
):
    """Apply ERAASR to a SpikeInterface RecordingExtractor.

    Parameters
    ----------
    recording : SpikeInterface BaseRecording
        Input recording (broadband; do not high-pass prior to ERAASR).
    cfg : StimConfig
        Stimulation configuration (trial starts, pulses per train, pulse rate, fs).
    params : ERAASRParams, optional
        Algorithm hyperparameters; defaults are sensible and match the paper.
    return_tensor : bool, default False
        If True, also return (X_raw, X_clean) tensors for inspection.

    Returns
    -------
    cleaned_recording : BaseRecording
        New in-memory SpikeInterface recording with cleaned segments spliced in.
    (optional) X_raw, X_clean : np.ndarray
        If return_tensor is True, include the raw and cleaned tensors.
    """
    if params is None:
        params = ERAASRParams()

    X, ch_ids, T, P, R = _build_tensor_from_recording(recording, cfg)
    X_clean = eraasr_tensor_clean(X, params)
    cleaned = _reinsert_tensor_into_recording(recording, X_clean, cfg, ch_ids, T)

    if return_tensor:
        return cleaned, X, X_clean
    return cleaned


# -----------------------------
#  Quick sanity-check utility
# -----------------------------

def check_no_saturation(X: np.ndarray, clip_value: Optional[float] = None) -> bool:
    """Return True if no sample appears clipped to ±clip_value (rough proxy for saturation)."""
    if clip_value is None:
        return True
    return not (np.any(np.isclose(X, clip_value)) or np.any(np.isclose(X, -clip_value)))


# -----------------------------
#  Example usage (SpikeInterface)
# -----------------------------
if __name__ == "__main__":
    # This block is just a commented example; adapt to your paths.
    #
    # import spikeinterface.extractors as se
    # rec = se.BinaryRecordingExtractor(
    #     file_paths=["/path/to/data.bin"],
    #     sampling_frequency=30000,
    #     num_channels=384,
    #     dtype="int16",
    #     time_axis=0,
    # )
    # cfg = StimConfig(
    #     sampling_frequency=30000.0,
    #     stim_starts_samples=[1000000, 2000000, 3000000],
    #     n_pulses=20,
    #     pulse_rate_hz=333.0,
    #     samples_per_pulse=None,  # defaults to round(fs/fstim)
    #     channel_ids=None,
    # )
    # params = ERAASRParams(Kc=4, Kp=2, Kr=4, lambda_c=1, lambda_p=0, lambda_r=0)
    # cleaned, X, Xc = apply_eraasr_si(rec, cfg, params, return_tensor=True)
    # cleaned = si.preprocessing.highpass_filter(cleaned, 250.0)
    # cleaned.save(folder="/path/to/out_cleaned", n_jobs=1, chunk_duration="1s")
    pass
