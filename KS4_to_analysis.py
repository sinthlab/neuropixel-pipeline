"""
KS4_to_analysis.py


Load KiloSort4 output, pack a single analysis file (NPZ) with:
- spike_times_s: 1D float array of spike times (seconds)
- spike_unit_ids: 1D int array (same length) giving the unit id for each spike
- unit_ids: 1D int array listing all unit IDs (order used elsewhere)
- icms_sec: 1D float array of trial start times (seconds)
- fs: float sampling rate (Hz) from the sorter
- meta: JSON string with creation info and params (includes absolute ks_folder path)


Also includes helper functions to:
- compute icms_sec from ttl_ap_frames (AP sample indices) and fs
- plot a PSTH with flexible selection:
• single unit (pass an int unit_id)
• grand‑average over good units (pass path to good_units.csv)
• grand‑average over all units (pass None)
• you can also pass a list/array of unit IDs to average
- overlay **per‑trial** low‑D neural trajectories (PCA) in a shared basis across trials
- plot **trial‑averaged** low‑D neural trajectories (PCA)
- optional filtering by **good units** for the trajectory plots via:
• good_units=True → auto‑load "{ks_folder}/good_units.csv" from NPZ meta
• good_units="/path/to/good_units.csv" → explicit CSV path
• good_units=[...] → list/array of unit IDs


CSV format for good units (robust to header/no‑header):
unit_id
92
93
...


Usage examples (Python):


>>> from KS4_to_analysis import build_npz, make_icms_sec, plot_psth, plot_trial_trajectories, plot_avg_trajectory
>>> icms_sec = make_icms_sec(ttl_ap_frames, fs) # if you have AP‑frame onsets
>>> build_npz(sorter_output='path/to/sorter_output', icms_sec=icms_sec, outfile='session.npz')


# PSTH — single unit
>>> plot_psth('session.npz', unit_id=10, t_before=0.2, t_after=0.6, bin_size=0.01)
# PSTH — grand‑average of good units from CSV
>>> plot_psth('session.npz', unit_id='path/to/output_folder/good_units.csv')
# PSTH — grand‑average of all units
>>> plot_psth('session.npz', unit_id=None)


# Overlay per‑trial trajectories (shared PCA space), filter to good units found next to KS output
>>> plot_trial_trajectories('session.npz', good_units=True, t_before=0.2, t_after=0.6, bin_size=0.02, n_components=3)


# Trial‑averaged trajectory with an explicit CSV of good units
>>> plot_avg_trajectory('session.npz', good_units='path/to/output_folder/good_units.csv', n_components=3)


CLI (optional):
$ python KS4_to_analysis.py --ks path/to/sorter_output --icms path/to/icms.npy --out session.npz


Notes:
- This is read‑only; it won’t modify sorter outputs.
- Robust to SpikeInterface versions by trying multiple reader entry points.
- Ensure your icms_sec and sorting share the same time reference.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# ---- SpikeInterface compatibility helpers ----

def _read_kilosort(folder: str | Path):
    """Return a SpikeInterface Sorting object from a KiloSort folder.
    Tries multiple import paths to be robust across SI versions.
    """
    folder = str(folder)
    # New API (>=0.98):
    try:
        import spikeinterface as si  # noqa: F401
        from spikeinterface.extractors import read_kilosort  # type: ignore
        return read_kilosort(folder)
    except Exception:
        pass
    # Older API:
    try:
        import spikeinterface.extractors as se  # type: ignore
        return se.read_kilosort(folder)  # type: ignore
    except Exception as exc:
        raise ImportError("Could not import SpikeInterface KiloSort reader. Please install spikeinterface.") from exc


# ---- Core packaging ----

@dataclass
class PackedSession:
    spike_times_s: np.ndarray   # shape (N_spikes,)
    spike_unit_ids: np.ndarray  # shape (N_spikes,)
    unit_ids: np.ndarray        # shape (N_units,)
    icms_sec: np.ndarray        # shape (N_trials,)
    fs: float
    meta: dict

    def save_npz(self, path: str | Path):
        path = str(path)
        np.savez_compressed(
            path,
            spike_times_s=self.spike_times_s.astype(np.float64),
            spike_unit_ids=self.spike_unit_ids.astype(np.int64),
            unit_ids=self.unit_ids.astype(np.int64),
            icms_sec=self.icms_sec.astype(np.float64),
            fs=np.float64(self.fs),
            meta=json.dumps(self.meta),
        )


def build_npz(
    sorter_output: str | Path,
    icms_sec: Sequence[float] | np.ndarray,
    outfile: str | Path = "session.npz",
    include_units: Optional[Iterable[int]] = None,
) -> str:
    """Create a single NPZ file containing spikes, unit ids and trial onsets.

    Parameters
    ----------
    sorter_output : path to KiloSort4 results folder (e.g., "sorter_output")
    icms_sec      : 1D array-like of trial start times (seconds)
    outfile       : output NPZ path
    include_units : optional subset of units to include; default keeps all

    Returns
    -------
    Path to the saved NPZ file (string).
    """
    sorting = _read_kilosort(sorter_output)
    fs = float(sorting.get_sampling_frequency())
    unit_ids = np.asarray(sorting.get_unit_ids(), dtype=int)

    if include_units is not None:
        include_units = set(int(u) for u in include_units)
        unit_ids = np.array([u for u in unit_ids if u in include_units], dtype=int)

    # Concatenate all spikes across units into parallel arrays
    all_spike_times = []  # seconds
    all_spike_units = []  # unit ids matching times

    for uid in unit_ids:
        # SpikeInterface returns frames; convert to seconds
        st_frames = sorting.get_unit_spike_train(unit_id=int(uid))
        st_s = st_frames.astype(np.float64) / fs
        if st_s.size:
            all_spike_times.append(st_s)
            all_spike_units.append(np.full(st_s.size, uid, dtype=np.int64))

    if all_spike_times:
        spike_times_s = np.concatenate(all_spike_times)
        spike_unit_ids = np.concatenate(all_spike_units)
        # Sort by time for convenience
        order = np.argsort(spike_times_s, kind="mergesort")
        spike_times_s = spike_times_s[order]
        spike_unit_ids = spike_unit_ids[order]
    else:
        spike_times_s = np.zeros((0,), dtype=np.float64)
        spike_unit_ids = np.zeros((0,), dtype=np.int64)

    icms_sec = np.asarray(icms_sec, dtype=np.float64)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ks_folder": str(Path(sorter_output).resolve()),
        "n_units": int(unit_ids.size),
        "n_spikes": int(spike_times_s.size),
        "n_trials": int(icms_sec.size),
        "fs": fs,
        "notes": "Built for PSTH and trial-averaged trajectories.",
    }

    packed = PackedSession(
        spike_times_s=spike_times_s,
        spike_unit_ids=spike_unit_ids,
        unit_ids=unit_ids,
        icms_sec=icms_sec,
        fs=fs,
        meta=meta,
    )
    packed.save_npz(outfile)
    return str(Path(outfile).resolve())


# ---- Utilities ----

def make_icms_sec(ttl_ap_frames: dict, fs: float) -> np.ndarray:
    """Pick the stimulation onsets key that exists (icms/nerve/opto), return seconds.
    Expects AP-sample indices as values in ttl_ap_frames[key].
    """
    for k in ("icms", "nerve", "opto"):
        if k in ttl_ap_frames:
            arr = np.asarray(ttl_ap_frames[k], dtype=np.float64)
            return arr / float(fs)
    raise KeyError("No 'icms', 'nerve', or 'opto' key found in ttl_ap_frames.")


# ---- Good-units utilities ----

def _load_good_units_from_csv(csv_path: str | Path) -> np.ndarray:
    """Load good unit IDs from a CSV (robust to header/no‑header). Returns 1D int array.
    Accepts a single column named 'unit_id' or the first column as IDs.
    """
    import numpy as np
    from pathlib import Path

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"good_units CSV not found: {p}")

    # Try headered CSV first
    try:
        arr = np.genfromtxt(str(p), delimiter=",", names=True, dtype=None, encoding="utf-8")
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            # Prefer column that looks like unit id
            for k in arr.dtype.names:
                kl = k.lower()
                if kl == "unit_id" or ("unit" in kl and "id" in kl) or kl in ("unit", "id"):
                    col = np.asarray(arr[k]).ravel()
                    return col.astype(int)
            # Fallback to the first column
            first = arr.dtype.names[0]
            col = np.asarray(arr[first]).ravel()
            return col.astype(int)
    except Exception:
        pass

    # Fallbacks: no header / different delimiter
    for delim in [",", None, "	", ";", " "]:
        try:
            arr = np.genfromtxt(str(p), delimiter=delim, dtype=float)
            if arr.ndim == 0:
                return np.array([int(arr.item())])
            if arr.ndim == 1:
                return arr.astype(int)
            if arr.ndim == 2:
                return arr[:, 0].astype(int)
        except Exception:
            continue

    raise ValueError(f"Could not parse good_units from CSV: {p}")


def _resolve_units_for_plot(npz_path: str | Path, units: Optional[Sequence[int]], good_units: Optional[Sequence[int] | str | bool]) -> Optional[np.ndarray]:
    """Intersect requested units with good_units (if provided). If good_units is:
      - Sequence[int]: use directly
      - str/Path: treat as file OR directory containing 'good_units.csv'
      - True: auto-load '{ks_folder}/good_units.csv' using meta in NPZ
    Returns an array of unit IDs or None (to use all from NPZ).
    """
    import json
    from pathlib import Path
    import numpy as np

    # Start from all units in NPZ
    dat = np.load(npz_path, allow_pickle=True)
    all_units = np.asarray(dat["unit_ids"], dtype=int)
    all_set = set(all_units.tolist())

    if good_units is None:
        # no filtering; keep 'units' as-is
        return np.array(units, dtype=int) if units is not None else None

    # Determine good_set
    good_set: set[int]
    csv_path: Optional[Path] = None

    if isinstance(good_units, (list, tuple, set, np.ndarray)):
        good_set = set(int(u) for u in good_units)
    else:
        # string/bool/path cases
        if isinstance(good_units, bool) and good_units is True:
            # auto from ks_folder in meta
            meta_raw = dat["meta"]
            try:
                meta_raw = meta_raw.item()
            except Exception:
                pass
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            meta = json.loads(meta_raw)
            ks_folder = Path(meta.get("ks_folder", "."))
            csv_path = ks_folder / "good_units.csv"
        else:
            # provided path or directory
            p = Path(str(good_units))
            csv_path = p / "good_units.csv" if p.is_dir() else p

        try:
            good_ids = _load_good_units_from_csv(csv_path)
            good_set = set(int(x) for x in good_ids)
        except Exception as exc:
            print(f"[warn] good_units not applied ({exc}). Using provided 'units' or all units.")
            return np.array(units, dtype=int) if units is not None else None

    # Intersect with available units in NPZ
    good_set &= all_set

    if units is not None:
        final = sorted(good_set & set(int(u) for u in units))
    else:
        final = sorted(good_set)

    if len(final) == 0:
        print("[warn] No units left after good_units filter; using all units.")
        return None
    return np.asarray(final, dtype=int)

# ---- Analysis: PSTH ----

def _event_aligned_spikes(spike_times_s: np.ndarray, events: np.ndarray, t_before: float, t_after: float) -> list[np.ndarray]:
    """Return list of per-trial arrays of spike times aligned to each event (seconds relative to event)."""
    aligned = []
    for ev in events:
        w0, w1 = ev - t_before, ev + t_after
        mask = (spike_times_s >= w0) & (spike_times_s < w1)
        aligned.append(spike_times_s[mask] - ev)
    return aligned


def psth(
    spikes_rel_trials: list[np.ndarray],
    bin_size: float = 0.01,
    smooth_sigma_bins: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSTH (Hz) from per-trial relative spike times.

    Returns (t_centers, rate_hz)
    """
    if not spikes_rel_trials:
        return np.array([]), np.array([])
    t_min = min((x.min() if x.size else 0.0) for x in spikes_rel_trials)
    t_max = max((x.max() if x.size else 0.0) for x in spikes_rel_trials)
    # guard for empty
    if t_min == 0 and t_max == 0:
        return np.array([]), np.array([])

    edges = np.arange(t_min, t_max + bin_size, bin_size)
    counts = np.zeros(edges.size - 1, dtype=float)
    for x in spikes_rel_trials:
        if x.size:
            counts += np.histogram(x, bins=edges)[0]
    rate = counts / (len(spikes_rel_trials) * bin_size)
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        from scipy.ndimage import gaussian_filter1d
        rate = gaussian_filter1d(rate, smooth_sigma_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, rate


def plot_psth(
    npz_path: str | Path,
    unit_id: Optional[object] = None,
    t_before: float = 0.2,
    t_after: float = 0.6,
    bin_size: float = 0.01,
    smooth_sigma_bins: float = 1.0,
):
    """Plot PSTH under three modes based on `unit_id`:

    - int:   plot single-unit PSTH for that unit id
    - str/Path (CSV): treat as path to good_units CSV and plot **grand-average** PSTH
    - None:  plot **grand-average** PSTH across *all* units in the NPZ

    You can also pass a sequence of ints to treat as the set of units for grand-average.
    """
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt

    dat = np.load(npz_path, allow_pickle=True)
    spike_times_s = dat["spike_times_s"]
    spike_unit_ids = dat["spike_unit_ids"].astype(int)
    unit_ids_all = dat["unit_ids"].astype(int)
    icms_sec = dat["icms_sec"]

    # Fixed edges to ensure consistent averaging across units
    edges = np.arange(-t_before, t_after + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_trials = int(icms_sec.size)

    # Resolve which units to include & mode
    mode = "grand"
    if unit_id is None:
        units_list = unit_ids_all
    elif isinstance(unit_id, (int, np.integer)):
        units_list = np.array([int(unit_id)], dtype=int)
        mode = "single"
    elif isinstance(unit_id, (list, tuple, set, np.ndarray)):
        units_list = np.array([int(u) for u in unit_id], dtype=int)
    else:
        # Assume CSV path for good_units
        try:
            csv_path = Path(str(unit_id))
            good_ids = _load_good_units_from_csv(csv_path)
            units_list = np.array([u for u in good_ids if u in set(unit_ids_all.tolist())], dtype=int)
            if units_list.size == 0:
                print(f"[warn] No overlap between good_units in {csv_path} and NPZ units; using all units.")
                units_list = unit_ids_all
        except Exception as exc:
            print(f"[warn] Could not parse units from {unit_id} ({exc}); using all units.")
            units_list = unit_ids_all

    # Helper: compute per-unit rate over fixed edges
    def _unit_psth_rate(st_unit: np.ndarray) -> np.ndarray:
        counts_total = np.zeros(edges.size - 1, dtype=float)
        if st_unit.size:
            for ev in icms_sec:
                rel = st_unit[(st_unit >= ev - t_before) & (st_unit < ev + t_after)] - ev
                if rel.size:
                    counts_total += np.histogram(rel, bins=edges)[0]
        rate = counts_total / (n_trials * bin_size)
        if smooth_sigma_bins and smooth_sigma_bins > 0:
            from scipy.ndimage import gaussian_filter1d
            rate = gaussian_filter1d(rate, smooth_sigma_bins, mode='nearest')
        return rate

    # Compute
    if mode == "single":
        u = int(units_list[0])
        st_unit = spike_times_s[spike_unit_ids == u]
        rate = _unit_psth_rate(st_unit)
        plt.figure()
        plt.title(f"Unit {u} PSTH (n_trials={n_trials})")
        plt.plot(centers, rate, lw=2)
        plt.axvline(0, ls='--')
        plt.xlabel('Time from event (s)')
        plt.ylabel('Firing rate (Hz)')
        plt.show()
        return

    # Grand-average across units
    rates = []
    for u in units_list:
        st_unit = spike_times_s[spike_unit_ids == int(u)]
        rates.append(_unit_psth_rate(st_unit))
    if not rates:
        print("[warn] No units available; nothing to plot.")
        return
    R = np.vstack(rates)
    mean_rate = R.mean(axis=0)

    plt.figure()
    plt.title(f"Grand-average PSTH (units={len(units_list)}, trials={n_trials})")
    plt.plot(centers, mean_rate, lw=2)
    plt.axvline(0, ls='--')
    plt.xlabel('Time from event (s)')
    plt.ylabel('Firing rate (Hz)')
    plt.show()


# ---- Analysis: Trial-aligned trajectories ----

def build_trial_tensor(
    npz_path: str | Path,
    units: Optional[Sequence[int]] = None,
    t_before: float = 0.2,
    t_after: float = 0.6,
    bin_size: float = 0.02,
    smooth_sigma_bins: Optional[float] = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tensor, unit_ids, t_centers) where
    - tensor has shape (n_trials, n_units, n_timebins) with binned/smoothed rates (Hz)
    - unit_ids lists the order of units in the tensor
    - t_centers are the time bin centers relative to event
    """
    dat = np.load(npz_path, allow_pickle=True)
    spike_times_s = dat["spike_times_s"]
    spike_unit_ids_all = dat["spike_unit_ids"].astype(int)
    unit_ids_all = dat["unit_ids"].astype(int)
    icms_sec = dat["icms_sec"]

    if units is None:
        unit_ids = unit_ids_all
    else:
        units = set(int(u) for u in units)
        unit_ids = np.array([u for u in unit_ids_all if u in units], dtype=int)

    edges = np.arange(-t_before, t_after + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0

    n_trials = icms_sec.size
    n_units = unit_ids.size
    n_tbins = centers.size

    tensor = np.zeros((n_trials, n_units, n_tbins), dtype=np.float32)

    # Pre-split spikes per unit for speed
    spikes_by_unit = {u: spike_times_s[spike_unit_ids_all == u] for u in unit_ids}

    for ti, ev in enumerate(icms_sec):
        w0, w1 = ev - t_before, ev + t_after
        for ui, u in enumerate(unit_ids):
            st = spikes_by_unit[u]
            if st.size == 0:
                continue
            rel = st[(st >= w0) & (st < w1)] - ev
            if rel.size:
                counts, _ = np.histogram(rel, bins=edges)
                rates = counts.astype(np.float32) / bin_size
                if smooth_sigma_bins and smooth_sigma_bins > 0:
                    from scipy.ndimage import gaussian_filter1d
                    rates = gaussian_filter1d(rates, smooth_sigma_bins, mode='nearest')
                tensor[ti, ui, :] = rates

    return tensor, unit_ids, centers


def plot_avg_trajectory(
    npz_path: str | Path,
    units: Optional[Sequence[int]] = None,
    good_units: Optional[Sequence[int] | str | bool] = None,
    t_before: float = 0.2,
    t_after: float = 0.6,
    bin_size: float = 0.02,
    smooth_sigma_bins: Optional[float] = 1.0,
    n_components: int = 3,
    show_3d: bool = True,
):
    """Compute PCA across units on the trial‑averaged rates and plot 2D/3D trajectory."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    # Resolve units with optional good_units CSV filtering
    units = _resolve_units_for_plot(npz_path, units, good_units)
    tensor, unit_ids, t = build_trial_tensor(
        npz_path=npz_path,
        units=units,
        t_before=t_before,
        t_after=t_after,
        bin_size=bin_size,
        smooth_sigma_bins=smooth_sigma_bins,
    )
    # Average across trials -> (n_units, n_timebins)
    mean_rates = tensor.mean(axis=0)  # (U, T)
    # z‑score per unit to equalize scale (optional but helpful)
    mu = mean_rates.mean(axis=1, keepdims=True)
    sd = mean_rates.std(axis=1, keepdims=True) + 1e-9
    z = (mean_rates - mu) / sd

    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(z.T)  # (T, C)

    if show_3d and n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=2)
        # Mark event time 0
        zero_idx = np.argmin(np.abs(t))
        ax.scatter(traj[zero_idx, 0], traj[zero_idx, 1], traj[zero_idx, 2], s=60)
        ax.set_title(f"PCA Trajectory (U={len(unit_ids)}, trials={tensor.shape[0]})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1], lw=2)
        zero_idx = np.argmin(np.abs(t))
        plt.scatter(traj[zero_idx, 0], traj[zero_idx, 1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA Trajectory (U={len(unit_ids)}, trials={tensor.shape[0]})")
        plt.show()


# ---- Per-trial population trajectories ----

def plot_trial_trajectories(
    npz_path: str | Path,
    units: Optional[Sequence[int]] = None,
    good_units: Optional[Sequence[int] | str | bool] = None,
    t_before: float = 0.2,
    t_after: float = 0.6,
    bin_size: float = 0.02,
    smooth_sigma_bins: Optional[float] = 1.0,
    n_components: int = 3,
    show_3d: bool = True,
    alpha: float = 0.35,
    linewidth: float = 1.5,
    trials: Optional[Sequence[int]] = None,
):
    """Overlay per-trial neural population trajectories in a shared low-D space.

    Steps:
      1) Build trial × unit × timebin rate tensor.
      2) Z-score per unit across all trials/time to equalize scale.
      3) Fit PCA on the concatenation of all trials/time bins (shared basis).
      4) Project each trial and plot as a line (2D or 3D) overlaid.

    Returns
      traj_trials: np.ndarray with shape (n_trials, n_tbins, n_components)
      t:          time vector (seconds, relative to event)
      pca:        fitted sklearn PCA object if available, else None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Build tensor: (T, U, B)
    units = _resolve_units_for_plot(npz_path, units, good_units)
    tensor, unit_ids, t = build_trial_tensor(
        npz_path=npz_path,
        units=units,
        t_before=t_before,
        t_after=t_after,
        bin_size=bin_size,
        smooth_sigma_bins=smooth_sigma_bins,
    )

    # Select subset of trials if requested
    if trials is not None:
        idx = np.array(list(trials), dtype=int)
        tensor = tensor[idx]

    T, U, B = tensor.shape

    # Z-score per unit over all trials and timebins so units contribute equally
    mu = tensor.mean(axis=(0, 2), keepdims=True)                    # (1, U, 1)
    sd = tensor.std(axis=(0, 2), keepdims=True) + 1e-9
    z = (tensor - mu) / sd                                          # (T, U, B)

    # Prepare data for PCA fit: stack trials and time bins -> (T*B, U)
    Z_fit = z.swapaxes(1, 2).reshape(T * B, U)

    # Fit PCA (prefer sklearn, fallback to SVD if not installed)
    pca = None
    try:
        from sklearn.decomposition import PCA  # type: ignore
        pca = PCA(n_components=n_components)
        pca.fit(Z_fit)
        transform = pca.transform
    except Exception:
        # Minimal PCA via SVD
        mean_ = Z_fit.mean(axis=0, keepdims=True)
        Zc = Z_fit - mean_
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        components = Vt[:n_components].T  # (U, C)
        def transform(Y):
            return (Y - mean_) @ components

    # Project each trial: (B, U) -> (B, C)
    traj_trials = np.zeros((T, B, n_components), dtype=float)
    for ti in range(T):
        Y = z[ti].T  # (B, U)
        traj_trials[ti] = transform(Y)

    # Plot overlay
    zero_idx = int(np.argmin(np.abs(t)))
    if show_3d and n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ti in range(T):
            X = traj_trials[ti]
            ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=linewidth, alpha=alpha)
            ax.scatter(X[zero_idx, 0], X[zero_idx, 1], X[zero_idx, 2], s=20, alpha=0.8)
        ax.set_title(f"Per-trial PCA trajectories (trials={T}, U={U})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()
    else:
        plt.figure()
        for ti in range(T):
            X = traj_trials[ti]
            plt.plot(X[:, 0], X[:, 1], lw=linewidth, alpha=alpha)
            plt.scatter(X[zero_idx, 0], X[zero_idx, 1], s=15, alpha=0.8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Per-trial PCA trajectories (trials={T}, U={U})")
        plt.show()

    return traj_trials, t, pca


# ---- CLI ----
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pack KiloSort4 output + trial onsets into a single NPZ file.")
    p.add_argument("--ks", dest="ks", required=True, help="Path to sorter_output (KiloSort folder)")
    p.add_argument("--icms", dest="icms", required=True, help="Path to a .npy/.npz/.txt file of event times (seconds)")
    p.add_argument("--out", dest="out", default="session.npz", help="Output NPZ path")
    p.add_argument("--units", dest="units", default=None, help="Optional comma-separated unit IDs to include (e.g., 1,2,3)")
    args = p.parse_args()

    ks_folder = Path(args.ks)
    # Load icms_sec from common formats
    icms_path = Path(args.icms)
    if icms_path.suffix.lower() == ".npy":
        icms_sec = np.load(icms_path)
    elif icms_path.suffix.lower() == ".npz":
        with np.load(icms_path) as f:
            # try common keys
            for k in ("icms_sec", "events", "onsets", "arr_0"):
                if k in f:
                    icms_sec = f[k]
                    break
            else:
                raise KeyError("No suitable key found in NPZ for events.")
    else:
        # assume whitespace-separated text
        icms_sec = np.loadtxt(icms_path)

    units = None
    if args.units:
        units = [int(x) for x in args.units.split(",") if x.strip()]

    out = build_npz(ks_folder, icms_sec=icms_sec, outfile=args.out, include_units=units)
    print(f"Saved -> {out}")


