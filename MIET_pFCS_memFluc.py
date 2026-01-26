from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


# ----------------------------
# Helpers (mHist equivalent)
# ----------------------------
def mhist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    MATLAB mHist(values, bin) equivalent for integer-valued bins.
    bins is expected to be consecutive integers (e.g., 0..N-1).
    Returns counts aligned to `bins`.
    """
    if values.size == 0:
        return np.zeros(len(bins), dtype=np.int64)

    b0 = int(bins[0])
    b1 = int(bins[-1])
    v = values.astype(np.int64, copy=False)

    # keep only in-range
    v = v[(v >= b0) & (v <= b1)]
    if v.size == 0:
        return np.zeros(len(bins), dtype=np.int64)

    # bincount expects 0-based
    counts = np.bincount(v - b0, minlength=(b1 - b0 + 1))
    return counts.astype(np.int64, copy=False)


def _channel_list_from_data(chan: np.ndarray, special: np.ndarray, min_occurrence: int = 10) -> np.ndarray:
    """
    Rough translation of:
        [dind,m]= unique(sort(chan(~special)),'legacy');
        occurence = diff([0;vertcat(m)]);
        dind(occurence<10)=[];
    but implemented sanely as "channels with >= min_occurrence photons".
    """
    good = chan[~special.astype(bool)]
    if good.size == 0:
        return np.array([], dtype=chan.dtype)

    u, c = np.unique(good, return_counts=True)
    u = u[c >= min_occurrence]
    return u


# ----------------------------
# Core: Harp_tcspc in Python
# ----------------------------
@dataclass
class HarpTCSPCResult:
    bin: np.ndarray              # shape (NChannels,)
    tcspcdata: np.ndarray         # shape (NChannels, n_detectors)
    head: object                  # whatever PTU_Read_Head returns
    Resolution: float             # seconds
    Deadtime: float               # seconds
    nRemovedPhotons: int


def harp_tcspc(
    name: Union[str, Path],
    resolution: Optional[float] = None,
    deadtime: Optional[float] = None,
    photons: Optional[int] = None,
    *,
    cache: bool = True,
    emulate_matlab_deadtime_histogram: bool = False,
) -> HarpTCSPCResult:
    """
    Python translation of MATLAB:
        [bin,tcspcdata,head] = Harp_tcspc(name,resolution,deadtime,photons)

    Parameters
    ----------
    name : path to .ptu (and optionally .ht3 if you wire HT3 readers)
    resolution : desired TCSPC bin width (seconds). If None => native resolution.
    deadtime : detector deadtime (seconds). If None => 0 (no filtering).
    photons : chunk size (#records) to read at a time. If None => 1e6.

    cache : if True, saves/loads a sidecar cache file "<name>.ht3tcspc.npz"
    emulate_matlab_deadtime_histogram :
        If False (default): photons inside deadtime are REMOVED (intended behavior).
        If True: emulate the MATLAB deadtime branch that seems to histogram the
        *violating* photons (likely unintended in the original).
    """

    name = Path(name)
    if photons is None:
        photons = int(1e6)

    if deadtime is None:
        deadtime = 0.0
    Deadtime = float(deadtime)

    ext = name.suffix.lower()
    if ext != ".ptu":
        raise ValueError(f"Only .ptu is implemented here (got {ext}). Add HT3 similarly if needed.")

    # ---- MIET_calc PTU readers ----
    # Adjust these imports to your repo layout if needed.
    # Common patterns:
    #   from MIET_calc.ptu import PTU_Read_Head, PTU_Read
    #   from miet_calc.ptu import PTU_Read_Head, PTU_Read
    try:
        from MIET_calc.ptu import PTU_Read_Head, PTU_Read  # type: ignore
    except Exception:
        from miet_calc.ptu import PTU_Read_Head, PTU_Read  # type: ignore

    head = PTU_Read_Head(str(name))
    SyncRate = float(head.TTResult_SyncRate)
    Timeunit = 1.0 / SyncRate
    native_Resolution = float(head.MeasDesc_Resolution)  # seconds
    NCounts = int(head.TTResult_NumberOfRecords)

    # Handle requested histogram resolution (channel divisor)
    if resolution is None:
        chdiv = 1
        Resolution = native_Resolution
    else:
        resolution = float(resolution)
        chdiv = int(np.ceil(resolution / native_Resolution))
        Resolution = max(native_Resolution, resolution)

    NChannels = int(np.ceil(Timeunit / Resolution))
    bin_edges = np.arange(NChannels, dtype=np.int64)  # MATLAB uses bin = 0:NChannels-1

    # Cache file (MATLAB uses .ht3tcspc, keep same spirit but safer as .npz)
    cachefile = name.with_suffix(name.suffix + ".ht3tcspc.npz")

    if cache and cachefile.exists():
        dat = np.load(cachefile, allow_pickle=True)
        cached_Resolution = float(dat["Resolution"])
        cached_Deadtime = float(dat["Deadtime"])

        if abs(cached_Resolution - Resolution) > 1e-12 or cached_Deadtime != Deadtime:
            cachefile.unlink(missing_ok=True)
        else:
            return HarpTCSPCResult(
                bin=dat["bin"],
                tcspcdata=dat["tcspcdata"],
                head=dat["head"].item(),
                Resolution=cached_Resolution,
                Deadtime=cached_Deadtime,
                nRemovedPhotons=int(dat["nRemovedPhotons"]),
            )

    # Streaming read loop
    cnt = 0
    num = 1
    dind = None
    tcspcdata = None
    nRemovedPhotons = 0

    while num > 0:
        # PTU_Read signature in your MATLAB: PTU_Read(name, [cnt+1 photons], head)
        sync, tcspc, chan, special, num = PTU_Read(str(name), [cnt + 1, photons], head)

        # Make arrays
        sync = np.asarray(sync)
        tcspc = np.asarray(tcspc)
        chan = np.asarray(chan)
        special = np.asarray(special).astype(bool)

        cnt += int(num)
        if num <= 0 or sync.size == 0:
            continue

        # Determine detector channels once
        if dind is None:
            dind = _channel_list_from_data(chan, special, min_occurrence=10)
            if dind.size == 0:
                # Fall back to all non-special unique chans
                dind = np.unique(chan[~special])
            dnum = int(dind.size)
            tcspcdata = np.zeros((NChannels, dnum), dtype=np.int64)

        assert tcspcdata is not None and dind is not None

        # Remove special records
        sync = sync[~special]
        tcspc = tcspc[~special]
        chan = chan[~special]

        if sync.size == 0:
            continue

        # Downsample tcspc channels if resolution was coarsened
        tcspc = np.rint(tcspc / chdiv).astype(np.int64)

        # Optionally apply deadtime filtering
        keep = np.ones(tcspc.shape[0], dtype=bool)
        if Deadtime > 0 and tcspc.size > 1:
            # Pseudo absolute time in seconds (same idea as MATLAB)
            tttr = sync.astype(np.float64) * Timeunit + tcspc.astype(np.float64) * Resolution
            difftttr = np.diff(np.concatenate(([0.0], tttr)))

            # Candidate second photons arriving too soon after previous photon
            viol = difftttr < Deadtime
            viol_idx = np.where(viol)[0]  # indices of "second photons"

            # MATLAB adds an additional sync-gating condition; keep it.
            viol_idx = viol_idx[viol_idx > 0]  # ensure idx-1 is valid
            if viol_idx.size:
                t1 = sync[viol_idx - 1].astype(np.int64)
                t2 = sync[viol_idx].astype(np.int64)

                # "nearest sync after deadtime" (MATLAB uses tcspc(prev)*resolution + deadtime)
                tgate = np.ceil((tcspc[viol_idx - 1] * Resolution + Deadtime) / Timeunit).astype(np.int64)

                xind = (t2 - t1) <= tgate
                bad = viol_idx[xind]

                if emulate_matlab_deadtime_histogram:
                    # Emulate MATLAB-looking behavior: histogram only these "bad" photons
                    keep[:] = False
                    keep[bad] = True
                else:
                    # Intended behavior: remove these "bad" photons
                    keep[bad] = False
                    nRemovedPhotons += int(bad.size)

        tcspc_k = tcspc[keep]
        chan_k = chan[keep]

        # Histogram per detector channel
        for jj, ch in enumerate(dind):
            vals = tcspc_k[chan_k == ch]
            if vals.size:
                tcspcdata[:, jj] += mhist(vals, bin_edges)

    # Save cache
    if cache:
        np.savez(
            cachefile,
            bin=bin_edges,
            tcspcdata=tcspcdata,
            head=np.array(head, dtype=object),
            Resolution=np.array(Resolution, dtype=np.float64),
            Deadtime=np.array(Deadtime, dtype=np.float64),
            nRemovedPhotons=np.array(nRemovedPhotons, dtype=np.int64),
        )

    return HarpTCSPCResult(
        bin=bin_edges,
        tcspcdata=tcspcdata,
        head=head,
        Resolution=Resolution,
        Deadtime=Deadtime,
        nRemovedPhotons=nRemovedPhotons,
    )


# ----------------------------
# Correlator: tttr2xfcs
# ----------------------------
def tttr2xfcs(y: np.ndarray, num: np.ndarray, Ncasc: int, Nsub: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asynchronous TTTR correlator (Wahl/Gregor/Patting/Enderlein multipletau-like),
    matching the structure of your Python snippet, but with correct NumPy calls.

    Parameters
    ----------
    y : (N,) photon arrival times (integer-like). Can be sync ticks or any discrete time.
    num : (N,) for single channel weights OR (N, C) for C channels weights.
          For classic channel assignment, num is boolean or counts per photon-time.
    Ncasc : number of cascade levels
    Nsub : number of sub-levels per cascade

    Returns
    -------
    auto : (M, C, C) correlation values (unnormalised as in your code, then corrected by dtt/(dtt-tau))
    autotime : (M,) tau values (same units as y)
    """
    y = np.asarray(y)
    num = np.asarray(num)

    if y.size == 0:
        return np.zeros((0, 1, 1), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    dtt = float(np.max(y) - np.min(y))
    y = np.rint(y).astype(np.int64)

    if num.ndim == 1:
        num = num[:, np.newaxis]

    auto = np.zeros(((Ncasc + 1) * Nsub, num.shape[1], num.shape[1]), dtype=np.float64)
    autotime = np.zeros(((Ncasc + 1) * Nsub,), dtype=np.float64)

    shift = 0.0
    delta = 1.0

    out_len = 0

    for j in range(Ncasc):
        # unique photon times + indices
        y_unique, k1 = np.unique(y, return_index=True)
        y = y_unique

        # sum weights in each unique bin
        tmp = np.cumsum(num, axis=0)
        num = np.diff(np.vstack([np.zeros((1, num.shape[1])), tmp[k1, :]]), axis=0)

        for k in range(Nsub):
            nmpd2 = Nsub + k + 1
            if y.size < 2 * nmpd2:
                break

            shift += delta
            lag = int(np.rint(shift / delta))

            # membership tests for y and y+lag
            i1 = np.in1d(y, y + lag, assume_unique=True)
            i2 = np.in1d(y + lag, y, assume_unique=True)

            autotime[k + j * Nsub] = shift

            if i1.any() and i2.any():
                jin = (num[i1, :].T @ num[i2, :]) / delta
                auto[k + j * Nsub, :, :] = jin

            out_len = max(out_len, k + j * Nsub + 1)

        # coarse-grain time axis
        y = np.ceil(0.5 * y).astype(np.int64)
        delta *= 2.0

    auto = auto[:out_len, :, :]
    autotime = autotime[:out_len]

    # edge correction term
    for j in range(auto.shape[0]):
        if dtt != autotime[j]:
            auto[j, :, :] = auto[j, :, :] * dtt / (dtt - autotime[j])

    # remove trailing zeros in autotime (if any)
    good = autotime != 0
    autotime = autotime[good]
    auto = auto[good, :, :]

    return auto, autotime

