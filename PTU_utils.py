from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from math import isfinite

# Additional imports for MIET calculations
try:
    import scipy.io
    from scipy.interpolate import CubicSpline
except ImportError:
    scipy = None
    CubicSpline = None

try:
    import h5py
except ImportError:
    h5py = None

# MIET calibration helpers (from MIET_main.py in the same folder)
try:
    from MIET_main import (
        MetalsDB, brightness_dipole, miet_calc, fresnel, dipoleL,
        lifetimeL, hash_waveguide_mode
    )
except ImportError as exc:  # pragma: no cover - import must succeed in runtime env
    raise ImportError(
        "MIET_main.py must be importable from the same folder for MIET calibration."
    ) from exc


def _load_metals_db(metals_path: Optional[Union[str, Path]] = None) -> MetalsDB:
    """Load metals.mat via the MIET_main MetalsDB helper."""
    path = Path(metals_path) if metals_path is not None else Path(__file__).with_name("metals.mat")
    if not path.exists():
        raise FileNotFoundError(f"Metals database not found at {path}")
    return MetalsDB(str(path))


# =============================================================================
# PicoQuant PTU: Tag types (from MATLAB)
# =============================================================================
TY_EMPTY8       = 0xFFFF0008
TY_BOOL8        = 0x00000008
TY_INT8         = 0x10000008
TY_BITSET64     = 0x11000008
TY_COLOR8       = 0x12000008
TY_FLOAT8       = 0x20000008
TY_TDATETIME    = 0x21000008
TY_FLOAT8_ARRAY = 0x2001FFFF
TY_ANSISTRING   = 0x4001FFFF
TY_WIDESTRING   = 0x4002FFFF
TY_BINARY_BLOB  = 0xFFFFFFFF


# =============================================================================
# PicoQuant TTTR Record types (from MATLAB)
# =============================================================================
rtPicoHarpT3     = 0x00010303
rtPicoHarpT2     = 0x00010203
rtHydraHarpT3    = 0x00010304
rtHydraHarpT2    = 0x00010204
rtHydraHarp2T3   = 0x01010304
rtHydraHarp2T2   = 0x01010204
rtTimeHarp260NT3 = 0x00010305
rtTimeHarp260NT2 = 0x00010205
rtTimeHarp260PT3 = 0x00010306
rtTimeHarp260PT2 = 0x00010206
rtMultiHarpT3    = 0x00010307
rtMultiHarpT2    = 0x00010207


# =============================================================================
# Binary IO helpers
# =============================================================================
def _read_bytes(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected EOF while reading PTU.")
    return b


def _read_i32(f) -> int:
    return int(np.frombuffer(_read_bytes(f, 4), dtype="<i4")[0])


def _read_u32(f) -> int:
    return int(np.frombuffer(_read_bytes(f, 4), dtype="<u4")[0])


def _read_i64(f) -> int:
    return int(np.frombuffer(_read_bytes(f, 8), dtype="<i8")[0])


def _read_f64(f) -> float:
    return float(np.frombuffer(_read_bytes(f, 8), dtype="<f8")[0])


def _to_namespace(x: Any) -> Any:
    """Recursively convert dict->SimpleNamespace for MATLAB-struct-like access."""
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_namespace(v) for v in x]
    return x


def _set_head_value(store: Dict[str, Any], key: str, idx: int, value: Any) -> None:
    """
    Mimic MATLAB PTU_Read_Head behavior:
      - TagIdx == -1: store scalar at head.TagIdent
      - TagIdx > -1 : store indexed at head.TagIdent(idx+1) (MATLAB 1-based)
    In Python we store indexed tags as lists with 0-based indices.
    """
    if idx < 0:
        store[key] = value
        return

    if key not in store or not isinstance(store[key], list):
        store[key] = []

    lst: List[Any] = store[key]
    needed = (idx + 1) - len(lst)
    if needed > 0:
        lst.extend([None] * needed)
    lst[idx] = value


# =============================================================================
# PTU header reader (MATLAB PTU_Read_Head port)
# =============================================================================
def PTU_Read_Head(path: Union[str, "os.PathLike[str]"]) -> SimpleNamespace:
    """
    Read PicoQuant Unified TTTR (.ptu) header and return a MATLAB-like struct.

    Notes
    -----
    - This version ignores Leica/Falcon oddities as requested (no tag-ident cleanup).
    - head.length is the byte offset where TTTR records begin, matching MATLAB logic:
      head.length = ftell(fid) + 8 after reading the Header_End tag header.
    """
    head_store: Dict[str, Any] = {}

    with open(path, "rb") as f:
        magic = _read_bytes(f, 8).decode("ascii", errors="ignore").rstrip("\x00 ").strip()
        if magic != "PQTTTR":
            raise ValueError("Magic invalid, this is not a PTU file.")
        version = _read_bytes(f, 8).decode("ascii", errors="ignore").rstrip("\x00 ").strip()

        head_store["Magic"] = magic
        head_store["Version"] = version

        # TagHead.Ident (32), TagHead.Idx (int32), TagHead.Typ (uint32)
        tag_ident = _read_bytes(f, 32).decode("ascii", errors="ignore").rstrip("\x00 ").strip()
        tag_idx = _read_i32(f)
        tag_typ = _read_u32(f)

        while tag_ident != "Header_End":
            if tag_typ == TY_EMPTY8:
                _ = _read_i64(f)
                value = None

            elif tag_typ == TY_BOOL8:
                value = bool(_read_i64(f) != 0)

            elif tag_typ in (TY_INT8, TY_BITSET64, TY_COLOR8):
                value = _read_i64(f)

            elif tag_typ == TY_FLOAT8:
                value = _read_f64(f)

            elif tag_typ == TY_FLOAT8_ARRAY:
                nbytes = _read_i64(f)
                n = int(nbytes // 8)
                value = np.frombuffer(_read_bytes(f, 8 * n), dtype="<f8").copy()

            elif tag_typ == TY_TDATETIME:
                # Keep raw float; convert to datetime outside if needed.
                value = _read_f64(f)

            elif tag_typ == TY_ANSISTRING:
                n = _read_i64(f)
                s = _read_bytes(f, int(n)).decode("latin1", errors="ignore")
                value = s.rstrip("\x00")

            elif tag_typ == TY_WIDESTRING:
                nbytes = _read_i64(f)
                raw = _read_bytes(f, int(nbytes))
                # Remove zero bytes as MATLAB does
                value = bytes([b for b in raw if b != 0]).decode("latin1", errors="ignore")

            elif tag_typ == TY_BINARY_BLOB:
                nbytes = _read_i64(f)
                _ = _read_bytes(f, int(nbytes))
                value = None

            else:
                # Unknown: safest is to read 8 bytes (most fixed-size types)
                value = _read_bytes(f, 8)

            _set_head_value(head_store, tag_ident, tag_idx, value)

            # next tag
            tag_ident = _read_bytes(f, 32).decode("ascii", errors="ignore").rstrip("\x00 ").strip()
            tag_idx = _read_i32(f)
            tag_typ = _read_u32(f)

        # MATLAB: head.length = ftell(fid) + 8
        # We have read Header_End tag header but not its 8-byte payload.
        head_store["length"] = int(f.tell() + 8)

    return _to_namespace(head_store)


# =============================================================================
# TTTR record reader (MATLAB PTU_Read port)
# =============================================================================
def PTU_Read(
    path: Union[str, "os.PathLike[str]"],
    cnts: Union[int, Tuple[int, int], List[int]],
    head: Optional[SimpleNamespace] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, SimpleNamespace]:
    """
    Read TTTR records from a PTU file.

    Parameters
    ----------
    cnts:
      - int N         : read N records from the start of TTTR stream
      - (start, N)    : seek to record #start (1-based like MATLAB) then read N

    Returns
    -------
    sync, tcspc, chan, special, num, loc, head

    Notes
    -----
    - num is the number of 32-bit records read from disk (including overflow records).
    - overflow records are removed from the returned arrays, like MATLAB.
    """
    if head is None:
        head = PTU_Read_Head(path)

    # normalize cnts
    if cnts is None:
        start, nread = 0, 0
    elif isinstance(cnts, int):
        start, nread = 0, int(cnts)
    else:
        if len(cnts) < 2:
            start, nread = 0, int(cnts[0])
        else:
            start, nread = int(cnts[0]), int(cnts[1])

    if nread <= 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty, 0, 0, head

    rec_type = int(getattr(head, "TTResultFormat_TTTRRecType"))

    # determine wraparound + parsing
    if rec_type == rtPicoHarpT3:
        WRAPAROUND = 65536
        mode = "PicoHarpT3"
    elif rec_type == rtPicoHarpT2:
        WRAPAROUND = 210698240
        mode = "PicoHarpT2"
    elif rec_type in (rtMultiHarpT3, rtHydraHarpT3, rtHydraHarp2T3, rtTimeHarp260NT3, rtTimeHarp260PT3):
        WRAPAROUND = 1024
        mode = "HHT3_like"
    elif rec_type == rtHydraHarpT2:
        WRAPAROUND = 33552000
        mode = "HydraHarpT2"
    elif rec_type in (rtMultiHarpT2, rtHydraHarp2T2, rtTimeHarp260NT2, rtTimeHarp260PT2):
        WRAPAROUND = 33554432
        mode = "HHT2_like"
    else:
        raise ValueError(f"Illegal/unsupported TTTRRecType: {rec_type:#x}")

    with open(path, "rb") as f:
        f.seek(int(head.length), 0)
        if start > 1:
            f.seek(4 * (start - 1), 1)
        raw = np.fromfile(f, dtype="<u4", count=nread)

    num = int(raw.size)
    if num == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty, 0, 0, head

    if mode == "PicoHarpT3":
        sync = (raw & 0xFFFF).astype(np.int64)
        chan = ((raw >> 28) & 0xF).astype(np.int64)
        tcspc = ((raw >> 16) & 0xFFF).astype(np.int64)
        special = ((chan == 15).astype(np.int64) * (tcspc & 0xF)).astype(np.int64)
        ind_overflow = (chan == 15) & ((tcspc & 0xF) == 0)

    elif mode == "PicoHarpT2":
        sync = (raw & 0x0FFFFFFF).astype(np.int64)
        tcspc = (raw & 0xF).astype(np.int64)
        chan = ((raw >> 28) & 0xF).astype(np.int64)
        special = ((chan == 15).astype(np.int64) * (tcspc & 0xF)).astype(np.int64)
        ind_overflow = (chan == 15) & ((tcspc & 0xF) == 0)

    elif mode == "HHT3_like":
        # Keep everything in small dtypes to avoid big temporary allocations
        special_bit = (raw & np.uint32(0x80000000)) != 0  # bool
        chan_u8 = ((raw >> 25) & np.uint32(0x3F)).astype(np.uint8)      # 0..63
        tcspc_u16 = ((raw >> 10) & np.uint32(0x7FFF)).astype(np.uint16) # 0..32767
        sync_u16 = (raw & np.uint32(0x3FF)).astype(np.uint16)           # 0..1023
        
        # overflow markers: special bit set and chan==63
        ind_overflow = special_bit & (chan_u8 == np.uint8(63))
        
        # "special" code: 0 for photons, else marker code (chan value)
        # Avoid bool->int64 conversion; store as uint8
        special_u8 = np.zeros(raw.shape[0], dtype=np.uint8)
        special_u8[special_bit] = chan_u8[special_bit]
        
        # Promote only what you must later (sync needs wraparound accumulation => int64)
        chan = chan_u8.astype(np.int64, copy=False)
        tcspc = tcspc_u16.astype(np.int64, copy=False)
        sync = sync_u16.astype(np.int64, copy=False)
        special = special_u8.astype(np.int64, copy=False)

    elif mode in ("HydraHarpT2", "HHT2_like"):
        sync = (raw & 0x01FFFFFF).astype(np.int64)
        chan = ((raw >> 25) & 0x3F).astype(np.int64)
        tcspc = (chan & 0xF).astype(np.int64)
        special_bit = ((raw >> 31) & 0x1) != 0

        ind_overflow = special_bit & (chan == 63)
        special = (special_bit.astype(np.int64) * chan).astype(np.int64)

    else:
        raise RuntimeError("Internal parse mode error.")

    # overflow correction + removal
    if np.any(ind_overflow):
        tmp = sync[ind_overflow].copy()
        tmp[tmp == 0] = 1
        sync[ind_overflow] = tmp

        corr = (WRAPAROUND * np.cumsum(ind_overflow.astype(np.int64) * sync)).astype(np.int64)
        sync = sync + corr

        keep = ~ind_overflow
        sync = sync[keep]
        tcspc = tcspc[keep]
        chan = chan[keep]
        special = special[keep]

        valid_idx = np.where(keep)[0]
        if valid_idx.size:
            last_valid = int(valid_idx[-1])
            loc = num - (last_valid + 1)
        else:
            loc = num
    else:
        loc = 0

    return sync, tcspc, chan, special, num, loc, head


# =============================================================================
# MATLAB mHist equivalent
# =============================================================================
def mhist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    MATLAB mHist(values, bin) for integer consecutive bins.
    Returns counts aligned to bins.
    """
    values = np.asarray(values)
    bins = np.asarray(bins)

    if values.size == 0:
        return np.zeros(len(bins), dtype=np.int64)

    b0 = int(bins[0])
    b1 = int(bins[-1])

    v = values.astype(np.int64, copy=False)
    v = v[(v >= b0) & (v <= b1)]
    if v.size == 0:
        return np.zeros(len(bins), dtype=np.int64)

    counts = np.bincount(v - b0, minlength=(b1 - b0 + 1))
    return counts.astype(np.int64, copy=False)


def _channel_list_from_data(chan: np.ndarray, special: np.ndarray, min_occurrence: int = 10) -> np.ndarray:
    """
    Choose detector channels with at least `min_occurrence` photons, ignoring special records.
    """
    chan = np.asarray(chan)
    special = np.asarray(special).astype(bool)
    good = chan[~special]
    if good.size == 0:
        return np.array([], dtype=chan.dtype)
    u, c = np.unique(good, return_counts=True)
    return u[c >= min_occurrence]


# =============================================================================
# Harp_tcspc (MATLAB Harp_tcspc port, PTU only)
# =============================================================================
@dataclass
class HarpTCSPCResult:
    bin: np.ndarray
    tcspcdata: np.ndarray
    head: SimpleNamespace
    Resolution: float
    Deadtime: float
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
    Python port of MATLAB Harp_tcspc for .ptu files.

    Parameters
    ----------
    name : PTU filename
    resolution : desired TCSPC bin width (seconds). None => native.
    deadtime : detector deadtime (seconds). None => 0.
    photons : chunk size (# records to read per call). None => 1e6.
    cache : save/load sidecar npz cache "<name>.ht3tcspc.npz"
    emulate_matlab_deadtime_histogram :
        Default False: remove photons violating deadtime (intended meaning).
        True: emulate MATLAB deadtime branch that appears to histogram violating photons.
    """
    name = Path(name)
    if name.suffix.lower() != ".ptu":
        raise ValueError(f"Only .ptu supported here (got {name.suffix}).")

    if photons is None:
        photons = int(1e6)

    Deadtime = float(0.0 if deadtime is None else deadtime)

    head = PTU_Read_Head(str(name))
    SyncRate = float(head.TTResult_SyncRate)
    Timeunit = 1.0 / SyncRate
    native_Resolution = float(head.MeasDesc_Resolution)

    # resolution / channel divisor
    if resolution is None:
        chdiv = 1
        Resolution = native_Resolution
    else:
        resolution = float(resolution)
        chdiv = int(np.ceil(resolution / native_Resolution))
        Resolution = max(native_Resolution, resolution)

    NChannels = int(np.ceil(Timeunit / Resolution))
    bin_edges = np.arange(NChannels, dtype=np.int64)

    cachefile = name.with_suffix(name.suffix + ".ht3tcspc.npz")
    if cache and cachefile.exists():
        dat = np.load(cachefile, allow_pickle=True)
        cached_Resolution = float(dat["Resolution"])
        cached_Deadtime = float(dat["Deadtime"])
        if abs(cached_Resolution - Resolution) <= 1e-12 and cached_Deadtime == Deadtime:
            return HarpTCSPCResult(
                bin=dat["bin"],
                tcspcdata=dat["tcspcdata"],
                head=dat["head"].item(),
                Resolution=cached_Resolution,
                Deadtime=cached_Deadtime,
                nRemovedPhotons=int(dat["nRemovedPhotons"]),
            )
        cachefile.unlink(missing_ok=True)

    cnt = 0
    num = 1

    dind: Optional[np.ndarray] = None
    tcspcdata: Optional[np.ndarray] = None
    nRemovedPhotons = 0

    while num > 0:
        sync, tcspc, chan, special, num, loc, head = PTU_Read(str(name), [cnt + 1, photons], head)
        cnt += int(num)
        if num <= 0 or sync.size == 0:
            continue

        sync = np.asarray(sync)
        tcspc = np.asarray(tcspc)
        chan = np.asarray(chan)
        special = np.asarray(special).astype(bool)

        if dind is None:
            dind = _channel_list_from_data(chan, special, min_occurrence=10)
            if dind.size == 0:
                dind = np.unique(chan[~special])
            dnum = int(dind.size)
            tcspcdata = np.zeros((NChannels, dnum), dtype=np.int64)

        assert dind is not None and tcspcdata is not None

        # remove special records
        sync = sync[~special]
        tcspc = tcspc[~special]
        chan = chan[~special]
        if sync.size == 0:
            continue

        # apply requested TCSPC resolution
        tcspc = np.rint(tcspc / chdiv).astype(np.int64)

        # deadtime filtering (optional)
        keep = np.ones(tcspc.shape[0], dtype=bool)
        if Deadtime > 0 and tcspc.size > 1:
            # pseudo absolute time in seconds
            tttr = sync.astype(np.float64) * Timeunit + tcspc.astype(np.float64) * Resolution
            difftttr = np.diff(np.concatenate(([0.0], tttr)))

            viol = difftttr < Deadtime
            viol_idx = np.where(viol)[0]
            viol_idx = viol_idx[viol_idx > 0]

            if viol_idx.size:
                t1 = sync[viol_idx - 1].astype(np.int64)
                t2 = sync[viol_idx].astype(np.int64)

                tgate = np.ceil((tcspc[viol_idx - 1] * Resolution + Deadtime) / Timeunit).astype(np.int64)
                xind = (t2 - t1) <= tgate
                bad = viol_idx[xind]

                if emulate_matlab_deadtime_histogram:
                    keep[:] = False
                    keep[bad] = True
                else:
                    keep[bad] = False
                    nRemovedPhotons += int(bad.size)

        tcspc_k = tcspc[keep]
        chan_k = chan[keep]

        # histogram per detector channel
        for jj, ch in enumerate(dind):
            vals = tcspc_k[chan_k == ch]
            if vals.size:
                tcspcdata[:, jj] += mhist(vals, bin_edges)

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


# =============================================================================
# Asynchronous TTTR correlator (your tttr2xfcs, packaged cleanly)
# =============================================================================
def tttr2xfcs(y: np.ndarray, num: np.ndarray, Ncasc: int, Nsub: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast TTTR multipletau-like correlator (Wahl/Gregor/Patting/Enderlein style).

    Parameters
    ----------
    y : (N,) photon arrival times (integer-like)
    num : (N,) or (N,C) weights per photon time
    Ncasc : number of cascade levels
    Nsub : sub-levels per cascade

    Returns
    -------
    auto : (M, C, C) correlation values
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
        y_unique, k1 = np.unique(y, return_index=True)
        y = y_unique

        tmp = np.cumsum(num, axis=0)
        num = np.diff(np.vstack([np.zeros((1, num.shape[1])), tmp[k1, :]]), axis=0)

        for k in range(Nsub):
            nmpd2 = Nsub + k + 1
            if y.size < 2 * nmpd2:
                break

            shift += delta
            lag = int(np.rint(shift / delta))

            i1 = np.in1d(y, y + lag, assume_unique=True)
            i2 = np.in1d(y + lag, y, assume_unique=True)

            autotime[k + j * Nsub] = shift

            if i1.any() and i2.any():
                auto[k + j * Nsub, :, :] = (num[i1, :].T @ num[i2, :]) / delta

            out_len = max(out_len, k + j * Nsub + 1)

        y = np.ceil(0.5 * y).astype(np.int64)
        delta *= 2.0

    auto = auto[:out_len, :, :]
    autotime = autotime[:out_len]

    # edge correction
    for j in range(auto.shape[0]):
        if dtt != autotime[j]:
            auto[j, :, :] = auto[j, :, :] * dtt / (dtt - autotime[j])

    good = autotime != 0
    return auto[good, :, :], autotime[good]


# =============================================================================
# MATLAB helper translations and MIET+PTU pipeline
# =============================================================================
def interp1_nan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """MATLAB interp1(...,'linear',NaN) equivalent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    return np.interp(xq, x, y, left=np.nan, right=np.nan)


def tttr2bin(sync_ticks: np.ndarray, macrobin_ticks: float) -> np.ndarray:
    """Bin photon arrivals by macro time bins (integer ticks)."""
    s = np.asarray(sync_ticks, dtype=np.int64)
    if s.size == 0:
        return np.zeros((0,), dtype=np.int64)

    w = int(np.round(macrobin_ticks))
    if w <= 0:
        raise ValueError("macrobin_ticks must be >= 1 tick")

    idx = (s // w).astype(np.int64)
    counts = np.bincount(idx, minlength=idx.max() + 1)
    return counts.astype(np.int64)


def tttr2bintcspc(
    sync_ticks: np.ndarray,
    params: Tuple[float, int],
    tcspc_shifted: np.ndarray,
    n_micro_bins: int,
) -> np.ndarray:
    """Bin TCSPC microtimes inside macro bins (MATLAB tttr2bintcspc analogue)."""
    s = np.asarray(sync_ticks, dtype=np.int64)
    t = np.asarray(tcspc_shifted, dtype=np.int64)
    if s.size == 0:
        return np.zeros((n_micro_bins, 0), dtype=np.int64)

    macro_w = int(np.round(params[0]))
    micro_rebin = int(params[1])
    if macro_w <= 0:
        raise ValueError("macrobin width must be >=1")
    if micro_rebin <= 0:
        raise ValueError("micro rebin must be >=1")

    valid = t > 0
    if not np.any(valid):
        n_macro = int((s.max() // macro_w) + 1)
        return np.zeros((n_micro_bins, n_macro), dtype=np.int64)

    s = s[valid]
    t = t[valid]

    macro_idx = (s // macro_w).astype(np.int64)
    micro_idx = (t // micro_rebin).astype(np.int64)

    keep = (micro_idx >= 0) & (micro_idx < n_micro_bins)
    macro_idx = macro_idx[keep]
    micro_idx = micro_idx[keep]

    n_macro = int(macro_idx.max() + 1) if macro_idx.size else int((s.max() // macro_w) + 1)
    out = np.zeros((n_micro_bins, n_macro), dtype=np.int64)
    np.add.at(out, (micro_idx, macro_idx), 1)
    return out


def photobleach_fit_exp(normtrace: np.ndarray) -> np.ndarray:
    """Fit A*exp(-t/tau)+C to a normalized trace and return the fitted curve."""
    y = np.asarray(normtrace, dtype=float)
    x = np.arange(y.size, dtype=float)

    y = np.where(np.isfinite(y), y, np.nan)
    m = np.nanmedian(y)
    y = np.where(np.isnan(y), m, y)

    c_grid = np.linspace(0.0, np.percentile(y, 30) * 0.9, 60)
    best = None
    best_sse = np.inf

    for c in c_grid:
        yy = y - c
        if np.any(yy <= 0):
            continue
        logy = np.log(yy)
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, logy, rcond=None)
        a, b = coef
        yhat = np.exp(a + b * x) + c
        sse = float(np.mean((y - yhat) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = (a, b, c)

    if best is None:
        return np.ones_like(y)

    a, b, c = best
    z = np.exp(a + b * x) + c
    z /= np.mean(z[z > 0])
    return z


def mseb_like(x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, label: str):
    """Minimal shaded-error plot replacement for MATLAB mseb."""
    x = np.asarray(x, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    plt.semilogx(x, y_mean, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)


def _load_gold_titan_tables(metals: MetalsDB) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wavelength, gold, titan arrays sampled on the DB grid."""
    wl = metals.wavelength
    gold = metals.get_index(20, wl)   # 20: gold in MetalsDB mapping
    titan = metals.get_index(80, wl)  # 80: titan in MetalsDB mapping
    return wl, gold, titan


def run_miet_ptu_pipeline(
    ptu_path: Union[str, Path],
    *,
    tau0: float = 2.9,
    qy0: float = 0.6,
    tau1: float = 2.2,
    lamex_um: float = 0.640,
    lamem_um: float = 0.690,
    NA: float = 1.49,
    glass_n: float = 1.52,
    n1: float = 1.33,
    n: float = 1.33,
    top_n: float = 1.46,
    d0_um: Tuple[float, float, float, float] = (2e-3, 10e-3, 1e-3, 10e-3),
    d_um: float = 3e-1,
    d1_um: Tuple[float, ...] = (),
    curveType: int = 2,
    al_res: int = 100,
    tbin_s: float = 1e-4,
    photons_per_chunk: int = int(1e6),
    cutoff_ns: float = 10.0,
    shift_ns: float = 0.3,
    micro_rebin: int = 8,
    Ncasc: int = 13,
    Nsub: int = 6,
    nbunches: int = 10,
    metals_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Full MIET + PTU workflow translated from MATLAB."""

    ptu_path = Path(ptu_path)
    metals = _load_metals_db(metals_path)
    wavelength, gold, titan = _load_gold_titan_tables(metals)

    qy1 = qy0 * tau1 / tau0
    qy = qy1
    tau_free = tau1

    lamem_nm = lamem_um * 1e3
    idx = int(np.argmin(np.abs(wavelength - lamem_nm)))
    n_ti = titan[idx]
    n_au = gold[idx]

    n0 = np.array([glass_n, n_ti, n_au, n_ti, top_n], dtype=complex)
    d0 = np.array(d0_um, dtype=float) * 1e3
    d1 = np.array(d1_um, dtype=float) * 1e3
    d = float(d_um * 1e3)

    z_nm, lifecurve = miet_calc(
        al_res,
        lamem_nm,
        n0,
        n,
        n1,
        d0,
        d,
        d1,
        qy,
        tau_free,
        False,
        curveType,
        metals_db=metals,
    )
    z_nm = np.asarray(z_nm, dtype=float)
    lifecurve = np.asarray(lifecurve, dtype=float)

    ind = np.isfinite(lifecurve) & np.isfinite(z_nm)
    z_nm = z_nm[ind]
    lifecurve = lifecurve[ind]

    fac = 2 * np.pi / lamem_um
    zfac = (z_nm * 1e-3) * fac

    d0fac = np.array(d0_um) * fac
    dfac = d_um * fac
    d1fac = (np.array(d1_um) * fac) if len(d1_um) else np.array([])

    br_out = brightness_dipole(zfac, n0, n, n1, d0fac, dfac, d1fac, NA, qy, False)
    br = np.asarray(br_out[-2], dtype=float)  # third element: rotating dipole brightness
    br = interp1_nan(z_nm[np.isfinite(br)], br[np.isfinite(br)], z_nm)

    maxz_life = int(np.nanargmax(lifecurve))

    harp = harp_tcspc(ptu_path)
    tcspc_full = harp.tcspcdata
    pos = int(np.argmax(np.sum(tcspc_full, axis=1)))

    head = harp.head
    SyncRate = float(head.TTResult_SyncRate)
    Resolution_s = float(head.MeasDesc_Resolution)
    Ngate = int(np.ceil((1.0 / SyncRate) / Resolution_s))
    bin_tcspc = np.arange(Ngate + 1, dtype=np.int64)

    cnts = 0
    flag = True
    ttrace_list: List[np.ndarray] = []
    tmptcspc = np.zeros(Ngate, dtype=np.int64)
    tmptau_list: List[np.ndarray] = []
    sync_offset = 0

    macrobin_ticks = tbin_s * SyncRate

    shift = int(np.round((shift_ns * 1e-9) / Resolution_s))
    ind1 = pos + shift
    cutoff_bins_native = int(np.ceil((cutoff_ns * 1e-9) / Resolution_s))
    ind2 = ind1 + cutoff_bins_native

    n_micro_bins = int(np.ceil((cutoff_ns * 1e-9) / Resolution_s))

    while flag:
        sync, tcspc, chan, special, num, loc, _ = PTU_Read(str(ptu_path), [cnts + 1, photons_per_chunk], head)

        sync = np.asarray(sync, dtype=np.int64)
        tcspc = np.asarray(tcspc, dtype=np.int64)
        special = np.asarray(special, dtype=np.int64)

        keep = special == 0
        sync = sync[keep]
        tcspc = tcspc[keep]

        cnts += int(num)
        flag = num > 0

        if sync.size == 0:
            continue

        sync_cont = sync + sync_offset
        sync_offset += int(sync[-1])

        tmptcspc += mhist(tcspc, np.arange(Ngate, dtype=np.int64))

        gate = (tcspc > ind1) & (tcspc <= ind2)
        tcspc_shifted = (tcspc - ind1) * gate.astype(np.int64)

        tau_block = tttr2bintcspc(
            sync_cont,
            (macrobin_ticks, micro_rebin),
            tcspc_shifted,
            n_micro_bins=n_micro_bins,
        )
        tmptau_list.append(tau_block)

        ttrace_list.append(tttr2bin(sync_cont, macrobin_ticks))

    ttrace = np.concatenate(ttrace_list, axis=0) if ttrace_list else np.zeros((0,), dtype=np.int64)
    tmptau = np.concatenate(tmptau_list, axis=1) if tmptau_list else np.zeros((n_micro_bins, 0), dtype=np.int64)

    meantrace = float(np.mean(ttrace[ttrace > 0])) if np.any(ttrace > 0) else float(np.mean(ttrace))
    normtrace = ttrace / meantrace if meantrace != 0 else np.zeros_like(ttrace, dtype=float)

    ztrace = photobleach_fit_exp(normtrace)
    normtrace = normtrace / ztrace

    delay_ns = (np.arange(tmptau.shape[0], dtype=float) * Resolution_s * 1e9 * micro_rebin)

    grid_lts = np.linspace(0.1, 3.0, 200)
    grid_bs = np.linspace(0.0, 0.2, 60)

    ltmat, bmat = np.meshgrid(grid_lts, grid_bs, indexing="xy")
    ltvec = ltmat.ravel()
    bvec = bmat.ravel()

    delay_col = delay_ns[:, None]
    exp_part = np.exp(-delay_col / ltvec[None, :])
    exp_part /= np.sum(exp_part, axis=0, keepdims=True)
    pmf = (bvec[None, :] / delay_ns.size) + (1.0 - bvec[None, :]) * exp_part
    logpmf = np.log(np.clip(pmf, 1e-300, None))

    nbunch = 100
    n_macro = tmptau.shape[1]
    edges = np.linspace(0, n_macro, nbunch + 1).astype(int)

    grid_ind = np.zeros(n_macro, dtype=int)
    for k in range(nbunch):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        scores = -(tmptau[:, a:b].T @ logpmf)
        grid_ind[a:b] = np.argmin(scores, axis=1)

    gridMLE_tau = ltvec[grid_ind]

    htrace_mle = interp1_nan(lifecurve[: maxz_life + 1], z_nm[: maxz_life + 1], gridMLE_tau)

    counts = tmptau.astype(float)
    denom = np.sum(counts, axis=0)
    denom = np.where(denom == 0, np.nan, denom)

    Et = np.sum(counts * delay_ns[:, None], axis=0) / denom
    Et2 = np.sum(counts * (delay_ns[:, None] ** 2), axis=0) / denom
    meantau = np.sqrt(np.maximum(Et2 - Et**2, 0.0))

    htrace_var = interp1_nan(lifecurve[: maxz_life + 1], z_nm[: maxz_life + 1], meantau)

    pos2 = int(np.argmax(tmptcspc))
    ind1b = pos2 + shift

    t_ns = bin_tcspc[ind1b:Ngate] * Resolution_s * 1e9
    y_tail = tmptcspc[ind1b:Ngate].astype(float)

    def _fit_biexp(t, y):
        t1_grid = np.linspace(0.2, 5.0, 60)
        t2_grid = np.linspace(0.2, 5.0, 60)

        best = None
        best_sse = np.inf
        for t1 in t1_grid:
            e1 = np.exp(-t / t1)
            for t2 in t2_grid:
                e2 = np.exp(-t / t2)
                A = np.vstack([e1, e2, np.ones_like(t)]).T
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                yhat = A @ coef
                sse = float(np.mean((y - yhat) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best = (t1, t2, coef)
        t1, t2, coef = best
        a1, a2, c = coef
        return float(t1), float(t2), float(a1), float(a2), float(c)

    t1, t2, a1, a2, c_bg = _fit_biexp(t_ns, y_tail)
    amps = np.clip(np.array([a1, a2], dtype=float), 0.0, None)
    taus = np.array([t1, t2], dtype=float)
    tau_avg = float(np.sum(amps) / np.sum(amps / np.clip(taus, 1e-12, None))) if np.sum(amps) > 0 else float(np.mean(taus))

    z_avg = float(interp1_nan(lifecurve[np.isfinite(lifecurve)], z_nm[np.isfinite(z_nm)], np.array([tau_avg]))[0])

    iz = int(np.argmin(np.abs(z_nm - z_avg)))
    br_fac = float(br[iz]) if np.isfinite(br[iz]) else float(np.nanmean(br))
    norm_br = br / br_fac

    maxz_br = int(np.nanargmax(norm_br))
    htrace_int = interp1_nan(norm_br[: maxz_br + 1], z_nm[: maxz_br + 1], normtrace)
    bhmean = meantrace * br_fac

    def _finite_mask(x):
        return np.isfinite(x)

    mask1 = _finite_mask(htrace_int)
    tpoints = np.where(mask1)[0]
    h1 = htrace_int[mask1]
    i1 = ttrace[mask1]

    mask2 = _finite_mask(htrace_var)
    tpoints2 = np.where(mask2)[0]
    h2 = htrace_var[mask2]

    mask3 = _finite_mask(htrace_mle)
    tpoints3 = np.where(mask3)[0]
    h3 = htrace_mle[mask3]

    def _bunch_edges(L, nb):
        return np.round(np.linspace(0, L, nb + 1)).astype(int)

    e1 = _bunch_edges(len(tpoints), nbunches)
    e2 = _bunch_edges(len(tpoints2), nbunches)
    e3 = _bunch_edges(len(tpoints3), nbunches)

    auto = None
    auto2 = None
    auto3 = None
    autoi = None
    autotime = None

    for k in range(nbunches):
        a, b = e1[k], e1[k + 1]
        a2, b2 = e2[k], e2[k + 1]
        a3, b3 = e3[k], e3[k + 1]

        tmpauto, autotime = tttr2xfcs(tpoints[a:b], h1[a:b], Ncasc, Nsub)
        tmpauto = tmpauto.squeeze()
        if auto is None:
            auto = np.zeros((tmpauto.size, nbunches), dtype=float)
        auto[:, k] = tmpauto

        tmpauto2, _ = tttr2xfcs(tpoints2[a2:b2], h2[a2:b2], Ncasc, Nsub)
        tmpauto2 = tmpauto2.squeeze()
        if auto2 is None:
            auto2 = np.zeros((tmpauto2.size, nbunches), dtype=float)
        auto2[:, k] = tmpauto2

        tmpauto3, _ = tttr2xfcs(tpoints3[a3:b3], h3[a3:b3], Ncasc, Nsub)
        tmpauto3 = tmpauto3.squeeze()
        if auto3 is None:
            auto3 = np.zeros((tmpauto3.size, nbunches), dtype=float)
        auto3[:, k] = tmpauto3

        tmpautoi, _ = tttr2xfcs(tpoints[a:b], i1[a:b], Ncasc, Nsub)
        tmpautoi = tmpautoi.squeeze()
        if autoi is None:
            autoi = np.zeros((tmpautoi.size, nbunches), dtype=float)
        autoi[:, k] = tmpautoi

    autotime = autotime.squeeze().astype(float) if autotime is not None else np.zeros((0,), dtype=float)

    def _norm_end(x: np.ndarray) -> np.ndarray:
        return x / x[-1:, :] - 1.0

    auto_n = _norm_end(auto)
    auto2_n = _norm_end(auto2)
    auto3_n = _norm_end(auto3)
    autoi_n = _norm_end(autoi)

    tau_s = autotime * tbin_s
    x = tau_s[:-1]

    plt.figure()
    mseb_like(x, np.mean(auto3_n[:-1, :], axis=1), np.std(auto3_n[:-1, :], axis=1), label="dACF_MLE")
    mseb_like(x, np.mean(auto2_n[:-1, :], axis=1), np.std(auto2_n[:-1, :], axis=1), label="dACF_var")
    mseb_like(x, np.mean(auto_n[:-1, :], axis=1), np.std(auto_n[:-1, :], axis=1), label="dACF_b(h)")
    mseb_like(
        x,
        np.mean(autoi_n[:-1, :], axis=1) * bhmean,
        np.std(autoi_n[:-1, :], axis=1) * bhmean,
        label="ACF (scaled)",
    )

    plt.xlabel("t / (s)")
    plt.ylabel("g(t)")
    plt.grid(True, which="both")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.figure()
    plt.semilogx(tau_s, np.mean(auto3 / auto3[0:1, :], axis=1), label="dACF_MLE")
    plt.semilogx(tau_s, np.mean(auto2 / auto2[0:1, :], axis=1), label="dACF_var")
    plt.semilogx(tau_s, np.mean(auto / auto[0:1, :], axis=1), label="dACF_b(h)")
    plt.semilogx(tau_s, np.mean(autoi / autoi[0:1, :], axis=1), label="ACF")
    plt.xlabel("t / (s)")
    plt.ylabel("g(t) / g(0)")
    plt.grid(True, which="both")
    plt.legend(frameon=False)
    plt.tight_layout()

    return {
        "z_nm": z_nm,
        "lifecurve": lifecurve,
        "br": br,
        "ttrace": ttrace,
        "normtrace": normtrace,
        "tmptcspc": tmptcspc,
        "tmptau": tmptau,
        "htrace_int": htrace_int,
        "htrace_var": htrace_var,
        "htrace_mle": htrace_mle,
        "auto": auto,
        "auto2": auto2,
        "auto3": auto3,
        "autoi": autoi,
        "autotime": autotime,
        "tau_s": tau_s,
        "z_avg_nm": z_avg,
        "tau_avg_ns": tau_avg,
        "bhmean": bhmean,
        "bi_exp_bg": c_bg,
    }


if __name__ == "__main__":
    example_ptu = Path(r"C:\Users\narai\OneDrive\Documents\MIET\fromTao\data for share\BOTTOM-22.ptu")
    if example_ptu.exists():
        _ = run_miet_ptu_pipeline(example_ptu)
        plt.show()
    else:
        print("Example PTU path does not exist; edit __main__ to run the pipeline.")


