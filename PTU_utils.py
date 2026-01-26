from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


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
        special_bit = (raw & 0x80000000) != 0
        chan = ((raw >> 25) & 0x3F).astype(np.int64)
        tcspc = ((raw >> 10) & 0x7FFF).astype(np.int64)
        sync = (raw & 0x3FF).astype(np.int64)

        ind_overflow = special_bit & (chan == 63)
        special = (special_bit.astype(np.int64) * chan).astype(np.int64)

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


