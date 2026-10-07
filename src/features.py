import numpy as np
import pandas as pd
from scipy import signal, stats

from src.config import PROCESSED, SPLITS, SAMPLING_RATE_HZ
from src.data import load_raw, feature_cols

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
FULL_BAND = (0.5, 40)      # the data has almost no power above 40 Hz

# "size" features grow with how big the signal swings; "shape" features do not.
SIZE_FEATURES = [f"logpow_{b}" for b in BANDS] + ["log_activity", "line_length", "peak_to_peak"]
SHAPE_FEATURES = ([f"rel_{b}" for b in BANDS]
                  + ["logratio_theta_alpha", "logratio_slow_fast",
                     "spectral_edge_95", "spectral_centroid", "spectral_entropy",
                     "hjorth_mobility", "hjorth_complexity", "zero_crossing_rate",
                     "skewness", "kurtosis"])
FEATURES = SIZE_FEATURES + SHAPE_FEATURES


def reconstruct_segments(df, cols):
    """Join each segment's 23 chunks, in chunk order, into one signal of 23 x 178 samples."""
    df = df.sort_values(["segment", "chunk"])
    assert (df.groupby("segment").size() == 23).all()
    names = sorted(df["segment"].unique())
    return names, df[cols].to_numpy(dtype=float).reshape(len(names), -1)


def segment_features(x, fs=SAMPLING_RATE_HZ):
    """The core feature set for one signal. Uses only this signal, nothing from other segments."""
    xc = np.asarray(x, dtype=float) - np.mean(x)            # remove the average level
    d1 = np.diff(xc)                                         # first difference (slope)
    d2 = np.diff(d1)

    # spectrum (Welch: average the spectra of overlapping 512-sample windows)
    freqs, psd = signal.welch(xc, fs=fs, nperseg=512)
    df_hz = freqs[1] - freqs[0]
    in_band = (freqs >= FULL_BAND[0]) & (freqs < FULL_BAND[1])
    f, p = freqs[in_band], psd[in_band]
    total = p.sum() * df_hz
    power = {b: psd[(freqs >= lo) & (freqs < hi)].sum() * df_hz for b, (lo, hi) in BANDS.items()}

    out = {}
    for b in BANDS:
        out[f"logpow_{b}"] = np.log10(power[b] + 1e-12)      # size: absolute power, log scale
        out[f"rel_{b}"] = power[b] / total                   # shape: share of the total
    out["logratio_theta_alpha"] = np.log10(power["theta"] / power["alpha"])
    out["logratio_slow_fast"] = np.log10((power["delta"] + power["theta"]) / (power["alpha"] + power["beta"]))
    out["spectral_edge_95"] = f[np.searchsorted(np.cumsum(p) / p.sum(), 0.95)]
    out["spectral_centroid"] = (f * p).sum() / p.sum()
    pn = p / p.sum()
    out["spectral_entropy"] = -(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn))      # 0 to 1

    # time domain
    var0, var1, var2 = xc.var(), d1.var(), d2.var()
    out["log_activity"] = np.log10(var0)                                              # Hjorth activity
    out["hjorth_mobility"] = np.sqrt(var1 / var0)
    out["hjorth_complexity"] = np.sqrt(var2 / var1) / out["hjorth_mobility"]
    out["line_length"] = np.abs(d1).mean()
    out["zero_crossing_rate"] = np.mean(np.signbit(xc[1:]) != np.signbit(xc[:-1]))
    out["peak_to_peak"] = np.ptp(x)
    out["skewness"] = stats.skew(x)
    out["kurtosis"] = stats.kurtosis(x)
    return out


def extract_features(signals):
    """One row of features per signal. Columns follow the order in FEATURES."""
    return pd.DataFrame([segment_features(x) for x in signals])[FEATURES]


def build_feature_tables():
    """Compute features for all 500 segments and save the dev and hold-out tables separately."""
    df = load_raw()
    names, signals = reconstruct_segments(df, feature_cols(df))
    feats = extract_features(signals)
    feats.insert(0, "segment", names)
    feats.insert(1, "y", df.groupby("segment")["y"].first().loc[names].to_numpy())
    groups = pd.read_csv(SPLITS / "groups_all.csv").set_index("segment")["group"]
    feats.insert(2, "group", groups.loc[names].to_numpy())
    hold_ids = set(pd.read_csv(SPLITS / "holdout_segments_allclass.csv")["segment"])
    is_hold = feats["segment"].isin(hold_ids)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dev, hold = feats[~is_hold].reset_index(drop=True), feats[is_hold].reset_index(drop=True)
    dev.to_csv(PROCESSED / "features_dev.csv", index=False)
    hold.to_csv(PROCESSED / "features_holdout.csv", index=False)
    return dev, hold