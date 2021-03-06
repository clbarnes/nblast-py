#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path

import numpy as np
from tqdm import trange
import pandas as pd
from scipy.spatial import cKDTree as KDTree

DOTPROPS = Path("data/dotprops")
fpaths = sorted(DOTPROPS.glob("*.csv"))
N_NEIGHBORS = 5


def read_csv(fpath):
    df = pd.read_csv(fpath, usecols=list(range(1, 8)))
    renames = dict()
    for idx, dim in enumerate("XYZ", 1):
        renames[f"points.{dim}"] = f"point_{dim.lower()}"
        renames[f"vect.{idx}"] = f"tangent_{dim.lower()}"
    df.rename(columns=renames, inplace=True)
    return df


def norm_direction(arr):
    arr[arr[:, 0] < 0] *= -1
    return arr


def eigen(mat, symmetric=False):
    """Return eigenvals, eigenvecs.

    Both are sorted by decreasing order of eigenvalue.
    Eigenvectors are unit-length columns.
    """
    fn = np.linalg.eigh if symmetric else np.linalg.eig
    vals_unsorted, vecs_unsorted = fn(mat)
    idx = vals_unsorted.argsort()[::-1]
    vals = vals_unsorted[idx]
    vecs = vecs_unsorted[:, idx]

    return vals, vecs


def make_tan_alpha(neighbors):
    centered = neighbors - np.mean(neighbors, axis=0)
    inertia = centered.T @ centered

    vals, vecs = eigen(inertia, symmetric=True)
    alpha = (vals[0] - vals[1]) / vals.sum()
    tangent = vecs[:, 0]
    return tangent, alpha


def make_tan_alpha_svd(neighbors):
    centered = neighbors - np.mean(neighbors, axis=0)
    u, s, vh = np.linalg.svd(centered)
    tangent = vh[0]
    vals = s ** 2
    alpha = (vals[0] - vals[1]) / vals.sum()
    return tangent, alpha


def make_dotprops(df: pd.DataFrame, k=N_NEIGHBORS):
    point_headers = ["point_" + dim for dim in "xyz"]
    points = df[point_headers].to_numpy()
    tree = KDTree(points)
    alpha = []
    tangents = []
    for idx, row in enumerate(points):
        _, neighbor_idxs = tree.query(row, k=k)
        assert len(neighbor_idxs) == k
        neighbors = points[neighbor_idxs]
        # tangent, a = make_tan_alpha_svd(neighbors)
        tangent, a = make_tan_alpha(neighbors)
        # assert (
        #     np.allclose(tangent, tangent2) or
        #     np.allclose(tangent, -1 * tangent2)
        # )
        # assert np.allclose(a, a2)
        alpha.append(a)
        tangents.append(tangent)

    tangents_arr = np.array(tangents)
    new_df = pd.DataFrame(points, columns=point_headers)
    new_df["alpha"] = alpha
    for d, t in zip("xyz", tangents_arr.T):
        new_df["tangent_" + d] = t
    return new_df, tree


class DotProps:
    def __init__(self, df, tree=None):
        self.df = df
        self.kdtree = tree or KDTree(self.points.to_numpy())

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_points(cls, df):
        new, tree = make_dotprops(df)
        return cls(new, tree)

    @classmethod
    def from_csv(cls, fpath):
        return cls.from_points(read_csv(fpath))

    @property
    def points(self):
        return self.df[["point_" + d for d in "xyz"]]

    @property
    def tangents(self):
        return self.df[["tangent_" + d for d in "xyz"]]

    @property
    def alpha(self):
        return self.df.alpha

    def _fast_dist_dots(self, other: DotProps):
        other_tangents = other.tangents.to_numpy()
        self_points = self.points.to_numpy()
        self_tangents = self.tangents.to_numpy()

        fast_dists, fast_idxs = other.kdtree.query(self_points)
        fast_tangents = other_tangents[fast_idxs]
        fast_dotprods = np.abs((self_tangents * fast_tangents).sum(axis=1))
        return fast_dists, fast_dotprods

    def dist_dots(self, other: DotProps):
        return self._fast_dist_dots(other)


def parse_interval(s):
    return float(s.strip("([])").split(",")[-1])


def intervals_to_bins(interval_strs):
    breaks = [-np.inf]
    for s in interval_strs:
        upper = parse_interval(s)
        breaks.append(upper)

    breaks[-1] = np.inf
    return np.array(breaks, float)


def csv_to_score_fn(fpath):
    df = pd.read_csv(fpath, index_col=0)
    return table_to_score_fn(
        [parse_interval(s) for s in df.index],
        [parse_interval(s) for s in df.columns],
        df.to_numpy(),
    )


def table_to_score_fn(dist_thresholds, dot_thresholds, cells):
    dist_thresholds = list(dist_thresholds)
    dist_thresholds[-1] = np.inf
    dist_bins = np.array(dist_thresholds, float)

    dot_thresholds = list(dot_thresholds)
    dot_thresholds[-1] = np.inf
    dot_bins = np.array(dot_thresholds, float)

    def fn(dist, dot):
        return cells[
            np.digitize(dist, dist_bins),
            np.digitize(dot, dot_bins),
        ]

    return fn


class Nblaster:
    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.dotprops = []

    def append(self, dotprops):
        idx = len(self.dotprops)
        self.dotprops.append(dotprops)
        return idx

    def self_hit(self, q_idx):
        return len(self.dotprops[q_idx]) * self.score_fn(0, 1.0)

    def query_target(self, q_idx, t_idx, symmetric=False, normalized=False):
        if q_idx == t_idx:
            if normalized:
                return 1
            return self.self_hit(q_idx)

        score = 0
        dists, dots = self.dotprops[q_idx].dist_dots(self.dotprops[t_idx])
        score = self.score_fn(dists, dots).sum()

        if normalized:
            score /= self.self_hit(q_idx)
        if symmetric:
            b_dists, b_dots = self.dotprops[t_idx].dist_dots(
                self.dotprops[q_idx]
            )
            backward = self.score_fn(b_dists, b_dots).sum()
            if normalized:
                backward /= self.self_hit(t_idx)
            score = (score + backward) / 2
        return score

    def all_v_all(self, symmetric=False, normalized=False):
        rows = []
        for q_idx in trange(len(self.dotprops)):
            rows.append([])
            for t_idx in trange(len(self.dotprops)):
                if symmetric and q_idx > t_idx:
                    score = rows[t_idx][q_idx]
                else:
                    score = self.query_target(
                        q_idx, t_idx, symmetric, normalized
                    )
                rows[-1].append(score)
        # transpose so that query is in column header, target in index
        # like R, I think
        return np.array(rows).T


def make_score_csv():
    nblaster = Nblaster(csv_to_score_fn("data/smat_fcwb.csv"))
    idxs = {
        nblaster.append(DotProps.from_csv(fpath)): fpath.stem
        for fpath in fpaths
    }
    out = nblaster.all_v_all()
    names = list(idxs.values())
    df = pd.DataFrame(out, index=names, columns=names)
    return df


def read_score_csv(fpath):
    orig = pd.read_csv(fpath, index_col=0)
    for idx in (0, 1):
        orig.sort_index(idx, inplace=True)
    return orig


def drop_diff_names(*dfs):
    first, *others = dfs
    keep = set(first.columns).intersection(first.index)
    for df in others:
        keep = keep.intersection(first.columns)
        keep = keep.intersection(first.index)
    names = sorted(keep)
    out = []
    for df in dfs:
        df = df[names]
        df = df.loc[names]
        out.append(df)
    return tuple(out)


def equivalent_tangents(*arrs):
    first, *others = arrs
    for other in others:
        if not (np.allclose(first, other) or np.allclose(first, other * -1)):
            print(f"first: {first}")
            print(f"other: {other}")
            return False
    return True


def check_tangents():
    points = np.array([
        [329.679_962_158_203, 72.718_803_405_761_7, 31.028_469_085_693_4],
        [328.647_399_902_344, 73.046_119_689_941_4, 31.537_061_691_284_2],
        [335.219_879_150_391, 70.710_479_736_328_1, 30.398_145_675_659_2],
        [332.611_389_160_156, 72.322_929_382_324_2, 30.887_334_823_608_4],
        [331.770_782_470_703, 72.434_440_612_793, 31.169_372_558_593_8],
    ])
    tan1, alpha1 = make_tan_alpha(points)
    tan2, alpha2 = make_tan_alpha_svd(points)

    assert equivalent_tangents(tan1, tan2)


def check_all_scores():
    orig = read_score_csv("data/kcscores.csv")
    # new = read_score_csv("data/py_kcscores.csv")
    new = make_score_csv()

    orig, new = drop_diff_names(orig, new)
    close = np.all(np.abs((orig - new)) < 0.0001)
    if close:
        print("SUCCESS, python matches R")
    else:
        print("FAILED, python does not match R")


def check_single():
    # query on index, target on columns
    orig_T = read_score_csv("data/kcscores.csv").T
    nblaster = Nblaster(csv_to_score_fn("data/smat_fcwb.csv"))
    idxs = {
        nblaster.append(DotProps.from_csv(fpath)): fpath.stem
        for fpath in fpaths[:2]
    }
    score = nblaster.query_target(0, 1)
    orig_score = orig_T.loc[idxs[0]][idxs[1]]
    assert np.abs(score - orig_score) < 0.0001


if __name__ == "__main__":
    check_tangents()
    check_all_scores()
