"""Joint PCA scenario model with conditional sampling.

Each training day is the vector x_d = (b_1, ..., b_J) obtained by stacking J
daily profiles of length T (e.g. load, PV, log signal and, optionally, the
day-ahead PV forecast). Each block is centred and divided by its overall
standard deviation; the first K principal components are retained and the
model is interpreted as a probabilistic PCA (Tipping and Bishop, 1999),
x = mean + W z + noise, z ~ N(0, I_K), noise ~ N(0, sigma^2 I).

Conditioning on any subset of observed entries (the values observed so far
today and/or the whole day-ahead forecast) gives a Gaussian posterior on z,
from which the unobserved entries are sampled.
"""

import numpy as np

from .system import T

LOG_FLOOR = 1.0


class ScenarioModel:
    def __init__(self, blocks, log_blocks=(), K=5, caps=None):
        """blocks: list of (n, T) arrays; log_blocks: indices modelled in log scale;
        caps: dict block index -> (lower, upper) bounds applied after sampling."""
        self.J = len(blocks)
        self.log_blocks = set(log_blocks)
        X = np.hstack([self._fwd(j, b) for j, b in enumerate(blocks)])
        self.n = len(X)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        self.scale = np.repeat([Xc[:, j * T : (j + 1) * T].std() + 1e-9 for j in range(self.J)], T)
        _, sig, Vt = np.linalg.svd(Xc / self.scale, full_matrices=False)
        lam = sig**2 / max(self.n - 1, 1)
        self.K = min(K, len(lam))
        self.lam = lam[: self.K]
        self.explained = self.lam.sum() / lam.sum()
        D = self.J * T
        self.sigma2 = max(lam[self.K :].sum() / (D - self.K), 1e-8)
        self.W = Vt[: self.K].T * np.sqrt(np.maximum(self.lam - self.sigma2, 1e-12))
        self.caps = caps or {}
        # periods in which a block is always zero in the training data (e.g. PV at night)
        self.zero = [np.abs(b).max(axis=0) <= 1e-9 for b in blocks]

    def _fwd(self, j, v):
        return np.log(np.maximum(v, LOG_FLOOR)) if j in self.log_blocks else np.asarray(v, float)

    def _inv(self, Z):
        X = Z * self.scale + self.mean
        out = []
        for j in range(self.J):
            b = X[:, j * T : (j + 1) * T]
            if j in self.log_blocks:
                b = np.exp(b)
            lo, hi = self.caps.get(j, (None, None))
            if lo is not None or hi is not None:
                b = np.clip(b, lo, hi)
            b[:, self.zero[j]] = 0.0
            out.append(b)
        return out

    def sample(self, n, rng, obs=None):
        """Sample n daily profiles. obs: dict block index -> observed values of
        the first len(values) periods of that block (len T = fully observed).
        Returns a list of J arrays (n, T); observed entries are returned as observed."""
        obs = {j: np.asarray(v, float) for j, v in (obs or {}).items() if len(v) > 0}
        if not obs:
            z = rng.normal(size=(n, self.K))
            return self._inv(z @ self.W.T)
        idx = np.concatenate([np.arange(len(v)) + j * T for j, v in obs.items()])
        xo = np.concatenate([self._fwd(j, v) for j, v in obs.items()])
        r = (xo - self.mean[idx]) / self.scale[idx]
        Wo = self.W[idx]
        cov = np.linalg.inv(np.eye(self.K) + Wo.T @ Wo / self.sigma2)
        mz = cov @ Wo.T @ r / self.sigma2
        L = np.linalg.cholesky((cov + cov.T) / 2)
        z = mz + rng.normal(size=(n, self.K)) @ L.T
        out = self._inv(z @ self.W.T)
        for j, v in obs.items():
            out[j][:, : len(v)] = v
        return out
