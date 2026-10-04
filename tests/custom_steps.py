"""A user-defined transformer, importable by the command line tests (PYTHONPATH=tests)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ClipOutliers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)
