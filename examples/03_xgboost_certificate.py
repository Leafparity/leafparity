"""A conversion that is correct: XGBoost -> ONNX with onnxmltools.
leafparity certifies it as EQUIVALENT over every possible input, missing values included."""
import warnings

import numpy as np
import onnxmltools
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType

from leafparity import analyze
from leafparity.report import to_text

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 8)) * 100
X[rng.random(X.shape) < 0.05] = np.nan
y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) > 0).astype(int)
model = xgb.XGBClassifier(n_estimators=200, max_depth=6).fit(X, y)
onx = onnxmltools.convert_xgboost(model, initial_types=[("X", FloatTensorType([None, 8]))], target_opset=15)
print(to_text(analyze(model, onx, background=X[:300])))
