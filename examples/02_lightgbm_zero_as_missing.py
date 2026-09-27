"""LightGBM trained with zero_as_missing=True, converted with onnxmltools.

The converter does not model LightGBM's missing_type=Zero, so an input of exactly
0.0 follows a different path in the ONNX model.  Test-set comparisons miss it
whenever the test data has no exact zeros in the affected features.
"""
import warnings

import lightgbm as lgb
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from leafparity import analyze
from leafparity.report import to_text

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2)
X = rng.normal(size=(3000, 4)) * 100
y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=3000)
model = lgb.LGBMRegressor(n_estimators=30, num_leaves=15, zero_as_missing=True, verbose=-1).fit(X, y)
onx = onnxmltools.convert_lightgbm(model, initial_types=[("X", FloatTensorType([None, 4]))], target_opset=15)

import onnxruntime as ort
sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
Xt = rng.normal(size=(5000, 4)) * 100
print("Random test set, max |difference|:",
      np.abs(model.predict(Xt) - sess.run(None, {"X": Xt.astype(np.float32)})[0].ravel()).max(), "\n")
print(to_text(analyze(model, onx, background=X[:300]), max_findings=3))
