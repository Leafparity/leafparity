"""Reproduces the sklearn-onnx documentation example "Issues when switching to float"
(https://onnx.ai/sklearn-onnx/auto_tutorial/plot_ebegin_float_double.html).

The tutorial measures the discrepancy by comparing predictions on a test set.
leafparity replaces the sample with a proof over *every* possible input.
"""
import warnings

import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.sklapi import CastTransformer
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from leafparity import analyze
from leafparity.report import to_text

warnings.filterwarnings("ignore")

X, y = make_regression(10000, 10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
for i in range(X.shape[1]):
    X_train[:, i] *= 2 ** i
    X_test[:, i] *= 2 ** i
X_train, X_test = np.rint(X_train), np.rint(X_test)

model = Pipeline([("scaler", StandardScaler()), ("dt", DecisionTreeRegressor(max_depth=10, random_state=0))])
model.fit(X_train, y_train)
onx = to_onnx(model, X_train[:1].astype(np.float32))

# --- what the tutorial does: compare on a test set
sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
X32 = X_test.astype(np.float32)
sampled = np.abs(model.predict(X32) - sess.run(None, {"X": X32})[0].ravel()).max()
print(f"Test-set comparison (what the tutorial does): max |difference| = {sampled:.2f}\n")

# --- what leafparity does: every float32 input, no missing values
a = analyze(model, onx, input_dtype="float32", allow_missing=False, background=X32)
print(to_text(a, max_findings=2))

# --- the tutorial's fix
fixed = Pipeline([("cast64", CastTransformer(dtype=np.float64)), ("scaler", StandardScaler()),
                  ("cast", CastTransformer()), ("dt", DecisionTreeRegressor(max_depth=10, random_state=0))])
fixed.fit(X_train, y_train)
onx_fixed = to_onnx(fixed, X_train[:1].astype(np.float32))
for dtype, nan in (("float32", False), ("float64", False), ("float32", True)):
    r = analyze(fixed, onx_fixed, input_dtype=dtype, allow_missing=nan, background=X32)
    print(f"Tutorial's fix, {dtype} inputs, missing values {'allowed' if nan else 'excluded'}: "
          f"{r.verdict['status']} - {r.verdict['headline']}\n")
