"""scikit-learn (>= 1.3) decision trees route missing values (NaN) with
`missing_go_to_left`, even for trees trained without NaN.  Here the model converted by
skl2onnx sends NaN the other way at some nodes, so any NaN reaching the served model
can change its prediction."""
import warnings

import numpy as np
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier

from leafparity import analyze
from leafparity.report import to_text

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)
X = rng.normal(size=(5000, 6))
y = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype(int)
model = RandomForestClassifier(50, max_depth=8, random_state=0).fit(X, y)   # no NaN in training
onx = to_onnx(model, X[:1].astype(np.float32), options={id(model): {"zipmap": False}})
print(to_text(analyze(model, onx, background=X[:300]), max_findings=2))
