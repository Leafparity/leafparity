# Audit of documented conversion recipes

`audit.py` builds original/ONNX pairs the way the skl2onnx and onnxmltools documentation shows (default settings, float32 input) and checks each pair with leafparity, once on finite inputs and once on all inputs including NaN.

```
pip install leafparity[all] skl2onnx onnxmltools
python audit.py --quick   # under a minute
python audit.py           # 126 pairs, 252 checks, about 12 minutes
```

Result of the full run on 2026-10-03 (xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.9.1, skl2onnx and onnxmltools 1.16.0, onnxruntime 1.30.0, leafparity 0.1.0). Each model type has 21 pairs (7 datasets, 3 sizes):

| Model type | Finite inputs | Including NaN |
|---|---|---|
| XGBoost via onnxmltools | 21 equivalent | 21 equivalent |
| scikit-learn GradientBoosting | 21 equivalent | 21 equivalent |
| scikit-learn RandomForest | 21 equivalent | 21 not equivalent |
| scikit-learn ExtraTrees | 21 equivalent | 21 not equivalent |
| scikit-learn DecisionTree | 21 equivalent | 21 not equivalent |
| LightGBM via onnxmltools | 6 equivalent, 15 not | 6 equivalent, 15 not |

These are models trained on built-in datasets following the documentation, not third-party production models. The details of the two findings (LightGBM float32 thresholds, scikit-learn NaN routing) are in the leafparity reports written to `reports/`.
