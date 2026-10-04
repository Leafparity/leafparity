# leafparity

**Exact equivalence checking for tree-ensemble models converted to ONNX.**

You trained an XGBoost, LightGBM or scikit-learn model. Production runs an ONNX
version of it. Are they the same model? Comparing predictions on a test set
cannot tell you: when a conversion goes wrong, it goes wrong in slivers of the
input space one float step wide, around a threshold, at exactly zero, or only for
missing values. Random and held-out samples practically never land there, and
production traffic eventually does.

leafparity does not sample. It reads both models, reproduces each runtime's exact
arithmetic (every cast, preprocessing step, comparison operator and missing-value
rule), and walks both models together over the whole input space at floating-point
precision. The result is one of two things:

* **EQUIVALENT**: a proof that for every possible input the two models reach
  corresponding leaves. Their outputs can differ only by the stated floating-point
  rounding allowance.
* **NOT EQUIVALENT**: every place in every tree where the models disagree, the exact
  set of inputs affected, and a concrete **witness input** for each problem. Both
  real runtimes are run on the witness so you can check the difference yourself.
  You also get the largest difference found anywhere and a proven upper bound.

```
$ leafparity check model.txt model.onnx

VERDICT: NOT EQUIVALENT
Not equivalent: 4 distinct problem(s) at 699 place(s) in the trees. For some
    inputs the raw outputs differ by 295.027 (proven to be the largest possible
    difference).

#1  zero / near-zero values are handled differently  [feature 'Column_0']
    occurs at 30 split node(s) in 30 tree(s); largest effect of a single tree: 22.6357
    example - tree 2:
      original  node 0: x <= -17.066402760988257 (double), missing_type=Zero, default -> left
      converted node 0: x <= -17.066402435302734 (float32), missing -> true branch
      inputs routed differently here: [-1.0000000180025095e-35, 1.0000000180025095e-35]
    witness: {Column_0=0.0, Column_1=114.41658720372287, ...}
      original predicts -198.276; converted predicts 54.7198  (verified by running both real runtimes)
```

*(Real output of `examples/02_lightgbm_zero_as_missing.py`, reproduced exactly — run it yourself to check.)*

## What it catches (real examples, reproduced in `examples/`)

| Case | What leafparity reports |
|---|---|
| The sklearn-onnx documentation's own "Issues when switching to float" example (StandardScaler + DecisionTreeRegressor) | The tutorial's test set shows a largest error of about 156. leafparity proves the largest possible error is **556** (721 with missing values), shows where each discrepancy is, and certifies the tutorial's `CastTransformer` fix as **EQUIVALENT** for float32 inputs without NaN. It also shows the fix does *not* hold for float64 inputs. |
| LightGBM trained with `zero_as_missing=True`, converted with onnxmltools | The converter ignores LightGBM's `missing_type=Zero`, so an input of exactly `0.0` takes a different path in every tree. Raw scores differ by up to ~295. |
| scikit-learn trees (>= 1.3) that receive NaN, converted with skl2onnx | scikit-learn routes NaN with `missing_go_to_left`, but the converted model sends NaN the other way at the affected nodes. |
| LightGBM (double thresholds) served through float32 ONNX | Narrow bands of float64 inputs next to thresholds are routed differently at almost every node. leafparity lists them and bounds their effect. With `--input-dtype float32` it tells you whether your float32 data can hit them at all. |

## Install

```
pip install leafparity[all]
# or [xgboost], [lightgbm], [sklearn] instead of [all]
```

Requires Python 3.9+, numpy, onnx and onnxruntime. The original model's library must
be installed, because leafparity runs it to verify its findings.

## Usage

### Command line

```
leafparity check ORIGINAL CONVERTED.onnx [options]
```

`ORIGINAL` can be an XGBoost `.json`/`.ubj` model, a LightGBM `.txt` model, or a pickled
(`.pkl`/`.joblib`) scikit-learn estimator, Pipeline, `XGBClassifier`, `LGBMRegressor`, and so on.

| Option | Meaning |
|---|---|
| `--input-dtype float64\|float32` | Precision of the feature values your application feeds in (default float64). This matters: many discrepancies exist only for float64 inputs. |
| `--no-missing` | Your inputs never contain NaN. |
| `--include-inf` | Also analyse ±infinity. |
| `--bounds bounds.json` | Restrict the domain, for example `{"age": [0, 120], "3": [0, null]}`. |
| `--background sample.csv` | A sample of real inputs. It gives features their names and makes witness inputs realistic. |
| `--json report.json` | Machine-readable report with every finding. |
| `--summary` | Print, and write with `--json`, only the verdict-level summary. See [Sharing a result without revealing the model](#sharing-a-result-without-revealing-the-model). |
| `--fail-above X` | CI gate: fail only if the proven maximum raw difference exceeds `X`. |
| `--worst-case-seconds S` | Time budget for tightening the worst-case bound (default 30). |

Exit codes: `0` EQUIVALENT, `1` NOT EQUIVALENT, `2` cannot certify (unsupported
construct, or failed self-check). You can use the command directly as a CI step.

### Python

```python
from leafparity import analyze
from leafparity.report import to_text

result = analyze(lgbm_model, "model.onnx", background=X_sample)
print(result.verdict["status"])     # 'EQUIVALENT' / 'NOT EQUIVALENT'
print(to_text(result))
result.to_dict()                    # everything, JSON-serialisable
result.to_summary_dict()            # verdict-level facts only, see below
```

## Sharing a result without revealing the model

The full report prints split thresholds, feature names and witness inputs, so it
reveals parts of the model. With `--summary` the model owner runs the check and
sends only the verdict:

```
$ leafparity check model.txt model.onnx --summary --json verdict.json

leafparity 0.1.0 - summary (model details withheld)
VERDICT: NOT EQUIVALENT
  Scope                         : every float64 input vector, including missing values (NaN)
  Distinct problems             : 4
  Kinds of problems             : zero value routing (2), threshold rounding to lower precision (2)
  Predicted class can change    : not applicable (regression model)
  Largest raw output difference : 295.027 (proven to be the largest possible)
  Examined                      : 30 trees, 4 features, 2327 joint regions
  Run time                      : 4.0 s
```

*(The model from `examples/02_lightgbm_zero_as_missing.py`, saved to files.)*

The summary never contains a threshold, a feature name or index, a witness input,
a leaf value, a node id or rule text. This applies to the printed output and to
`verdict.json`, which carries the same facts in a reduced schema of its own. When
the pair cannot be certified, the verdict is `CANNOT CERTIFY` with a general reason;
run again without `--summary` to see the details. Exit codes are the same as
without `--summary`.

## Why you can trust the verdict

leafparity checks its own work at three levels. The report shows the result of each.

1. **Self-check, before any analysis.** Inputs are constructed to *reach every split
   node* and sit exactly on, and one float step either side of, its decision
   boundary. Zeros, tiny values and NaN are included. They run through the real
   libraries (XGBoost / LightGBM / scikit-learn / onnxruntime) and through
   leafparity's exact model. Every tree decision must agree. If a single one differs,
   for example because a new library version changed its arithmetic, leafparity stops
   and refuses to certify.
2. **Witness verification.** Every reported problem and the worst case come with an
   input that has been run through both real runtimes. The report states whether the
   observed difference matches the predicted one.
3. **Independent cross-check.** Thousands of boundary inputs run through both real
   runtimes without using the analysis engine at all. Every observed difference must
   fall within the proven bound.

The test suite (`pytest tests/`) covers 28 model configurations, Pipelines included,
in both input precisions.
It proves leaf-for-leaf agreement at every node boundary. It also injects known
defects into ONNX models (a threshold moved by one float step, a flipped
missing-value flag, a changed leaf weight, a changed comparison operator) and
checks that each defect is found at exactly the right node, with exactly the right
set of affected inputs, and nothing else.

## Supported

| Original | Details |
|---|---|
| XGBoost | `gbtree` booster, numerical splits; regression, binary, multiclass |
| LightGBM | gbdt / rf, numerical splits, all missing-value modes (None / Zero / NaN); regression, binary, multiclass |
| scikit-learn | DecisionTree, ExtraTree, RandomForest, ExtraTrees, GradientBoosting (regressor / classifier) |
| scikit-learn Pipeline | any of the above, `XGBRegressor` / `XGBClassifier` or `LGBMRegressor` / `LGBMClassifier` as the last step, after scalers, `'passthrough'` or a ColumnTransformer of them (see [Pipelines](#pipelines)) |

| Converted | Details |
|---|---|
| ONNX | `ai.onnx.ml` TreeEnsembleRegressor / TreeEnsembleClassifier (opset <= 3), optionally after Cast / Scaler / Add / Sub / Mul / Div by constants, and ArrayFeatureExtractor / Gather / Concat of columns; float or double input; as produced by skl2onnx and onnxmltools |

Not yet supported, and refused rather than guessed: categorical splits, XGBoost
`dart`, LightGBM linear trees, ONNX `ai.onnx.ml` opset-5 `TreeEnsemble`, PMML,
compiled code (m2cgen, treelite) and SQL.

## Pipelines

`ORIGINAL` can be a fitted scikit-learn `Pipeline` saved with joblib, compared with its
skl2onnx conversion. For an XGBoost or LightGBM last step, register the onnxmltools
converter with skl2onnx (`update_registered_converter`) before converting.

```
leafparity check pipeline.joblib pipeline.onnx
```

Supported steps before the last one:

* `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `CastTransformer`
* `'passthrough'`
* a `ColumnTransformer` of those scalers and `'passthrough'`, each applied to whole
  columns selected by position; other columns may be dropped

The last step is a scikit-learn tree model, XGBoost or LightGBM (scikit-learn API) on
numeric features.

No threshold is moved by algebra such as `thr * scale + mean`: rounding makes that
differ from the real runtimes for some inputs. leafparity evaluates each scaler with
the operations, order and dtype of the runtime it models: scikit-learn's own arithmetic
for the original, and for the ONNX file whatever the graph contains (an `ai.onnx.ml`
Scaler computing `(x - offset) * scale` in float32, or Cast, Mul and Add nodes). The raw
input at which a split changes side is then found by binary search over every
representable float. NaN passes through every scaler unchanged and keeps each model's
own routing. Before the analysis, both descriptions of the preprocessing are checked
bit for bit against the Pipeline's real transform steps and against onnxruntime
running the graph's own preprocessing nodes; if either does not match, the pair is
refused.

Refused with exit code 2, naming the step:

* any other transformer: imputers, encoders such as `OneHotEncoder`, feature
  selectors, a `FunctionTransformer` with a function, custom transformers (also when
  their class cannot be imported to load the file), nested Pipelines
* a `ColumnTransformer` containing any of those, selecting columns by name or with a
  callable (leafparity runs the Pipeline on plain numeric arrays), or using
  `transformer_weights`
* categorical splits in the last step, for example LightGBM `categorical_feature`
* in the ONNX file, any node in front of the trees other than Cast, Identity, Scaler,
  Add / Sub / Mul / Div by a constant, ArrayFeatureExtractor / Gather of constant
  columns and Concat

In the report, features and witness inputs refer to the Pipeline's input columns. The
threshold in a rule (`x <= ...`) is on the scaled value the tree compares, and "inputs
routed differently" lists raw input values.

## What "raw output" means

Differences are reported in each model's raw output space before the final link
function. For boosted models that is the margin / log-odds. For forests it is the
averaged probability or prediction. Witnesses also show the final predictions,
probabilities and classes from both runtimes.

## Limits you should know

* The worst-case search is exact for most models. For very large models with
  thousands of discrepancy slivers it runs under a time budget. You then get the
  largest difference *found* and a *sound upper bound*. The true maximum lies
  between them, and the report says so.
* The analysed domain is every value of the chosen input precision with magnitude up
  to float32 max. Infinity is included only on request.
