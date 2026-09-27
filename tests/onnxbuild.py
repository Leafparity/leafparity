"""Tiny helper: build an ai.onnx.ml TreeEnsembleRegressor from python tree specs.
A tree is a nested structure: ("leaf", w) or (feat, mode, thr, tracks, true_sub, false_sub)."""
import numpy as np
from onnx import helper, TensorProto


def build(trees, n_features, base=0.0):
    A = {k: [] for k in ["nodes_treeids", "nodes_nodeids", "nodes_featureids", "nodes_modes",
                         "nodes_values", "nodes_truenodeids", "nodes_falsenodeids",
                         "nodes_missing_value_tracks_true", "target_treeids", "target_nodeids",
                         "target_ids", "target_weights"]}
    for t, tree in enumerate(trees):
        counter = [0]

        def emit(sub):
            nid = counter[0]; counter[0] += 1
            idx = len(A["nodes_nodeids"])
            A["nodes_treeids"].append(t); A["nodes_nodeids"].append(nid)
            for k in ["nodes_featureids", "nodes_modes", "nodes_values", "nodes_truenodeids",
                      "nodes_falsenodeids", "nodes_missing_value_tracks_true"]:
                A[k].append(None)
            if sub[0] == "leaf":
                A["nodes_featureids"][idx] = 0; A["nodes_modes"][idx] = "LEAF"
                A["nodes_values"][idx] = 0.0; A["nodes_truenodeids"][idx] = 0
                A["nodes_falsenodeids"][idx] = 0; A["nodes_missing_value_tracks_true"][idx] = 0
                A["target_treeids"].append(t); A["target_nodeids"].append(nid)
                A["target_ids"].append(0); A["target_weights"].append(float(sub[1]))
                return nid
            f, mode, thr, tracks, ts, fs = sub
            A["nodes_featureids"][idx] = f; A["nodes_modes"][idx] = mode
            A["nodes_values"][idx] = float(np.float32(thr))
            A["nodes_missing_value_tracks_true"][idx] = int(tracks)
            A["nodes_truenodeids"][idx] = emit(ts)
            A["nodes_falsenodeids"][idx] = emit(fs)
            return nid
        emit(tree)
    node = helper.make_node("TreeEnsembleRegressor", ["X"], ["Y"], domain="ai.onnx.ml",
                            n_targets=1, aggregate_function="SUM", post_transform="NONE",
                            base_values=[float(base)], **A)
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])],
                          [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 15), helper.make_opsetid("ai.onnx.ml", 3)])
    m.ir_version = 8
    return m


def run(proto, X):
    import onnxruntime as ort
    s = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return s.run(None, {"X": np.asarray(X, np.float32)})[0].reshape(-1).astype(np.float64)
