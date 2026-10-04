"""leafparity - exact equivalence checking for converted tree-ensemble models.

    from leafparity import analyze
    result = analyze(xgb_model, "model.onnx")
    print(result.verdict["status"])
"""
__version__ = "0.1.2"

from .analyze import Analysis, SelfCheckError, analyze  # noqa: E402
from .ir import UnsupportedModelError  # noqa: E402

__all__ = ["analyze", "Analysis", "SelfCheckError", "UnsupportedModelError", "__version__"]
