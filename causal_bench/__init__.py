"""Evaluation library and model wrappers for the CFM benchmark."""

# Must run before anything can import causalpfn (which imports faiss before
# torch and segfaults on Apple Silicon). No-op on every other platform.
from .macos_compat import ensure_causalpfn_importable, needs_faiss_shim, shim_active

ensure_causalpfn_importable()

from .data_generators import (
    SyntheticDataset,
    get_dataset,
    list_datasets,
    GENERATORS,
)
from .data_loader import (
    load_lalonde,
    LalondeDataset,
    list_available_datasets,
    load_lalonde_realcause,
    RealCauseLalondeRealization,
)
from .metrics import evaluate_cate, pehe, ate_abs_error, ate_rel_error, bias, coverage_95
from .wrap_causalpfn import CausalPFNWrapper
from .wrap_dopfn import DoPFNWrapper
from .wrap_causalfm import CausalFMWrapper
from .wrap_metalearners import (
    HPOConfig,
    SLearnerWrapper,
    TLearnerWrapper,
    XLearnerWrapper,
    DebiasedMLWrapper,
    IPWWrapper,
    DRWrapper,
    METALEARNER_WRAPPERS,
)

__all__ = [
    "ensure_causalpfn_importable",
    "needs_faiss_shim",
    "shim_active",
    "SyntheticDataset",
    "get_dataset",
    "list_datasets",
    "GENERATORS",
    "load_lalonde",
    "LalondeDataset",
    "list_available_datasets",
    "load_lalonde_realcause",
    "RealCauseLalondeRealization",
    "evaluate_cate",
    "pehe",
    "ate_abs_error",
    "ate_rel_error",
    "bias",
    "coverage_95",
    "CausalPFNWrapper",
    "DoPFNWrapper",
    "CausalFMWrapper",
    "HPOConfig",
    "SLearnerWrapper",
    "TLearnerWrapper",
    "XLearnerWrapper",
    "DebiasedMLWrapper",
    "IPWWrapper",
    "DRWrapper",
    "METALEARNER_WRAPPERS",
]
