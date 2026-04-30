from importlib.metadata import PackageNotFoundError, version

from cube_harness.experiment import EXP_DIR, make_experiment_output_dir

try:
    __version__ = version("cube-harness")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["EXP_DIR", "__version__", "make_experiment_output_dir"]
