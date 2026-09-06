"""PromptLab: reproducible prompt optimisation as experimental science."""

from .experiment import ExperimentRunner
from .genome import PromptGenome, variant_library
from .models import Task, Trial

__all__ = ["ExperimentRunner", "PromptGenome", "Task", "Trial", "variant_library"]
__version__ = "0.1.0"
