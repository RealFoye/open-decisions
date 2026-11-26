"""open-decisions: a self-hosted Decisions API.

Score up to 1,000 yes/no, multiple-choice and rating questions against one shared
input in a single request, and get calibrated probabilities back instead of text.
"""

__version__ = "0.1.0"

from .calibration import Calibrator  # noqa: E402
from .schema import DecisionRequest, DecisionResponse  # noqa: E402
from .service import DecisionService  # noqa: E402
