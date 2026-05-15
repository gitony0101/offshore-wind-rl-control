"""Baseline controllers for the OffshoreWind floating platform environment.

Provides reference controllers to benchmark RL policies against:
- :class:`NoControlController`: zero-action baseline (uncontrolled dynamics).
- :class:`PDController`: classical proportional-derivative feedback controller.
"""

from src.baselines.no_control import NoControlController
from src.baselines.pd_controller import PDController

__all__ = ["NoControlController", "PDController"]
