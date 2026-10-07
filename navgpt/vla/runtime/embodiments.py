"""Inference-time embodiment scaling.

Action ranges used to de-normalise predicted waypoints for the simulated Habitat
embodiment evaluated in this release.
"""

import math


ROBOT_SCALE_CONFIGS = {
    "habitat_nav": {"x_range": 1.0, "y_range": 0.433, "theta_range": math.pi / 6 * 4},
}
