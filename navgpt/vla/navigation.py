"""Waypoint-to-TELEPORT conversion, derived from VLNCE-EVAL by EPIC Lab (MIT;
see navgpt/vla/LICENSE)."""
import numpy as np
import quaternion


def add_is_stuck(action_dict, is_stuck):
    if "action_args" in action_dict:
        action_dict["action_args"]["is_stuck"] = is_stuck
    return action_dict


def update_agent_state(current_global_state, relative_change):
    dx, dy = relative_change["x"], -relative_change["y"]
    dtheta = -relative_change["theta"]
    local_quat = np.quaternion(np.cos(-dtheta / 2), 0, np.sin(-dtheta / 2), 0)
    local_position = np.asarray([dy, 0, -dx])
    new_global_orientation = current_global_state['orientation'] * local_quat
    T_c_rotated = quaternion.as_rotation_matrix(current_global_state['orientation']).dot(local_position)
    new_global_position = T_c_rotated + current_global_state["position"]
    return {
        "action": "TELEPORT",
        "action_args": {
            "position": new_global_position.tolist(),
            "rotation": [new_global_orientation.x, new_global_orientation.y,
                         new_global_orientation.z, new_global_orientation.w],
            "theta": dtheta, "dx": dx, "dy": dy
        }
    }
