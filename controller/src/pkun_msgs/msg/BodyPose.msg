# 6-DOF body posture target, matching POSE_COLS in gestures.py.
# pkun_motion turns this into foot targets and then into leg joint angles,
# keeping all four feet planted.

std_msgs/Header header

float64 height        # mm, body origin {B} above the ground
float64 x             # mm, body shift forward
float64 y             # mm, body shift left
float64 roll          # radians, + = lean left
float64 pitch         # radians, + = nose up
float64 yaw           # radians, + = turn left
