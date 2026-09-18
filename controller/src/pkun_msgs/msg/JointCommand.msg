# Name-addressed joint command. Any publisher may command a SUBSET of joints,
# which is what lets pkun_motion (12 leg DOF) and pkun_expression (head/ear/tail)
# share this one topic without fighting over the same array indices.
#
# Leg joint names:  <leg>_<joint>  with leg in {fl, fr, rl, rr}
#                   and joint in {abduction, hip_pitch, knee}
# Expression names: head_pitch, head_yaw, head_roll, ear_l, ear_r, tail

std_msgs/Header header

string[] name
float64[] position      # radians, in the same frame convention as leg_kinematics.py

# Time the driver should take to reach the target, seconds.
# 0.0 = go as fast as the servo allows (the driver still rate-limits).
float64 duration
