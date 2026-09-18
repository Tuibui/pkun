# Feedback from pkun_servo_driver. Hobby servos have no position feedback, so
# `position` is what the driver last actually wrote out after clamping and rate
# limiting -- not a measurement. Treat it as "commanded", not "achieved".

std_msgs/Header header

string[] name
float64[] position        # radians, last value written to the servo
float64[] pulse_us        # pulse width actually sent, useful for calibration

bool torque_enabled       # false = all channels released (no pulse)
uint8 boards_online       # number of PCA9685 boards responding
