# The 10 expression channels, matching EXPR_COLS in gestures.py exactly.
# Six of these are servos and get forwarded to the servo driver as joint
# commands; eye/led/led_v/sound are rendered by pkun_expression on the
# LCD and speaker.

uint8 LED_OFF=0
uint8 LED_REST=1
uint8 LED_LOW=2
uint8 LED_SOON=3
uint8 LED_URGENT=4
uint8 LED_GOOD=5
uint8 LED_NEUTRAL=6
uint8 LED_WARN=7
uint8 LED_LISTEN=8
uint8 LED_THINK=9
uint8 LED_WARM=10
uint8 LED_POWER=11
uint8 LED_FOCUS=12
uint8 LED_SPEAK=13

std_msgs/Header header

float32 head_pitch    # degrees, + = look up
float32 head_yaw      # degrees, + = turn left
float32 head_roll     # degrees, + = tilt left ("confused")
float32 ear_l         # -1 = flat down, 0 = neutral, +1 = perked up
float32 ear_r
float32 eye           # 0 = fully closed, 1 = fully open
float32 tail          # degrees, sweep left-right
uint8 led             # LED_* colour index above (steps, never interpolated)
float32 led_v         # brightness 0..1
float32 sound         # loudness 0..1, 0 = silent
