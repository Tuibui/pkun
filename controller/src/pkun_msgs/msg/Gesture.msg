# Request a named gesture clip from pkun_motion.
# Ids mirror the clip set already defined in gestures.py.

# lifecycle
string SLEEP_MODE=sleep_mode
string WAKE_UP=wake_up
string IDLE_BREATHE=idle_breathe
string DOZE_OFF=doze_off

# attention
string NOTICE_LOW=notice_low
string NOTICE_SOON=notice_soon
string NOTICE_URGENT=notice_urgent
string SETTLE=settle
string FOCUS_WITH_YOU=focus_with_you

# conversation
string LISTENING=listening
string THINKING=thinking
string SPEAKING=speaking
string CONFIRM_DONE=confirm_done
string DIDNT_GET_IT=didnt_get_it

# social
string GREET=greet
string CELEBRATE=celebrate
string COMFORT=comfort

# system
string ERROR_STATE=error_state
string LOW_POWER=low_power
string HOME_RESET=home_reset

std_msgs/Header header

string id
float32 speed 1.0    # playback rate multiplier, 1.0 = as authored
bool loop false      # repeat from the clip's loop point until preempted
