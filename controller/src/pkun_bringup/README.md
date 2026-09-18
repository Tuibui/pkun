# P-kun controller workspace

Six ROS 2 Jazzy packages driving the desk elephant: 12 leg DOF plus a 6-servo
expression layer, on two PCA9685 boards over I²C.

## Package layout

| Package | Owns | Touches hardware |
|---|---|---|
| `pkun_msgs` | JointCommand, ServoState, Gesture, Expression, BodyPose, SetTrim, SetTorque | no |
| `pkun_servo_driver` | all 18 servos, PCA9685 I²C, calibration | **yes** — the PCA9685 boards |
| `pkun_motion` | leg IK/FK, trot gait, gesture body track | no |
| `pkun_teleop` | Bluetooth gamepad → intent | no |
| `pkun_expression` | face: OLED, sound, head/ear/tail track | **yes** — the OLED panel |
| `pkun_bringup` | launch files, startup ordering | no |

Data flows one direction, and nothing depends on another node's package —
only on `pkun_msgs`:

```
pkun_teleop ──/pkun/gesture────▶ pkun_motion ────┐
     │                                            ├──/pkun/joint_command──▶ pkun_servo_driver ──I²C──▶ 2× PCA9685
     ├──/pkun/cmd_vel──────────▶ pkun_motion ────┤
     └──/pkun/body_pose────────▶ pkun_motion     │
                                                  │
     /pkun/gesture ───────────▶ pkun_expression ─┘  (head, ears, tail)
                                       └──────────▶ OLED + speaker
```

`joint_command` is addressed **by joint name**, and a message may carry any
subset. That is what lets motion (12 leg joints) and expression (6 face joints)
publish to the same topic without colliding.

### Why there is no "connect to devices" node

Opening the bus belongs to whoever owns the device. `pkun_servo_driver` is a
lifecycle node: `on_configure()` opens `/dev/i2c-1` and probes both boards,
`on_activate()` starts writing pulses, `on_deactivate()` releases every channel.
A wiring fault fails one clean transition and leaves the node in `unconfigured`
with the reason logged — no crash, no retry loop, no second node holding a file
descriptor that the driver needs.

## Build

```bash
cd ~/p_kun/controller
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
colcon test && colcon test-result --all      # 19 tests
```

## Run

```bash
ros2 launch pkun_bringup pkun.launch.py
ros2 launch pkun_bringup pkun.launch.py display_backend:=ssd1306
ros2 launch pkun_bringup pkun.launch.py use_joy:=false
```

The launch configures the servo driver first and only activates it once
configure reports success.

## I²C permissions

The driver needs read/write on the bus. On a fresh machine `/dev/i2c-1` is
`root`-only, and configure fails with `Permission denied`:

```bash
sudo apt install i2c-tools          # creates the i2c group and its udev rule
sudo usermod -aG i2c $USER
# log out and back in, then confirm both boards ack:
i2cdetect -y 1                      # expect 0x40 and 0x41 (0x3C for the OLED)
```

On a Raspberry Pi also enable the bus in `raspi-config`; on a Jetson check which
bus number the header maps to and set `i2c_bus` accordingly.

## Servo calibration

```bash
ros2 launch pkun_bringup calibrate.launch.py
```

Nothing publishes joint commands in that launch, so the robot holds still.

```bash
# release everything and fit the horns by hand
ros2 service call /pkun/servo_driver/set_torque pkun_msgs/srv/SetTorque \
    "{enable: false, name: []}"

# re-enable, then drive one joint to its zero
ros2 service call /pkun/servo_driver/set_torque pkun_msgs/srv/SetTorque \
    "{enable: true, name: []}"
ros2 topic pub --once /pkun/joint_command pkun_msgs/msg/JointCommand \
    "{name: ['fl_knee'], position: [0.0], duration: 1.0}"

# nudge the zero until the link is genuinely straight
ros2 service call /pkun/servo_driver/set_trim pkun_msgs/srv/SetTrim \
    "{name: 'fl_knee', trim: 0.035, persist: true}"
```

Copy the settled `trim_deg` values into
`pkun_servo_driver/config/servos.yaml`. If a joint moves the wrong direction,
flip `invert` — do not use a negative `us_per_deg`, because limits are applied
before the sign.

## Relationship to the Python in `~/p_kun`

The Python (`leg_kinematics.py`, `walking_gait.py`, `gestures.py`, …) is
untouched and remains the reference implementation. The C++ DH table, leg signs,
analytic IK, joint limits and posture convention are ports of it, and
`pkun_motion`'s tests re-run the same checks as its `verify()`. **If the two ever
disagree, the Python is right and the C++ is the bug.**

Two things are deliberately *not* ported:

- `gestures.py` authors full 50 Hz clips with breathing, noise and CoG-aware
  weight shifts. The C++ clip tables are posture-keyframe sketches of the same
  20 gestures. To play the real clips, export them to CSV and load those instead
  — the sequencer already interpolates keyframes, so it is a loader change only.
- `find_sleep_pose()` searches for the true minimum body height. `sleep_mode`
  here uses a conservative fixed crouch.

## Safety notes

- `max_rate_dps` in `servos.yaml` is the 300 °/s planning rate from
  `walking_gait.py`, not the no-load spec. Analog servos overheat if actually
  driven at 500 °/s. Servo torque is the binding constraint on this robot.
- Motion halts if `cmd_vel` goes stale for 0.5 s; teleop sends an explicit zero
  when the deadman is released and again if `/joy` goes silent for 1 s (a
  Bluetooth pad that leaves range produces no disconnect event, only silence).
- Unreachable IK never reaches the servos. `leg_ik` returns false, and motion
  holds the last good pose — publishing NaN would arrive as a full-speed slam
  into the end stops.
