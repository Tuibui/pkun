#!/usr/bin/env python3
"""
pca_ch1.py — คุมเซอร์โวช่อง 1 บน PCA9685 ตัวเดียวจบ สำหรับก๊อปไปวางบน Raspberry Pi

ไฟล์นี้ "ยืนได้ด้วยตัวเอง" ใช้แค่ไลบรารีมาตรฐาน (fcntl, os, time)
ไม่ต้องมี numpy, ไม่ต้อง smbus, ไม่ต้อง import อะไรจากโปรเจกต์ P-kun
ก๊อปไฟล์เดียวไปที่ Pi แล้วรันได้เลย:

    scp pca_ch1.py pi@raspberrypi.local:~
    ssh pi@raspberrypi.local
    python3 pca_ch1.py --center

ช่อง 1 บนบอร์ด 0x40 = fl_hip_pitch (θ2 ขาหน้าซ้าย) ตามตารางใน servos.yaml
  โมเดล 0°   = เซอร์โว 90°  = 1500 us   (ท่า zero pose ขาเหยียดตรงดิ่งลง)
  โมเดล +90° = เซอร์โว 180° = 2500 us   (เท้าถอยไปทางท้าย)
  โมเดล -90° = เซอร์โว 0°   = 500 us    (เท้าไปข้างหน้า)

>>> ก่อนรัน: ยกหุ่นขึ้นแท่นให้ขาลอยพ้นพื้น และจ่ายไฟเซอร์โวแยกจากไฟ Pi <<<

คำสั่งที่ใช้บ่อย:
    python3 pca_ch1.py --center             ไปกลางราง (90°)
    python3 pca_ch1.py --angle 120          ไปมุมเซอร์โว 120°
    python3 pca_ch1.py --model -30          สั่งด้วยมุมโมเดล (แปลงให้)
    python3 pca_ch1.py --us 1500            สั่งความกว้างพัลส์ตรง ๆ
    python3 pca_ch1.py --sweep              กวาดไป-กลับทั้งช่วง
    python3 pca_ch1.py --interactive        พิมพ์มุมแล้วเอ็นเทอร์ ขยับทันที (จูนง่ายสุด)
    python3 pca_ch1.py --release            หยุดพัลส์ หมุนฮอร์นด้วยมือได้
    python3 pca_ch1.py --angle 120 --dry-run  ดูว่าจะส่งอะไร โดยไม่แตะฮาร์ดแวร์

ถ้าเปิดบัสไม่ได้:
    sudo raspi-config          -> Interface Options -> I2C -> Enable
    sudo apt install i2c-tools
    i2cdetect -y 1             -> ต้องเห็น 40 (และ 41 ถ้าต่อบอร์ดสอง)
    sudo usermod -aG i2c $USER -> แล้ว log out / log in ใหม่
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import os
import time

# ---------------------------------------------------------------------------
# 1) ค่าคงที่ของ PCA9685 (ชุดเดียวกับ pkun_servo_driver/src/pca9685.cpp)
# ---------------------------------------------------------------------------

I2C_SLAVE = 0x0703          # ioctl เลือกที่อยู่อุปกรณ์บนบัส

REG_MODE1 = 0x00
REG_LED0_ON_L = 0x06
REG_PRESCALE = 0xFE

MODE1_RESTART = 0x80
MODE1_AUTOINC = 0x20
MODE1_SLEEP = 0x10

OSC_HZ = 25_000_000.0       # ออสซิลเลเตอร์ในชิป ใช้คำนวณ prescale
TICKS = 4096                # ความละเอียด PWM 12 บิต

# ---------------------------------------------------------------------------
# 2) สเปกเซอร์โว + การแปลงมุม (ตรงกับ servo_map.py / servos.yaml)
# ---------------------------------------------------------------------------

PULSE_MIN_US = 500.0        # เซอร์โวที่ 0°
PULSE_MAX_US = 2500.0       # เซอร์โวที่ 180°
SERVO_SPAN_DEG = 180.0
US_PER_DEG = (PULSE_MAX_US - PULSE_MIN_US) / SERVO_SPAN_DEG   # 11.111 us/deg
PWM_HZ = 50.0               # analog servo รับได้แค่ 50 Hz

# ช่อง 1 = fl_hip_pitch: home 90 (โมเดล 0), dir +1, ลิมิตโมเดล -90..+90
JOINT_NAME = "fl_hip_pitch"
HOME_DEG = 90.0
DIRECTION = +1
MODEL_MIN = -90.0
MODEL_MAX = +90.0
TRIM_DEG = 0.0              # ใส่ค่าชดเชยร่องฮอร์นตรงนี้หลังจูนเสร็จ

# กันชนปลายราง: ไม่สั่งเข้าใกล้ 0/180 กว่านี้ เว้นแต่ใส่ --raw
SERVO_MIN_DEG = 5.0
SERVO_MAX_DEG = 175.0

MAX_RATE_DPS = 300.0        # ความเร็วสูงสุดที่วางแผนไว้ใน walking_gait.py


def to_servo(model_deg: float) -> float:
    """มุมโมเดล -> มุมบนหน้าปัดเซอร์โว 0..180."""
    return HOME_DEG + DIRECTION * (model_deg + TRIM_DEG)


def to_model(servo_deg: float) -> float:
    """มุมบนหน้าปัดเซอร์โว 0..180 -> มุมโมเดล."""
    return DIRECTION * (servo_deg - HOME_DEG) - TRIM_DEG


def to_pulse_us(servo_deg: float) -> float:
    """มุมเซอร์โว -> ความกว้างพัลส์ (ไมโครวินาที)."""
    return PULSE_MIN_US + servo_deg * US_PER_DEG


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


# ---------------------------------------------------------------------------
# 3) ตัวขับ PCA9685 — เขียนผ่าน i2c-dev ตรง ๆ ไม่ต้องพึ่งไลบรารีนอก
# ---------------------------------------------------------------------------


class Pca9685:
    """ตัวขับ PCA9685 ขั้นต่ำที่พอสำหรับสั่งเซอร์โว.

    ใช้ read()/write() ธรรมดาหลังเรียก ioctl(I2C_SLAVE) แทน SMBus helper
    จะได้ไม่ต้องลง python3-smbus เพิ่มบน Pi
    """

    def __init__(self, bus: str = "/dev/i2c-1", address: int = 0x40) -> None:
        self.bus = bus
        self.address = address
        self.fd = -1
        self.freq_hz = PWM_HZ

    # -- เปิด / ปิด -------------------------------------------------------

    def open(self) -> None:
        try:
            self.fd = os.open(self.bus, os.O_RDWR)
            fcntl.ioctl(self.fd, I2C_SLAVE, self.address)
        except (OSError, PermissionError) as exc:
            raise SystemExit(
                f"เปิด {self.bus} ที่ {self.address:#04x} ไม่ได้: {exc}\n"
                f"  1) เปิด I2C หรือยัง:  sudo raspi-config -> Interface Options -> I2C\n"
                f"  2) เห็นบอร์ดไหม:      i2cdetect -y {self.bus[-1]}   (ควรเห็น 40)\n"
                f"  3) สิทธิ์พอไหม:        sudo usermod -aG i2c $USER แล้ว log out/in"
            ) from exc
        self._write_reg(REG_MODE1, MODE1_AUTOINC)
        time.sleep(0.005)
        self.all_off()

    def close(self) -> None:
        if self.fd >= 0:
            try:
                self.all_off()
            except OSError:
                pass
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "Pca9685":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- รีจิสเตอร์ -------------------------------------------------------

    def _write_reg(self, reg: int, value: int) -> None:
        os.write(self.fd, bytes((reg & 0xFF, value & 0xFF)))

    def _read_reg(self, reg: int) -> int:
        os.write(self.fd, bytes((reg & 0xFF,)))
        return os.read(self.fd, 1)[0]

    # -- ความถี่ / พัลส์ ---------------------------------------------------

    def set_frequency(self, hz: float = PWM_HZ) -> None:
        """ตั้งความถี่ร่วมของทั้ง 16 ช่อง — ต้องพักออสซิลเลเตอร์ก่อนถึงเขียน prescale ได้."""
        prescale = int(round(OSC_HZ / (TICKS * hz))) - 1
        prescale = max(3, min(255, prescale))

        old = self._read_reg(REG_MODE1)
        self._write_reg(REG_MODE1, (old & ~MODE1_RESTART) | MODE1_SLEEP)
        self._write_reg(REG_PRESCALE, prescale)
        self._write_reg(REG_MODE1, old)
        time.sleep(0.001)
        self._write_reg(REG_MODE1, old | MODE1_RESTART | MODE1_AUTOINC)

        # prescale เป็นจำนวนเต็ม 50 Hz จริง ๆ จึงได้ ~49.72 Hz
        # ต้องคิดพัลส์จากค่าจริง ไม่งั้นความกว้างพัลส์เพี้ยนไปทั้งแผงประมาณ 0.6%
        self.freq_hz = OSC_HZ / (TICKS * (prescale + 1))

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        """ตั้งค่า tick ดิบของช่องหนึ่ง (12 บิต) — เงียบถ้าบัสปิดไปแล้ว."""
        if self.fd < 0:
            return
        reg = REG_LED0_ON_L + 4 * channel
        os.write(self.fd, bytes((reg, on & 0xFF, (on >> 8) & 0x0F,
                                 off & 0xFF, (off >> 8) & 0x0F)))

    def set_pulse_us(self, channel: int, microseconds: float) -> None:
        """สั่งพัลส์กว้างกี่ไมโครวินาที — ใส่ 0 = ปล่อยช่อง (ไม่มีพัลส์ออกเลย)."""
        if microseconds <= 0.0:
            self.set_pwm(channel, 0, 0x1000)      # บิต full-off
            return
        ticks = int(round(microseconds * 1e-6 * self.freq_hz * TICKS))
        self.set_pwm(channel, 0, max(1, min(TICKS - 1, ticks)))

    def release(self, channel: int) -> None:
        self.set_pwm(channel, 0, 0x1000)

    def all_off(self) -> None:
        for ch in range(16):
            self.set_pwm(ch, 0, 0x1000)


# ---------------------------------------------------------------------------
# 4) ชั้นบน: สั่งเป็น "องศา" แล้วเดินไปช้า ๆ ไม่กระชาก
# ---------------------------------------------------------------------------


class Servo:
    """เซอร์โวหนึ่งตัวบนช่องหนึ่ง จำมุมล่าสุดไว้เพื่อเดินไปทีละก้าว."""

    def __init__(self, board: Pca9685 | None, channel: int, raw: bool = False) -> None:
        self.board = board          # None = dry run
        self.channel = channel
        self.raw = raw
        self.current: float | None = None   # None = ยังไม่เคยสั่ง ไม่รู้ว่าฮอร์นอยู่ไหน

    def limits(self) -> tuple[float, float]:
        """ช่วงมุมเซอร์โวที่ยอมให้สั่ง."""
        if self.raw:
            return 0.0, SERVO_SPAN_DEG
        return SERVO_MIN_DEG, SERVO_MAX_DEG

    def write(self, servo_deg: float) -> float:
        """สั่งมุมเดียว ไม่สนใจความเร็ว (ใช้ตอนก้าวสั้น ๆ)."""
        lo, hi = self.limits()
        servo_deg = clamp(servo_deg, lo, hi)
        if self.board is not None:
            self.board.set_pulse_us(self.channel, to_pulse_us(servo_deg))
        self.current = servo_deg
        return servo_deg

    def move_to(self, servo_deg: float, rate_dps: float = 120.0,
                hz: float = 50.0) -> float:
        """เดินจากมุมปัจจุบันไปมุมเป้าหมายด้วยความเร็วจำกัด.

        ครั้งแรกที่สั่ง เราไม่รู้ว่าฮอร์นค้างอยู่มุมไหน จะกระโดดไปเป้าหมายเลย
        แล้วรอให้เซอร์โวตามให้ทัน หลังจากนั้นถึงเดินทีละก้าวได้
        """
        lo, hi = self.limits()
        target = clamp(servo_deg, lo, hi)

        if self.current is None:
            self.write(target)
            time.sleep(0.4)
            return target

        dt = 1.0 / hz
        step = max(0.1, rate_dps * dt)
        start = self.current
        span = target - start
        n = max(1, int(abs(span) / step + 0.5))
        for i in range(1, n + 1):
            self.write(start + span * i / n)
            time.sleep(dt)
        return target

    def release(self) -> None:
        if self.board is not None:
            self.board.release(self.channel)
        self.current = None


def describe(servo_deg: float) -> str:
    """หนึ่งบรรทัด: มุมเซอร์โว = มุมโมเดล = ความกว้างพัลส์."""
    return (f"servo {servo_deg:6.1f}°   model {to_model(servo_deg):+7.1f}°   "
            f"{to_pulse_us(servo_deg):7.1f} us")


# ---------------------------------------------------------------------------
# 5) โหมดการทำงาน
# ---------------------------------------------------------------------------


def do_sweep(servo: Servo, step: float, rate: float, hold: float) -> None:
    """กวาดจากปลายล่างไปปลายบนแล้วกลับ พิมพ์ทุก ๆ 15 องศา."""
    lo, hi = servo.limits()
    print(f"กวาด {lo:.0f}° -> {hi:.0f}° -> {lo:.0f}°  ที่ {rate:.0f}°/s")
    print()

    servo.move_to(lo, rate)
    print("  " + describe(lo))
    time.sleep(hold)

    points: list[float] = []
    v = lo
    while v < hi:
        v = min(hi, v + step)
        points.append(v)
    points += points[-2::-1] + [lo]

    for target in points:
        servo.move_to(target, rate)
        if abs(target - round(target / 15.0) * 15.0) < step / 2.0:
            print("  " + describe(target))
    print()
    print("กลับถึงจุดเริ่มแล้ว")


def do_interactive(servo: Servo, rate: float) -> None:
    """พิมพ์มุมแล้วเอ็นเทอร์ ขยับทันที — โหมดที่ใช้จูนฮอร์นสะดวกที่สุด."""
    lo, hi = servo.limits()
    print("โหมดพิมพ์สด — ใส่มุมแล้วกด Enter")
    print(f"  ตัวเลขเปล่า ๆ  = มุมเซอร์โว ({lo:.0f}..{hi:.0f})")
    print("  m<ตัวเลข>     = มุมโมเดล เช่น  m-30")
    print("  u<ตัวเลข>     = พัลส์ไมโครวินาที เช่น  u1500")
    print("  c             = กลับกลางราง 90°")
    print("  r             = ปล่อยพัลส์ (หมุนด้วยมือได้)")
    print("  q             = ออก")
    print()
    while True:
        try:
            raw = input("มุม> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            continue
        if raw in ("q", "quit", "exit"):
            return
        if raw == "r":
            servo.release()
            print("  ปล่อยพัลส์แล้ว")
            continue
        if raw == "c":
            servo.move_to(90.0, rate)
            print("  " + describe(90.0))
            continue

        try:
            if raw.startswith("m"):
                target = to_servo(float(raw[1:]))
            elif raw.startswith("u"):
                target = (float(raw[1:]) - PULSE_MIN_US) / US_PER_DEG
            else:
                target = float(raw)
        except ValueError:
            print("  อ่านไม่ออก — ใส่ตัวเลข, m<เลข>, u<เลข>, c, r หรือ q")
            continue

        lo, hi = servo.limits()
        if not lo - 1e-9 <= target <= hi + 1e-9:
            print(f"  {target:.1f}° อยู่นอกช่วง {lo:.0f}..{hi:.0f} "
                  f"-> จะบีบให้อยู่ในช่วง (ใส่ --raw ถ้าอยากไปสุดราง)")
        reached = servo.move_to(target, rate)
        print("  " + describe(reached))


# ---------------------------------------------------------------------------
# 6) CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="คุมเซอร์โวช่องเดียวบน PCA9685 (ตั้งต้นที่ช่อง 1 = fl_hip_pitch)")
    ap.add_argument("--channel", type=int, default=1, help="ช่องบนบอร์ด 0-15 (ตั้งต้น 1)")
    ap.add_argument("--addr", type=lambda v: int(v, 0), default=0x40,
                    help="ที่อยู่บอร์ด (ตั้งต้น 0x40)")
    ap.add_argument("--bus", default="/dev/i2c-1", help="บัส i2c (ตั้งต้น /dev/i2c-1)")

    act = ap.add_mutually_exclusive_group(required=True)
    act.add_argument("--angle", type=float, metavar="DEG", help="ไปที่มุมเซอร์โว 0-180")
    act.add_argument("--model", type=float, metavar="DEG", help="ไปที่มุมโมเดล (แปลงให้)")
    act.add_argument("--us", type=float, metavar="US", help="สั่งความกว้างพัลส์ตรง ๆ")
    act.add_argument("--center", action="store_true", help="ไปกลางราง 90°")
    act.add_argument("--sweep", action="store_true", help="กวาดไป-กลับทั้งช่วง")
    act.add_argument("--interactive", action="store_true", help="พิมพ์มุมสด ๆ ทีละค่า")
    act.add_argument("--release", action="store_true", help="หยุดพัลส์ หมุนด้วยมือได้")

    ap.add_argument("--rate", type=float, default=120.0,
                    help="ความเร็ว °/s (ตั้งต้น 120, เพดาน %.0f)" % MAX_RATE_DPS)
    ap.add_argument("--step", type=float, default=5.0, help="ก้าวของ --sweep (ตั้งต้น 5°)")
    ap.add_argument("--hold", type=float, default=0.5,
                    help="ค้างที่ปลายทางกี่วินาทีก่อนปล่อย (ตั้งต้น 0.5)")
    ap.add_argument("--keep", action="store_true",
                    help="ค้างพัลส์ไว้หลังจบ (ตั้งต้นคือปล่อย ไม่ให้เซอร์โวร้อน)")
    ap.add_argument("--raw", action="store_true",
                    help="ใช้ราง 0-180 เต็ม ข้ามกันชนปลาย (ระวังโครงชนกัน)")
    ap.add_argument("--dry-run", action="store_true",
                    help="พิมพ์อย่างเดียว ไม่แตะฮาร์ดแวร์")
    args = ap.parse_args(argv)

    if not 0 <= args.channel <= 15:
        ap.error("--channel ต้องอยู่ระหว่าง 0 ถึง 15")
    rate = clamp(args.rate, 1.0, MAX_RATE_DPS)

    print("=" * 68)
    print(f"PCA9685 {args.addr:#04x} ช่อง {args.channel} บน {args.bus}"
          + (f"   [{JOINT_NAME}]" if args.channel == 1 and args.addr == 0x40 else ""))
    if args.channel == 1 and args.addr == 0x40:
        print(f"  โมเดล 0° = เซอร์โว {HOME_DEG:.0f}° = {to_pulse_us(HOME_DEG):.0f} us"
              f"   (ลิมิตโมเดล {MODEL_MIN:+.0f}..{MODEL_MAX:+.0f})")
    lo, hi = (0.0, SERVO_SPAN_DEG) if args.raw else (SERVO_MIN_DEG, SERVO_MAX_DEG)
    print(f"  ช่วงที่ยอมให้สั่ง {lo:.0f}..{hi:.0f}°   ความเร็ว {rate:.0f}°/s")
    if args.dry_run:
        print("  DRY RUN — ไม่แตะฮาร์ดแวร์")
    else:
        print("  ยกหุ่นขึ้นแท่นให้ขาลอยพ้นพื้นก่อน กด Ctrl-C เพื่อหยุดและปล่อยพัลส์")
    print("=" * 68)

    board: Pca9685 | None = None
    if not args.dry_run:
        board = Pca9685(args.bus, args.addr)
        board.open()
        board.set_frequency(PWM_HZ)
        print(f"เปิดบอร์ดแล้ว พัลส์จริง {board.freq_hz:.2f} Hz")

    servo = Servo(board, args.channel, args.raw)
    atexit.register(servo.release)          # ปล่อยพัลส์เสมอ ไม่ว่าจบแบบไหน

    try:
        if args.release:
            servo.release()
            print("ปล่อยพัลส์ช่องนี้แล้ว — หมุนฮอร์นด้วยมือได้")
            return 0

        if args.interactive:
            do_interactive(servo, rate)
            return 0

        if args.sweep:
            do_sweep(servo, args.step, rate, args.hold)
        else:
            if args.center:
                target = 90.0
            elif args.angle is not None:
                target = args.angle
            elif args.model is not None:
                target = to_servo(clamp(args.model, MODEL_MIN, MODEL_MAX)
                                  if not args.raw else args.model)
            else:
                target = (args.us - PULSE_MIN_US) / US_PER_DEG
            reached = servo.move_to(target, rate)
            print("  " + describe(reached))

        if args.keep:
            print()
            print("ค้างพัลส์ไว้ตามที่สั่ง (--keep) — กด Ctrl-C เพื่อปล่อย")
            while True:
                time.sleep(1.0)
        else:
            time.sleep(args.hold)

    except KeyboardInterrupt:
        print("\nหยุดแล้ว")
    finally:
        servo.release()
        if board is not None:
            board.close()
        print("ปล่อยพัลส์เรียบร้อย")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
