#!/usr/bin/env python3
"""
servo_sweep.py — ขยับเซอร์โวทีละตัวในราง 0-180 เพื่อจูนให้ "องศาจริง" ตรงกับ "องศาในโมเดล"

ใช้ตอนประกอบ/คาลิเบรต ก่อนที่ ROS จะเข้ามาถือบัสแทน คุยกับ PCA9685 ตรง ๆ
ผ่าน /dev/i2c-* ไม่ต้อง build workspace ก่อน (รีจิสเตอร์ชุดเดียวกับ pca9685.cpp)

ทุกอย่างอ้างจาก servo_map.py ที่เดียว ถ้าตารางมุมตรงนั้นถูก ตัวนี้ก็ถูกตาม

>>> ค่าตั้งต้นคือ DRY RUN ไม่แตะฮาร์ดแวร์ ต้องใส่ --run ถึงจะขยับจริง <<<

ก่อนสั่ง --run:
  * ยกหุ่นขึ้นแท่น ให้ขาลอยพ้นพื้น ขาที่กำลังจูนจะรับน้ำหนักไม่ได้
  * จ่ายไฟเซอร์โวแยกจากไฟลอจิก อย่าดึงจากขา 5V ของ Pi
  * มือแตะสวิตช์ตัดไฟไว้ ถ้าฮอร์นใส่ผิดร่อง ขาจะกระแทกสุดรางทันทีที่พัลส์แรกออก

รัน:
  python3 servo_sweep.py --list                        ดูช่องทั้งหมด
  python3 servo_sweep.py --joint fl_knee --sweep       ดูตารางว่าจะสั่งอะไรบ้าง (dry run)
  python3 servo_sweep.py --joint fl_knee --sweep --run  กวาดจริง
  python3 servo_sweep.py --joint fl_knee --goto 90 --run
  python3 servo_sweep.py --joint fl_knee --model 0 --run   ไปที่ท่า zero ของข้อนั้น
  python3 servo_sweep.py --home --run                  ทุกข้อไปท่า zero pose พร้อมกัน
  python3 servo_sweep.py --release --run               ปล่อยพัลส์ทุกช่อง (ขยับด้วยมือได้)
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from leg_kinematics import PARAMS, leg_fk
from servo_map import (
    JOINT_ORDER,
    PULSE_MAX_US,
    PULSE_MIN_US,
    PWM_HZ,
    SERVO_MAP,
    SERVO_SPAN_DEG,
    JointMap,
)

# ---------------------------------------------------------------------------
# 1) PCA9685 — รีจิสเตอร์ชุดเดียวกับ controller/src/pkun_servo_driver/src/pca9685.cpp
# ---------------------------------------------------------------------------

I2C_SLAVE = 0x0703          # ioctl เลือกที่อยู่อุปกรณ์บนบัส

REG_MODE1 = 0x00
REG_LED0_ON_L = 0x06
REG_PRESCALE = 0xFE

MODE1_RESTART = 0x80
MODE1_AUTOINC = 0x20
MODE1_SLEEP = 0x10

OSC_HZ = 25_000_000.0
TICKS = 4096
NUM_CHANNELS = 16

BOARD_ADDR: Tuple[int, ...] = (0x40, 0x41)   # ตรงกับ board_addresses ใน servos.yaml


class Pca9685:
    """ตัวขับ PCA9685 ขั้นต่ำ เขียนผ่าน i2c-dev ตรง ๆ ไม่ต้องพึ่ง smbus."""

    def __init__(self, bus: str, address: int) -> None:
        self.bus = bus
        self.address = address
        self.fd = -1
        self.freq_hz = PWM_HZ

    # -- เปิด/ปิด ---------------------------------------------------------

    def open(self) -> None:
        """เปิดบัสแล้วตั้งชิปให้อยู่ในสถานะที่รู้แน่ (auto-increment on, ทุกช่องดับ)."""
        self.fd = os.open(self.bus, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, self.address)
        self._write_reg(REG_MODE1, MODE1_AUTOINC)
        time.sleep(0.005)
        self.all_off()

    def close(self) -> None:
        if self.fd >= 0:
            try:
                self.all_off()
            finally:
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

    def set_frequency(self, hz: float) -> None:
        """ตั้งความถี่ร่วมของทั้ง 16 ช่อง — ต้องพักออสซิลเลเตอร์ก่อนถึงเขียน prescale ได้."""
        prescale = int(round(OSC_HZ / (TICKS * hz))) - 1
        prescale = max(3, min(255, prescale))

        old = self._read_reg(REG_MODE1)
        self._write_reg(REG_MODE1, (old & ~MODE1_RESTART) | MODE1_SLEEP)
        self._write_reg(REG_PRESCALE, prescale)
        self._write_reg(REG_MODE1, old)
        time.sleep(0.001)
        self._write_reg(REG_MODE1, old | MODE1_RESTART | MODE1_AUTOINC)

        # prescale เป็นจำนวนเต็ม ความถี่จริงจึงไม่ตรง 50.000 เป๊ะ (ได้ ~49.7)
        # ต้องคิดพัลส์จากค่าจริง ไม่ใช่ค่าที่ขอ ไม่งั้นความกว้างพัลส์เพี้ยนไปทั้งแผง
        self.freq_hz = OSC_HZ / (TICKS * (prescale + 1))

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        reg = REG_LED0_ON_L + 4 * channel
        os.write(self.fd, bytes((reg, on & 0xFF, (on >> 8) & 0x0F,
                                 off & 0xFF, (off >> 8) & 0x0F)))

    def set_pulse_us(self, channel: int, microseconds: float) -> None:
        """สั่งพัลส์กว้างกี่ไมโครวินาที — 0 = ปล่อยช่อง (ไม่มีพัลส์ออก)."""
        if microseconds <= 0.0:
            self.set_pwm(channel, 0, 0x1000)     # บิต full-off
            return
        ticks = int(round(microseconds * 1e-6 * self.freq_hz * TICKS))
        self.set_pwm(channel, 0, max(1, min(TICKS - 1, ticks)))

    def all_off(self) -> None:
        for ch in range(NUM_CHANNELS):
            self.set_pwm(ch, 0, 0x1000)


# ---------------------------------------------------------------------------
# 2) สิ่งที่โมเดลทำนายว่าจะเกิดขึ้น
# ---------------------------------------------------------------------------


def foot_for(joint: JointMap, model_deg: float) -> np.ndarray:
    """ปลายเท้าจะไปอยู่ตรงไหน ถ้าขยับ 'ข้อนี้ข้อเดียว' ส่วนอีกสองข้อค้างที่ 0.

    ใช้เทียบกับของจริง: วัดปลายเท้าบนหุ่นแล้วดูว่าตรงกับคอลัมน์นี้ไหม
    ถ้าเครื่องหมายกลับด้าน แปลว่า direction ของข้อนั้นตั้งผิด
    """
    q = np.zeros(3)
    idx = ("abduction", "hip_pitch", "knee").index(joint.joint)
    q[idx] = np.radians(model_deg)
    return leg_fk(joint.leg, q, PARAMS)[-1][:3, 3]


def sweep_points(joint: JointMap, step: float, raw: bool) -> List[float]:
    """รายการมุมเซอร์โวที่จะไล่: ต่ำ -> สูง -> กลับ (ไป-กลับเพื่อดู backlash)."""
    if raw:
        lo, hi = 0.0, SERVO_SPAN_DEG
    else:
        lo, hi = joint.servo_range()
    n = max(2, int(round((hi - lo) / step)) + 1)
    up = list(np.linspace(lo, hi, n))
    return up + up[-2::-1]


# ---------------------------------------------------------------------------
# 3) การสั่งงาน
# ---------------------------------------------------------------------------


def describe(joint: JointMap, servo_deg: float) -> str:
    """หนึ่งบรรทัด: เซอร์โวเท่าไร = โมเดลเท่าไร = พัลส์เท่าไร = ปลายเท้าอยู่ไหน."""
    model = joint.to_model(servo_deg)
    us = joint.to_pulse_us(servo_deg)
    foot = foot_for(joint, model)
    warn = "" if joint.model_min <= model <= joint.model_max else "  << นอกลิมิตโมเดล"
    return (f"  servo {servo_deg:6.1f}°   model {model:+7.1f}°   {us:7.1f} us   "
            f"foot ({foot[0]:+7.1f}, {foot[1]:+7.1f}, {foot[2]:+7.1f}){warn}")


class Driver:
    """ห่อ PCA9685 ให้สั่งเป็น 'ชื่อข้อต่อ + มุมเซอร์โว' และรองรับโหมด dry run."""

    def __init__(self, bus: str, live: bool) -> None:
        self.live = live
        self.boards: Dict[int, Pca9685] = {}
        if not live:
            return
        for idx, addr in enumerate(BOARD_ADDR):
            board = Pca9685(bus, addr)
            try:
                board.open()
                board.set_frequency(PWM_HZ)
            except OSError as exc:
                self.close()
                raise SystemExit(
                    f"เปิดบอร์ด {addr:#04x} บน {bus} ไม่ได้: {exc}\n"
                    f"  - บัสถูกไหม? ลอง: i2cdetect -y {bus[-1]}  (ควรเห็น 0x40 กับ 0x41)\n"
                    f"  - สิทธิ์พอไหม? sudo usermod -aG i2c $USER แล้ว log out/in") from exc
            self.boards[idx] = board
        print(f"เปิดบอร์ดแล้ว {len(self.boards)} ตัว @ "
              f"{', '.join(f'{a:#04x}' for a in BOARD_ADDR)}  "
              f"(พัลส์จริง {self.boards[0].freq_hz:.2f} Hz)")

    def write(self, joint: JointMap, servo_deg: float) -> None:
        if not self.live:
            return
        self.boards[joint.board].set_pulse_us(joint.channel, joint.to_pulse_us(servo_deg))

    def release(self, joint: JointMap | None = None) -> None:
        if not self.live:
            return
        if joint is None:
            for board in self.boards.values():
                board.all_off()
        else:
            self.boards[joint.board].set_pulse_us(joint.channel, 0.0)

    def close(self) -> None:
        for board in self.boards.values():
            board.close()
        self.boards.clear()


def move_to(drv: Driver, joint: JointMap, target: float, dwell: float,
            step: float, current: float | None) -> float:
    """เดินจาก current ไป target ทีละ step เพื่อไม่ให้กระชาก. คืนมุมปลายทาง."""
    if current is None:
        drv.write(joint, target)
        time.sleep(max(dwell, 0.3))          # ก้าวแรกอาจไกล ให้เวลาเซอร์โวตามให้ทัน
        return target
    n = max(1, int(np.ceil(abs(target - current) / step)))
    for v in np.linspace(current, target, n + 1)[1:]:
        drv.write(joint, float(v))
        time.sleep(dwell)
    return target


# ---------------------------------------------------------------------------
# 4) CLI
# ---------------------------------------------------------------------------


def print_list() -> None:
    print(f"{'joint':<16}{'board':>6}{'ch':>4}{'home':>6}{'dir':>5}"
          f"{'servo range':>15}{'model range':>16}")
    print("-" * 68)
    for name in JOINT_ORDER:
        j = SERVO_MAP[name]
        lo, hi = j.servo_range()
        print(f"{j.name:<16}{BOARD_ADDR[j.board]:#06x}{j.channel:>4}{j.home_deg:>6.0f}"
              f"{j.direction:>+5}{f'{lo:.0f}..{hi:.0f}':>15}"
              f"{f'{j.model_min:+.0f}..{j.model_max:+.0f}':>16}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="กวาดเซอร์โว 0-180 เพื่อจูนมุมจริงให้ตรงกับมุมในโมเดล",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="ดูช่องทั้งหมดแล้วจบ")
    ap.add_argument("--joint", metavar="NAME", help="ชื่อข้อต่อ เช่น fl_knee")
    ap.add_argument("--sweep", action="store_true", help="กวาดทั้งช่วง ไป-กลับ")
    ap.add_argument("--goto", type=float, metavar="DEG", help="ไปที่มุมเซอร์โว 0-180")
    ap.add_argument("--model", type=float, metavar="DEG", help="ไปที่มุมโมเดล (แปลงให้)")
    ap.add_argument("--home", action="store_true", help="ทุกข้อไปท่า zero pose")
    ap.add_argument("--release", action="store_true", help="ปล่อยพัลส์ทุกช่อง")
    ap.add_argument("--raw", action="store_true",
                    help="ใช้ราง 0-180 เต็ม ไม่หยุดที่ลิมิตโมเดล (อันตราย)")
    ap.add_argument("--step", type=float, default=2.0, help="ก้าวละกี่องศา (ตั้งต้น 2)")
    ap.add_argument("--dwell", type=float, default=0.02,
                    help="หน่วงต่อก้าว วินาที (ตั้งต้น 0.02 = 100°/s)")
    ap.add_argument("--bus", default="/dev/i2c-1", help="บัส i2c (ตั้งต้น /dev/i2c-1)")
    ap.add_argument("--run", action="store_true",
                    help="สั่งฮาร์ดแวร์จริง (ไม่ใส่ = dry run พิมพ์อย่างเดียว)")
    args = ap.parse_args(argv)

    if args.list:
        print_list()
        return 0

    if not (args.sweep or args.goto is not None or args.model is not None
            or args.home or args.release):
        ap.error("ต้องเลือกอย่างน้อยหนึ่ง: --sweep / --goto / --model / --home / --release "
                 "(หรือ --list)")

    joint: JointMap | None = None
    if args.joint:
        if args.joint not in SERVO_MAP:
            print(f"ไม่รู้จักข้อต่อ {args.joint!r}\nมีให้เลือก: "
                  f"{', '.join(JOINT_ORDER)}", file=sys.stderr)
            return 2
        joint = SERVO_MAP[args.joint]
    elif not (args.home or args.release):
        ap.error("ต้องระบุ --joint ด้วย (ยกเว้น --home / --release)")

    rate = args.step / max(args.dwell, 1e-6)
    print("=" * 78)
    print("DRY RUN — ไม่แตะฮาร์ดแวร์ ใส่ --run เพื่อสั่งจริง"
          if not args.run else "LIVE — กำลังจะสั่งเซอร์โวจริง")
    if args.run:
        print("  ยกหุ่นขึ้นแท่นให้ขาลอยพ้นพื้นก่อน และเตรียมมือไว้ที่สวิตช์ตัดไฟ")
    print(f"  ความเร็ว {rate:.0f}°/s  (step {args.step}° x dwell {args.dwell}s)")
    if rate > 300.0:
        print("  เตือน: เกิน 300°/s ที่ walking_gait.py วางแผนไว้ -> ลด --step หรือเพิ่ม --dwell")
    if args.raw:
        print("  RAW: จะวิ่งเต็มราง 0-180 ข้ามลิมิตโมเดล — โครงอาจชนกันเอง")
    print("=" * 78)

    drv = Driver(args.bus, args.run)
    current: float | None = None
    try:
        if args.release:
            print("ปล่อยพัลส์ทุกช่อง — ขยับด้วยมือได้ ใส่ฮอร์นให้ตรงร่องได้เลย")
            drv.release()
            return 0

        if args.home:
            print("ส่งทุกข้อไปท่า zero pose (ขาเหยียดตรงดิ่งลงพื้นทั้ง 4 ขา):")
            for name in JOINT_ORDER:
                j = SERVO_MAP[name]
                print(f"  {j.name:<16} -> servo {j.home_deg:5.1f}°  "
                      f"({j.to_pulse_us(j.home_deg):7.1f} us)")
                drv.write(j, j.home_deg)
            if args.run:
                time.sleep(0.6)
            return 0

        assert joint is not None
        print(f"ข้อต่อ {joint.name}  [{joint.symbol} {joint.joint}, ขา {joint.leg}]  "
              f"board {BOARD_ADDR[joint.board]:#04x} ch {joint.channel}  "
              f"home {joint.home_deg:.0f}  dir {joint.direction:+d}")
        print()

        if args.model is not None:
            target = joint.to_servo(args.model)
            if not args.raw:
                target = joint.to_servo(joint.clamp_model(args.model))
            print(describe(joint, target))
            current = move_to(drv, joint, target, args.dwell, args.step, current)

        elif args.goto is not None:
            target = float(np.clip(args.goto, 0.0, SERVO_SPAN_DEG))
            if not args.raw:
                lo, hi = joint.servo_range()
                target = float(np.clip(target, lo, hi))
            print(describe(joint, target))
            current = move_to(drv, joint, target, args.dwell, args.step, current)

        else:  # --sweep
            pts = sweep_points(joint, args.step, args.raw)
            print(f"กวาด {len(pts)} จุด ไป-กลับ:")
            print()
            # พิมพ์เฉพาะทุก ๆ 5 องศา ไม่งั้นจอเต็มไปหมด
            shown = 0
            for v in pts:
                if abs(v - round(v / 5.0) * 5.0) < args.step / 2.0:
                    print(describe(joint, float(v)))
                    shown += 1
                current = move_to(drv, joint, float(v), args.dwell, args.step, current)
            print()
            print(f"({shown} บรรทัดจาก {len(pts)} จุด — แสดงทุก ๆ 5°)")

        if args.run:
            print()
            print("ค้างไว้ที่จุดสุดท้าย 1 วินาที แล้วปล่อยพัลส์")
            time.sleep(1.0)
            drv.release(joint)
    except KeyboardInterrupt:
        print("\nยกเลิก — ปล่อยพัลส์ทุกช่อง")
        drv.release()
        return 130
    finally:
        drv.close()

    print()
    print("ถ้าของจริงไม่ตรงกับตารางข้างบน:")
    print("  * ขยับผิดทาง        -> พลิก LEG_DIRECTION ของขานั้นใน servo_map.py")
    print("                        (และ invert ใน servos.yaml ให้ตรงกัน)")
    print("  * ทิศถูกแต่เยื้อง    -> ใส่ trim_deg ของข้อนั้น (องศาที่เยื้อง)")
    print("  * ชนสุดรางก่อนถึงมุม -> วัดมุมที่ชนจริง แล้วหุบ JOINT_SPEC/JointLimits ลง")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
