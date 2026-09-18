#include "pkun_servo_driver/pca9685.hpp"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <utility>

namespace pkun_servo_driver
{
namespace
{

constexpr uint8_t kRegMode1 = 0x00;
constexpr uint8_t kRegMode2 = 0x01;
constexpr uint8_t kRegLed0OnL = 0x06;
constexpr uint8_t kRegPrescale = 0xFE;

constexpr uint8_t kMode1Restart = 0x80;
constexpr uint8_t kMode1AutoInc = 0x20;
constexpr uint8_t kMode1Sleep = 0x10;
constexpr uint8_t kMode1AllCall = 0x01;

constexpr uint8_t kMode2OutDrive = 0x04;  // totem pole output

std::string errno_text(const std::string & what)
{
  return what + ": " + std::strerror(errno);
}

}  // namespace

Pca9685::Pca9685(std::string bus, uint8_t address)
: bus_(std::move(bus)), address_(address) {}

Pca9685::~Pca9685()
{
  close();
}

bool Pca9685::open(std::string & error)
{
  close();

  fd_ = ::open(bus_.c_str(), O_RDWR);
  if (fd_ < 0) {
    error = errno_text("open " + bus_);
    return false;
  }
  if (::ioctl(fd_, I2C_SLAVE, static_cast<int>(address_)) < 0) {
    error = errno_text("select i2c address");
    close();
    return false;
  }

  // A read is the cheapest way to prove something is actually acking at this
  // address -- open() alone succeeds even with nothing plugged in.
  uint8_t mode1 = 0;
  if (!read_register(kRegMode1, mode1, error)) {
    close();
    return false;
  }

  if (!write_register(kRegMode1, kMode1AutoInc | kMode1AllCall, error) ||
    !write_register(kRegMode2, kMode2OutDrive, error))
  {
    close();
    return false;
  }

  std::string ignored;
  all_off(ignored);
  return true;
}

void Pca9685::close()
{
  if (fd_ >= 0) {
    std::string ignored;
    all_off(ignored);
    ::close(fd_);
    fd_ = -1;
  }
}

bool Pca9685::write_register(uint8_t reg, uint8_t value, std::string & error)
{
  if (fd_ < 0) {
    error = "i2c device not open";
    return false;
  }
  uint8_t buffer[2] = {reg, value};
  if (::write(fd_, buffer, sizeof(buffer)) != static_cast<ssize_t>(sizeof(buffer))) {
    error = errno_text("i2c write");
    return false;
  }
  return true;
}

bool Pca9685::read_register(uint8_t reg, uint8_t & value, std::string & error)
{
  if (fd_ < 0) {
    error = "i2c device not open";
    return false;
  }
  if (::write(fd_, &reg, 1) != 1) {
    error = errno_text("i2c set read pointer");
    return false;
  }
  if (::read(fd_, &value, 1) != 1) {
    error = errno_text("i2c read");
    return false;
  }
  return true;
}

bool Pca9685::set_frequency(double hz, std::string & error)
{
  if (hz < 24.0 || hz > 1526.0) {
    error = "pwm frequency out of range for PCA9685 (24..1526 Hz)";
    return false;
  }

  const double raw = kOscHz / (static_cast<double>(kTicks) * hz);
  int prescale = static_cast<int>(std::lround(raw)) - 1;
  if (prescale < 3) {prescale = 3;}
  if (prescale > 255) {prescale = 255;}

  uint8_t mode1 = 0;
  if (!read_register(kRegMode1, mode1, error)) {return false;}

  // The prescaler is only writable while the oscillator is asleep.
  const uint8_t sleeping = static_cast<uint8_t>((mode1 & ~kMode1Restart) | kMode1Sleep);
  if (!write_register(kRegMode1, sleeping, error) ||
    !write_register(kRegPrescale, static_cast<uint8_t>(prescale), error) ||
    !write_register(kRegMode1, mode1, error))
  {
    return false;
  }

  // Datasheet: allow >=500 us for the oscillator to stabilise before RESTART.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!write_register(
      kRegMode1, static_cast<uint8_t>(mode1 | kMode1Restart | kMode1AutoInc), error))
  {
    return false;
  }

  // Store the frequency the chip actually produces, not the one asked for --
  // the prescaler is an integer, so 50 Hz is really 49.7 Hz. Pulse widths are
  // computed from this, and being wrong here skews every servo angle.
  frequency_hz_ = kOscHz / (static_cast<double>(kTicks) * (prescale + 1));
  return true;
}

bool Pca9685::set_pwm(uint8_t channel, uint16_t on, uint16_t off, std::string & error)
{
  if (channel >= kNumChannels) {
    error = "pca9685 channel out of range";
    return false;
  }
  if (fd_ < 0) {
    error = "i2c device not open";
    return false;
  }

  uint8_t buffer[5];
  buffer[0] = static_cast<uint8_t>(kRegLed0OnL + 4 * channel);
  buffer[1] = static_cast<uint8_t>(on & 0xFF);
  buffer[2] = static_cast<uint8_t>(on >> 8);
  buffer[3] = static_cast<uint8_t>(off & 0xFF);
  buffer[4] = static_cast<uint8_t>(off >> 8);

  if (::write(fd_, buffer, sizeof(buffer)) != static_cast<ssize_t>(sizeof(buffer))) {
    error = errno_text("i2c write pwm");
    return false;
  }
  return true;
}

bool Pca9685::set_pulse_us(uint8_t channel, double microseconds, std::string & error)
{
  if (microseconds <= 0.0) {
    // Bit 12 of the OFF register is the "full off" flag: a true release, not a
    // zero-width pulse.
    return set_pwm(channel, 0, kTicks, error);
  }

  const double period_us = 1'000'000.0 / frequency_hz_;
  double ticks = microseconds / period_us * static_cast<double>(kTicks);
  if (ticks < 0.0) {ticks = 0.0;}
  if (ticks > kTicks - 1) {ticks = kTicks - 1;}

  return set_pwm(channel, 0, static_cast<uint16_t>(std::lround(ticks)), error);
}

bool Pca9685::all_off(std::string & error)
{
  bool ok = true;
  for (uint8_t ch = 0; ch < kNumChannels; ++ch) {
    ok = set_pwm(ch, 0, kTicks, error) && ok;
  }
  return ok;
}

}  // namespace pkun_servo_driver
