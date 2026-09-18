#ifndef PKUN_SERVO_DRIVER__PCA9685_HPP_
#define PKUN_SERVO_DRIVER__PCA9685_HPP_

#include <cstdint>
#include <string>

namespace pkun_servo_driver
{

/// Minimal PCA9685 16-channel PWM driver over Linux i2c-dev.
///
/// Deliberately uses plain read()/write() after an I2C_SLAVE ioctl rather than
/// the SMBus helpers, so the package needs no libi2c-dev dependency.
///
/// Not thread safe: one instance is owned by one node and touched only from
/// the control timer.
class Pca9685
{
public:
  /// PCA9685 oscillator, used for the prescale calculation.
  static constexpr double kOscHz = 25'000'000.0;
  static constexpr uint16_t kTicks = 4096;
  static constexpr uint8_t kNumChannels = 16;

  Pca9685(std::string bus, uint8_t address);
  ~Pca9685();

  Pca9685(const Pca9685 &) = delete;
  Pca9685 & operator=(const Pca9685 &) = delete;

  /// Open the bus and put the chip into a known state (auto-increment on, all
  /// outputs off). Returns false and fills `error` on failure.
  bool open(std::string & error);
  void close();
  bool is_open() const {return fd_ >= 0;}

  /// Set the shared PWM frequency for all 16 channels. Analog servos want 50 Hz.
  /// The chip must be put to sleep to change the prescaler, so this drops the
  /// outputs briefly -- only call it while deactivated.
  bool set_frequency(double hz, std::string & error);

  /// Raw 12-bit on/off tick setting for one channel.
  bool set_pwm(uint8_t channel, uint16_t on, uint16_t off, std::string & error);

  /// Drive one channel with a pulse of `microseconds`, using the frequency last
  /// passed to set_frequency(). A width of 0 releases the channel (no pulse).
  bool set_pulse_us(uint8_t channel, double microseconds, std::string & error);

  /// Release every channel on this board.
  bool all_off(std::string & error);

  const std::string & bus() const {return bus_;}
  uint8_t address() const {return address_;}
  double frequency() const {return frequency_hz_;}

private:
  bool write_register(uint8_t reg, uint8_t value, std::string & error);
  bool read_register(uint8_t reg, uint8_t & value, std::string & error);

  std::string bus_;
  uint8_t address_;
  int fd_{-1};
  double frequency_hz_{50.0};
};

}  // namespace pkun_servo_driver

#endif  // PKUN_SERVO_DRIVER__PCA9685_HPP_
