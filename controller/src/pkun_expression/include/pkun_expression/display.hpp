#ifndef PKUN_EXPRESSION__DISPLAY_HPP_
#define PKUN_EXPRESSION__DISPLAY_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace pkun_expression
{

/// What the face should currently look like.
struct FaceState
{
  double eye{1.0};        // 0 = closed, 1 = wide open
  double led_v{0.35};     // brightness 0..1
  uint8_t led{1};         // colour index, see Expression.msg
  std::string led_name{"rest"};
};

/// A face display. Kept behind an interface so the node can run on a laptop
/// with no panel attached -- pick the console backend and the gestures are
/// still observable in the log.
class Display
{
public:
  virtual ~Display() = default;
  virtual bool open(std::string & error) = 0;
  virtual void close() = 0;
  virtual bool render(const FaceState & face, std::string & error) = 0;
};

/// Logs the face as text. Default backend, no hardware needed.
class ConsoleDisplay : public Display
{
public:
  bool open(std::string & error) override;
  void close() override {}
  bool render(const FaceState & face, std::string & error) override;

  /// Last frame rendered, as an ASCII line. The node logs this.
  const std::string & last_frame() const {return last_frame_;}

private:
  std::string last_frame_;
};

/// SSD1306 128x64 monochrome OLED over I2C.
///
/// Draws two eyes whose lid opening tracks `eye`, and a brightness bar for
/// led_v. Colour is monochrome here, so the LED index is shown as bar style
/// rather than hue -- swap in an RGB panel and only this class changes.
class Ssd1306Display : public Display
{
public:
  static constexpr int kWidth = 128;
  static constexpr int kHeight = 64;
  static constexpr int kPages = kHeight / 8;

  Ssd1306Display(std::string bus, uint8_t address);
  ~Ssd1306Display() override;

  bool open(std::string & error) override;
  void close() override;
  bool render(const FaceState & face, std::string & error) override;

private:
  bool command(uint8_t value, std::string & error);
  bool flush(std::string & error);
  void clear();
  void fill_rect(int x, int y, int w, int h, bool on);

  std::string bus_;
  uint8_t address_;
  int fd_{-1};
  std::array<uint8_t, kWidth * kPages> buffer_{};
};

/// Factory: "console" or "ssd1306".
std::unique_ptr<Display> make_display(
  const std::string & kind, const std::string & bus, uint8_t address,
  std::string & error);

}  // namespace pkun_expression

#endif  // PKUN_EXPRESSION__DISPLAY_HPP_
