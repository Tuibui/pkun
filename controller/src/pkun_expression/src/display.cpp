#include "pkun_expression/display.hpp"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <sstream>
#include <utility>
#include <vector>

namespace pkun_expression
{
namespace
{

std::string errno_text(const std::string & what)
{
  return what + ": " + std::strerror(errno);
}

}  // namespace

// ---------------------------------------------------------------------------
// console
// ---------------------------------------------------------------------------

bool ConsoleDisplay::open(std::string &)
{
  return true;
}

bool ConsoleDisplay::render(const FaceState & face, std::string &)
{
  // Five lid steps is enough to see a blink go by at 50 Hz in the log.
  static constexpr std::array<const char *, 5> kEyes{"(-)", "(_)", "(o)", "(O)", "(@)"};
  const auto step = static_cast<std::size_t>(
    std::clamp(std::lround(face.eye * (kEyes.size() - 1)), 0L,
    static_cast<long>(kEyes.size() - 1)));

  std::ostringstream out;
  out << kEyes[step] << " " << kEyes[step]
      << "  led=" << face.led_name
      << " (" << static_cast<int>(std::lround(face.led_v * 100.0)) << "%)";
  last_frame_ = out.str();
  return true;
}

// ---------------------------------------------------------------------------
// SSD1306
// ---------------------------------------------------------------------------

Ssd1306Display::Ssd1306Display(std::string bus, uint8_t address)
: bus_(std::move(bus)), address_(address) {}

Ssd1306Display::~Ssd1306Display()
{
  close();
}

bool Ssd1306Display::command(uint8_t value, std::string & error)
{
  if (fd_ < 0) {
    error = "display not open";
    return false;
  }
  uint8_t frame[2] = {0x00, value};    // 0x00 = command stream
  if (::write(fd_, frame, sizeof(frame)) != static_cast<ssize_t>(sizeof(frame))) {
    error = errno_text("ssd1306 command");
    return false;
  }
  return true;
}

bool Ssd1306Display::open(std::string & error)
{
  close();

  fd_ = ::open(bus_.c_str(), O_RDWR);
  if (fd_ < 0) {
    error = errno_text("open " + bus_);
    return false;
  }
  if (::ioctl(fd_, I2C_SLAVE, static_cast<int>(address_)) < 0) {
    error = errno_text("select display address");
    close();
    return false;
  }

  // Standard 128x64 init sequence from the datasheet's application note.
  const uint8_t init[] = {
    0xAE,                 // display off
    0xD5, 0x80,           // clock divide
    0xA8, 0x3F,           // multiplex = 63
    0xD3, 0x00,           // display offset
    0x40,                 // start line 0
    0x8D, 0x14,           // charge pump on
    0x20, 0x00,           // horizontal addressing mode
    0xA1,                 // segment remap
    0xC8,                 // COM scan direction, flipped
    0xDA, 0x12,           // COM pins
    0x81, 0xCF,           // contrast
    0xD9, 0xF1,           // precharge
    0xDB, 0x40,           // VCOM detect
    0xA4,                 // resume from RAM
    0xA6,                 // normal (not inverted)
    0xAF,                 // display on
  };
  for (const auto value : init) {
    if (!command(value, error)) {
      close();
      return false;
    }
  }

  clear();
  return flush(error);
}

void Ssd1306Display::close()
{
  if (fd_ >= 0) {
    std::string ignored;
    clear();
    flush(ignored);
    command(0xAE, ignored);    // display off, so it does not burn in
    ::close(fd_);
    fd_ = -1;
  }
}

void Ssd1306Display::clear()
{
  buffer_.fill(0);
}

void Ssd1306Display::fill_rect(int x, int y, int w, int h, bool on)
{
  const int x0 = std::max(0, x);
  const int y0 = std::max(0, y);
  const int x1 = std::min(kWidth, x + w);
  const int y1 = std::min(kHeight, y + h);

  for (int py = y0; py < y1; ++py) {
    const int page = py / 8;
    const auto bit = static_cast<uint8_t>(1u << (py % 8));
    for (int px = x0; px < x1; ++px) {
      auto & cell = buffer_[static_cast<std::size_t>(page * kWidth + px)];
      cell = on ? static_cast<uint8_t>(cell | bit) : static_cast<uint8_t>(cell & ~bit);
    }
  }
}

bool Ssd1306Display::flush(std::string & error)
{
  if (fd_ < 0) {
    error = "display not open";
    return false;
  }
  if (!command(0x21, error) || !command(0, error) || !command(kWidth - 1, error) ||
    !command(0x22, error) || !command(0, error) || !command(kPages - 1, error))
  {
    return false;
  }

  // 1 KiB of framebuffer in 32-byte chunks: most I2C adapters refuse a single
  // transfer that large.
  constexpr std::size_t kChunk = 32;
  std::vector<uint8_t> frame(kChunk + 1);
  frame[0] = 0x40;    // data stream

  for (std::size_t offset = 0; offset < buffer_.size(); offset += kChunk) {
    const std::size_t n = std::min(kChunk, buffer_.size() - offset);
    std::memcpy(frame.data() + 1, buffer_.data() + offset, n);
    if (::write(fd_, frame.data(), n + 1) != static_cast<ssize_t>(n + 1)) {
      error = errno_text("ssd1306 data");
      return false;
    }
  }
  return true;
}

bool Ssd1306Display::render(const FaceState & face, std::string & error)
{
  clear();

  // Two eyes. The lid closes from the top, which reads as a blink rather than
  // the eye simply shrinking.
  constexpr int kEyeW = 34;
  constexpr int kEyeH = 34;
  constexpr int kEyeY = 12;
  const std::array<int, 2> eye_x{22, 72};

  const double opening = std::clamp(face.eye, 0.0, 1.0);
  const int visible = std::max(2, static_cast<int>(std::lround(kEyeH * opening)));
  const int top = kEyeY + (kEyeH - visible) / 2;

  for (const int x : eye_x) {
    fill_rect(x, top, kEyeW, visible, true);
    // Pupil highlight, punched back out of the filled eye.
    if (visible > 10) {
      fill_rect(x + kEyeW / 2 - 4, top + visible / 2 - 4, 8, 8, false);
    }
  }

  // Brightness bar along the bottom stands in for LED intensity.
  const int bar = static_cast<int>(
    std::lround(std::clamp(face.led_v, 0.0, 1.0) * (kWidth - 8)));
  fill_rect(4, kHeight - 6, bar, 4, true);

  return flush(error);
}

// ---------------------------------------------------------------------------

std::unique_ptr<Display> make_display(
  const std::string & kind, const std::string & bus, uint8_t address,
  std::string & error)
{
  if (kind == "console") {
    return std::make_unique<ConsoleDisplay>();
  }
  if (kind == "ssd1306") {
    return std::make_unique<Ssd1306Display>(bus, address);
  }
  error = "unknown display backend '" + kind + "' (want 'console' or 'ssd1306')";
  return nullptr;
}

}  // namespace pkun_expression
