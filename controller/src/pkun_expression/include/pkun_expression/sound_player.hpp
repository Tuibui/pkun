#ifndef PKUN_EXPRESSION__SOUND_PLAYER_HPP_
#define PKUN_EXPRESSION__SOUND_PLAYER_HPP_

#include <string>
#include <vector>

#include <sys/types.h>

namespace pkun_expression
{

/// Fire-and-forget WAV playback via an external player (aplay by default).
///
/// Deliberately not an audio library: the robot plays short cues, and shelling
/// out keeps ALSA device handling out of this process. The child is reaped
/// without blocking, so a stuck player cannot wedge the node.
class SoundPlayer
{
public:
  SoundPlayer(std::string command, std::vector<std::string> args, std::string sound_dir);
  ~SoundPlayer();

  SoundPlayer(const SoundPlayer &) = delete;
  SoundPlayer & operator=(const SoundPlayer &) = delete;

  /// Play `file` (relative names are resolved against sound_dir).
  /// Returns false with `error` filled if the cue could not be started.
  /// A cue already playing is stopped first: overlapping aplay processes on one
  /// ALSA device produce a device-busy error, not a mix.
  bool play(const std::string & file, std::string & error);

  /// Stop the current cue if one is running.
  void stop();

  /// Reap the child if it has exited. Call periodically.
  void poll();

  bool playing() const {return child_ > 0;}

private:
  std::string command_;
  std::vector<std::string> args_;
  std::string sound_dir_;
  pid_t child_{-1};
};

}  // namespace pkun_expression

#endif  // PKUN_EXPRESSION__SOUND_PLAYER_HPP_
