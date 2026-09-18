#include "pkun_expression/sound_player.hpp"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>
#include <vector>

namespace pkun_expression
{

SoundPlayer::SoundPlayer(
  std::string command, std::vector<std::string> args, std::string sound_dir)
: command_(std::move(command)), args_(std::move(args)), sound_dir_(std::move(sound_dir))
{
}

SoundPlayer::~SoundPlayer()
{
  stop();
}

bool SoundPlayer::play(const std::string & file, std::string & error)
{
  if (file.empty()) {
    error = "empty sound file";
    return false;
  }

  poll();
  if (playing()) {
    stop();
  }

  std::string path = file;
  if (!path.empty() && path.front() != '/' && !sound_dir_.empty()) {
    path = sound_dir_ + "/" + path;
  }
  if (::access(path.c_str(), R_OK) != 0) {
    error = "cannot read sound file " + path + ": " + std::strerror(errno);
    return false;
  }

  const pid_t pid = ::fork();
  if (pid < 0) {
    error = std::string("fork failed: ") + std::strerror(errno);
    return false;
  }

  if (pid == 0) {
    // Child. Detach from the parent's process group so a Ctrl-C on the launch
    // does not deliver SIGINT here mid-write to the sound card.
    ::setsid();

    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(command_.c_str()));
    for (const auto & arg : args_) {
      argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(const_cast<char *>(path.c_str()));
    argv.push_back(nullptr);

    ::execvp(command_.c_str(), argv.data());
    ::_exit(127);    // exec failed; never return into the forked ROS node
  }

  child_ = pid;
  return true;
}

void SoundPlayer::stop()
{
  if (child_ <= 0) {
    return;
  }
  ::kill(child_, SIGTERM);
  ::waitpid(child_, nullptr, 0);
  child_ = -1;
}

void SoundPlayer::poll()
{
  if (child_ <= 0) {
    return;
  }
  int status = 0;
  const pid_t done = ::waitpid(child_, &status, WNOHANG);
  if (done == child_ || done < 0) {
    child_ = -1;
  }
}

}  // namespace pkun_expression
