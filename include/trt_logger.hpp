/**
 * TensorRT Logger for Depth Anything V2
 */

#ifndef DEPTH_ANYTHING_V2_TRT_LOGGER_HPP
#define DEPTH_ANYTHING_V2_TRT_LOGGER_HPP

#include <NvInfer.h>
#include <iostream>
#include <string>

namespace depth_anything_v2 {

/**
 * Custom TensorRT logger
 */
class TRTLogger : public nvinfer1::ILogger {
public:
    enum class Level {
        VERBOSE = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
        NONE = 4
    };

    explicit TRTLogger(Level level = Level::WARNING);

    void log(Severity severity, const char* msg) noexcept override;

    void setLevel(Level level);
    Level getLevel() const;

private:
    Level level_;

    static const char* severityToString(Severity severity);
    static bool shouldLog(Severity severity, Level level);
};

// Global logger instance
extern TRTLogger gLogger;

} // namespace depth_anything_v2

#endif // DEPTH_ANYTHING_V2_TRT_LOGGER_HPP
