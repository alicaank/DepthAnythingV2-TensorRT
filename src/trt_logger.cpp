/**
 * TensorRT Logger Implementation
 */

#include "trt_logger.hpp"
#include <iostream>
#include <ctime>

namespace depth_anything_v2 {

// Global logger instance
TRTLogger gLogger(TRTLogger::Level::WARNING);

TRTLogger::TRTLogger(Level level) : level_(level) {}

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (!shouldLog(severity, level_)) {
        return;
    }

    // Get current time
    time_t now = time(nullptr);
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    // Print log message
    std::cerr << "[" << time_buf << "] "
              << "[TensorRT " << severityToString(severity) << "] "
              << msg << std::endl;
}

void TRTLogger::setLevel(Level level) {
    level_ = level;
}

TRTLogger::Level TRTLogger::getLevel() const {
    return level_;
}

const char* TRTLogger::severityToString(Severity severity) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case Severity::kERROR:
            return "ERROR";
        case Severity::kWARNING:
            return "WARNING";
        case Severity::kINFO:
            return "INFO";
        case Severity::kVERBOSE:
            return "VERBOSE";
        default:
            return "UNKNOWN";
    }
}

bool TRTLogger::shouldLog(Severity severity, Level level) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            return static_cast<int>(level) <= static_cast<int>(Level::ERROR);
        case Severity::kWARNING:
            return static_cast<int>(level) <= static_cast<int>(Level::WARNING);
        case Severity::kINFO:
            return static_cast<int>(level) <= static_cast<int>(Level::INFO);
        case Severity::kVERBOSE:
            return static_cast<int>(level) <= static_cast<int>(Level::VERBOSE);
        default:
            return false;
    }
}

} // namespace depth_anything_v2
