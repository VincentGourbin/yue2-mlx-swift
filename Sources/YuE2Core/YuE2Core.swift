// YuE2Core.swift - version, errors and debug logging for the YuE2 port
// Copyright 2026 Vincent Gourbin

import Foundation

/// Package-wide namespace: version and shared errors.
public enum YuE2 {
    public static let version = "0.1.0"
}

/// Errors raised by the port, from request validation to cancellation.
public enum YuE2Error: Error, CustomStringConvertible {
    case invalidRequest(String)
    case missingFile(String)
    case weightMismatch(String)
    case nonFiniteLatents
    case cancelled

    public var description: String {
        switch self {
        case .invalidRequest(let reason): return "invalid request: \(reason)"
        case .missingFile(let path): return "missing file: \(path)"
        case .weightMismatch(let reason): return "weight mismatch: \(reason)"
        case .nonFiniteLatents: return "non-finite latents"
        case .cancelled: return "cancelled"
        }
    }
}

/// Debug logging, gated by the `YUE2_DEBUG` environment variable.
public enum YuE2Debug {
    /// Logging is compiled in only when `YUE2_DEBUG` is set in the environment.
    static let enabled = ProcessInfo.processInfo.environment["YUE2_DEBUG"] != nil

    /// Prints `s` (evaluated lazily) on stderr when `YUE2_DEBUG` is set.
    public static func log(_ s: @autoclosure () -> String) {
        guard enabled else { return }
        FileHandle.standardError.write(Data("YUE2 DEBUG: \(s())\n".utf8))
    }
}
