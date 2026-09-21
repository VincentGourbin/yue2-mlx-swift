// YuE2MemoryManager.swift - per-stage GPU cache limits (plan §7, O4; pattern from netflix-void-swift-mlx)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Bounds `MLX.Memory`'s buffer cache per pipeline stage, and reports usage.
public enum YuE2MemoryManager {
    public enum Stage {
        case load
        case ar
        case nar
        case vae
    }

    /// Which hardware budget `configure(for:)` targets. Selected automatically (`#if os(iOS)`);
    /// override only for testing.
    public enum Profile {
        /// M3 Max 96 GB budget, plan §3 — a few GB of cache per stage.
        case mac
        /// iPhone budget (plan §13.1) — a single small cache ceiling and a hard 3 GB memory
        /// limit regardless of stage, since the device has no room for stage-sized headroom.
        case mobile
    }

    /// `YUE2_MEMORY_PROFILE=mobile|mac` in the environment overrides the platform default —
    /// to measure the iPhone cache/memory limits on a Mac (`yue2 decode`, `bench-coreai-nar`,
    /// `generate --profile`) before a device is in hand.
    nonisolated(unsafe) public static var profile: Profile = {
        switch ProcessInfo.processInfo.environment["YUE2_MEMORY_PROFILE"] {
        case "mobile": return .mobile
        case "mac": return .mac
        default: break
        }
        #if os(iOS)
        return .mobile
        #else
        return .mac
        #endif
    }()

    /// `Memory.cacheLimit` (and, on `.mobile`, `Memory.memoryLimit`) recommended per stage.
    public static func configure(for stage: Stage) {
        switch profile {
        case .mac:
            let limitMB: Int
            switch stage {
            case .load: limitMB = 1024
            case .ar: limitMB = 2048
            case .nar: limitMB = 4096
            case .vae: limitMB = 1024
            }
            MLX.Memory.cacheLimit = limitMB * 1024 * 1024
        case .mobile:
            MLX.Memory.cacheLimit = 256 * 1024 * 1024
            MLX.Memory.memoryLimit = 3 * 1024 * 1024 * 1024
        }
    }

    /// Frees cached (not active) GPU buffers between pipeline stages.
    public static func releaseBetweenStages() {
        MLX.Memory.clearCache()
    }

    public static func snapshot() -> (activeMB: Int, peakMB: Int, cacheMB: Int) {
        let snapshot = MLX.Memory.snapshot()
        return (
            snapshot.activeMemory / (1024 * 1024),
            snapshot.peakMemory / (1024 * 1024),
            snapshot.cacheMemory / (1024 * 1024)
        )
    }
}
