// YuE2MemoryManager.swift - per-stage GPU cache limits (plan §7, O4; pattern from netflix-void-swift-mlx)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
#if os(iOS)
import os
#endif

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
    public enum Profile: Sendable {
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
            let (cacheMB, limitMB) = mobileLimitsMB()
            MLX.Memory.cacheLimit = cacheMB * 1024 * 1024
            MLX.Memory.memoryLimit = limitMB * 1024 * 1024
        }
    }

    /// Mobile budget, sized from what iOS actually leaves the process rather than fixed: the
    /// fixed 256 MB cache / 3 GB limit of the first cut cost +32 % on a 70 s song with the 4-bit
    /// pack on the Mac (buffers released to Metal and reallocated every step) and +73 % in bf16
    /// (working set above the GC threshold). Cache = min(1 GB, available / 6); memory limit
    /// (mlx's cache-GC threshold, not a hard cap) = available − 1.25 GB, never below 3 GB.
    /// `YUE2_CACHE_LIMIT_MB` / `YUE2_MEMORY_LIMIT_MB` override both (measurements). On macOS
    /// (profile forced to `.mobile`) "available" is taken as 6 GB, the iPhone 15 Pro Max figure.
    static func mobileLimitsMB() -> (cache: Int, limit: Int) {
        let env = ProcessInfo.processInfo.environment
        #if os(iOS)
        let availableMB = Int(os_proc_available_memory() / (1024 * 1024))
        #else
        let availableMB = 6 * 1024
        #endif
        let cache = env["YUE2_CACHE_LIMIT_MB"].flatMap(Int.init) ?? min(1024, max(256, availableMB / 6))
        let limit = env["YUE2_MEMORY_LIMIT_MB"].flatMap(Int.init) ?? max(3 * 1024, availableMB - 1280)
        return (cache, limit)
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
