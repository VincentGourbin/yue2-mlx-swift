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

    /// `Memory.cacheLimit` recommended per stage (M3 Max 96 GB budget, plan §3).
    public static func configure(for stage: Stage) {
        let limitMB: Int
        switch stage {
        case .load: limitMB = 1024
        case .ar: limitMB = 2048
        case .nar: limitMB = 4096
        case .vae: limitMB = 1024
        }
        MLX.Memory.cacheLimit = limitMB * 1024 * 1024
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
