// FootprintSampler.swift - peak physical footprint of this process, sampled on a background thread
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXProfiler

/// Tracks the peak of this process's physical footprint (`task_vm_info.phys_footprint`, the
/// number iOS's jetsam judges an app against) while it runs — for measuring what MLX's own
/// counters (`Memory.snapshot().peakMemory`) never see: Core AI's buffers, the mapped weight
/// files, Metal's own allocations. Sampling is a plain `Thread` (never a `Task`: it must keep
/// running while the calling code is blocked inside a synchronous MLX `eval`).
public final class FootprintSampler: @unchecked Sendable {
    private let interval: TimeInterval
    private let lock = NSLock()
    private var peakBytes: Int64 = 0
    private var mlxPeakBytes: Int = 0
    private var running = false
    private var thread: Thread?

    public init(intervalMilliseconds: Int = 5) {
        interval = Double(intervalMilliseconds) / 1000
    }

    /// The footprint right now, in bytes.
    public static func current() -> Int64 {
        SystemMetrics.processFootprint()
    }

    public func start() {
        lock.lock()
        if running {
            lock.unlock()
            return
        }
        running = true
        lock.unlock()
        sample() // `NSLock` is not recursive: never sample while holding the lock
        let thread = Thread { [weak self] in
            while let self, self.isRunning {
                self.sample()
                Thread.sleep(forTimeInterval: self.interval)
            }
        }
        thread.name = "yue2.footprint-sampler"
        thread.qualityOfService = .utility
        self.thread = thread
        thread.start()
    }

    /// Stops sampling and returns the peak seen since `start()`, in bytes (a final sample is
    /// taken here, so a peak that persists at the end is never missed).
    @discardableResult
    public func stop() -> Int64 {
        lock.lock()
        running = false
        thread = nil
        lock.unlock()
        sample()
        return peak
    }

    public var peak: Int64 {
        lock.lock()
        defer { lock.unlock() }
        return peakBytes
    }

    public var peakMB: Int { Int(peak / (1024 * 1024)) }

    /// Peak of `MLX.Memory.activeMemory` *sampled* since `start()` — unlike `Memory.peakMemory`
    /// (process-lifetime, no public reset in mlx-swift), this isolates one stage's own peak, at
    /// the cost of missing a spike shorter than the sampling interval.
    public var mlxActivePeakMB: Int {
        lock.lock()
        defer { lock.unlock() }
        return mlxPeakBytes / (1024 * 1024)
    }

    private var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return running
    }

    private func sample() {
        let now = SystemMetrics.processFootprint()
        let mlxNow = MLX.Memory.activeMemory
        lock.lock()
        if now > peakBytes { peakBytes = now }
        if mlxNow > mlxPeakBytes { mlxPeakBytes = mlxNow }
        lock.unlock()
    }
}
