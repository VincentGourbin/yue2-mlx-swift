// ReferenceProfiles.swift - the six pre-qualified configurations (4/8/16 bits × fast/lean)
// Copyright 2026 Vincent Gourbin
//
// One place that names what the app and the CLI ship: for each weight width, a "fast" profile
// (everything resident, NAR computed in bf16, Mac-sized caches) and a "lean" profile (stage-
// scoped residency, packed NAR, fp16, mobile-sized caches, VAE tile 256). Every field maps to an
// existing knob; the numbers behind each choice are in docs/knowledge/benchmarks/ (70 s song,
// M3 Max, 2026-09-26) and are re-measured by `Scripts/bench-references.sh`.

import Foundation

public struct YuE2ReferenceProfile: Sendable, Identifiable, Equatable {
    public enum Bits: String, CaseIterable, Sendable { case four = "4", eight = "8", sixteen = "16" }
    public enum Kind: String, CaseIterable, Sendable { case fast, lean, tiny }

    public let bits: Bits
    public let kind: Kind
    /// Weight pack (`mlx-prequantized/<preset>[-head]`, or the bf16 checkpoint for `.none`).
    public let quant: YuE2Quantization
    public let quantizeHead: Bool
    public let precision: YuE2ComputePrecision
    public let narCompute: YuE2ExecutionPolicy.NARCompute
    public let compiledDecode: Bool
    public let vaePrecision: VAEPrecision
    public let vaeCoreFrames: Int
    public let releaseWeightsBetweenStages: Bool
    public let memoryProfile: YuE2MemoryManager.Profile
    /// `nil` = the checkpoint's default (32 midpoint steps).
    public let odeSteps: Int?
    /// Fixed MLX cache / GC-threshold (MB) on the mobile profile; `nil` = adaptive sizing.
    public let mobileLimitsMB: (cache: Int, limit: Int)?
    public let summary: String

    public init(
        bits: Bits, kind: Kind, quant: YuE2Quantization, quantizeHead: Bool, precision: YuE2ComputePrecision,
        narCompute: YuE2ExecutionPolicy.NARCompute, compiledDecode: Bool, vaePrecision: VAEPrecision,
        vaeCoreFrames: Int, releaseWeightsBetweenStages: Bool, memoryProfile: YuE2MemoryManager.Profile,
        odeSteps: Int?, mobileLimitsMB: (cache: Int, limit: Int)? = nil, summary: String
    ) {
        self.bits = bits; self.kind = kind; self.quant = quant; self.quantizeHead = quantizeHead
        self.precision = precision; self.narCompute = narCompute; self.compiledDecode = compiledDecode
        self.vaePrecision = vaePrecision; self.vaeCoreFrames = vaeCoreFrames
        self.releaseWeightsBetweenStages = releaseWeightsBetweenStages; self.memoryProfile = memoryProfile
        self.odeSteps = odeSteps; self.mobileLimitsMB = mobileLimitsMB; self.summary = summary
    }

    public static func == (a: YuE2ReferenceProfile, b: YuE2ReferenceProfile) -> Bool { a.id == b.id }

    public var id: String { "\(bits.rawValue)bit-\(kind.rawValue)" }

    /// The published prequantized pack this profile loads (`yue2 download --model <pack>`), or
    /// `nil` for the bf16 checkpoint.
    public var pack: YuE2Pack? { YuE2Pack.matching(quant, quantizeHead: quantizeHead) }

    /// Applies the process-wide knobs (memory profile, NAR compute, compiled decode). The
    /// per-call ones (`quant`, `precision`, VAE, residency, ODE steps) are read by the caller.
    public func applyGlobalPolicy() {
        YuE2MemoryManager.profile = memoryProfile
        YuE2MemoryManager.mobileLimitsOverrideMB = mobileLimitsMB
        YuE2ExecutionPolicy.narCompute = narCompute
        YuE2ExecutionPolicy.compiledDecode = compiledDecode
    }

    public static func named(_ id: String) -> YuE2ReferenceProfile? {
        all.first { $0.id == id }
    }

    public static let all: [YuE2ReferenceProfile] = [
        YuE2ReferenceProfile(
            bits: .four, kind: .fast, quant: .int4Mixed, quantizeHead: true, precision: .bf16,
            narCompute: .dequantized, compiledDecode: false, vaePrecision: .fp16, vaeCoreFrames: 1024,
            releaseWeightsBetweenStages: false, memoryProfile: .mac, odeSteps: nil,
            summary: "int4-mixed-head pack (2.5 GB), everything resident, NAR dequantized to bf16 for the solve"),
        YuE2ReferenceProfile(
            bits: .four, kind: .lean, quant: .int4Mixed, quantizeHead: true, precision: .fp16,
            narCompute: .packed, compiledDecode: false, vaePrecision: .fp16, vaeCoreFrames: 256,
            releaseWeightsBetweenStages: true, memoryProfile: .mobile, odeSteps: nil,
            summary: "int4-mixed-head pack, stage-scoped residency, packed NAR, fp16, VAE tile 256, mobile limits — the iPhone profile"),
        YuE2ReferenceProfile(
            bits: .four, kind: .tiny, quant: .int4Mixed, quantizeHead: true, precision: .fp16,
            narCompute: .packed, compiledDecode: false, vaePrecision: .fp16, vaeCoreFrames: 64,
            releaseWeightsBetweenStages: true, memoryProfile: .mobile, odeSteps: nil,
            mobileLimitsMB: (cache: 256, limit: 3072),
            summary: "int4-mixed-head pack, 4bit-lean plus a 64-frame VAE tile and a 256 MB cache: ≈ 2.6 GB peak for +29 % time, same audio — the most constrained devices"),
        YuE2ReferenceProfile(
            bits: .eight, kind: .fast, quant: .qint8All, quantizeHead: true, precision: .bf16,
            narCompute: .dequantized, compiledDecode: false, vaePrecision: .fp16, vaeCoreFrames: 1024,
            releaseWeightsBetweenStages: false, memoryProfile: .mac, odeSteps: nil,
            summary: "qint8-all-head pack (3.5 GB), everything resident, NAR dequantized to bf16"),
        YuE2ReferenceProfile(
            bits: .eight, kind: .lean, quant: .qint8All, quantizeHead: true, precision: .fp16,
            narCompute: .packed, compiledDecode: false, vaePrecision: .fp16, vaeCoreFrames: 256,
            releaseWeightsBetweenStages: true, memoryProfile: .mobile, odeSteps: nil,
            summary: "qint8-all-head pack, stage-scoped residency, packed NAR, fp16, VAE tile 256, mobile limits"),
        YuE2ReferenceProfile(
            bits: .sixteen, kind: .fast, quant: .none, quantizeHead: false, precision: .bf16,
            narCompute: .packed, compiledDecode: false, vaePrecision: .fp16, vaeCoreFrames: 1024,
            releaseWeightsBetweenStages: false, memoryProfile: .mac, odeSteps: nil,
            summary: "bf16 checkpoint (7.3 GB), everything resident, Mac caches"),
        YuE2ReferenceProfile(
            bits: .sixteen, kind: .lean, quant: .none, quantizeHead: false, precision: .bf16,
            narCompute: .packed, compiledDecode: false, vaePrecision: .fp16, vaeCoreFrames: 256,
            releaseWeightsBetweenStages: true, memoryProfile: .mac, odeSteps: nil,
            summary: "bf16 checkpoint, stage-scoped residency, VAE tile 256 — Mac caches (mobile limits thrash a bf16 working set)"),
    ]
}
