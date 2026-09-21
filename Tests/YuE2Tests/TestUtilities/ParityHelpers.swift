// ParityHelpers.swift - shared tier-2 real-checkpoint parity utilities (plan §8.2)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
@testable import YuE2Core

/// `$YUE2_MODELS_DIR`, or `nil` when unset — tier-2 suites gate on this via `.enabled(if:)`.
func modelsDir() -> URL? {
    guard let path = ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"], !path.isEmpty else { return nil }
    return URL(fileURLWithPath: path)
}

/// Loads `$YUE2_MODELS_DIR/parity/<name>.safetensors`, written by `Scripts/reference/real_fixtures.py`.
func loadFixture(name: String) throws -> [String: MLXArray] {
    guard let dir = modelsDir() else {
        throw YuE2Error.missingFile("YUE2_MODELS_DIR is not set")
    }
    return try loadArrays(url: dir.appendingPathComponent("parity/\(name).safetensors"))
}

/// `mean|a - b| / mean|b|`.
func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
    let num = MLX.mean(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    let den = MLX.mean(MLX.abs(b.asType(.float32))).item(Float.self)
    return num / den
}

/// `max|a - b|`.
func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
}

/// `10*log10(Σref² / Σ(ref-test)²)`, in dB — higher is better (E1, plan/13-ios-backend.md §13.5).
func snrDB(reference: MLXArray, test: MLXArray) -> Float {
    let ref32 = reference.asType(.float32)
    let signal = MLX.sum(ref32 * ref32).item(Float.self)
    let noise = MLX.sum(MLX.square(ref32 - test.asType(.float32))).item(Float.self)
    return 10 * log10(signal / noise)
}
