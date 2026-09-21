// NDArrayBridge.swift - MLXArray <-> CoreAI.NDArray conversion (fp16), T-6.3 (E3)
// Copyright 2026 Vincent Gourbin
//
// Every contract used here is verified directly against this SDK's real `.swiftinterface`
// files (not demangled symbols, not third-party docs):
//   MacOSX27.0.sdk/System/Library/SubFrameworks/CoreAIRuntime.framework/.../CoreAIRuntime.swiftinterface
//   MacOSX27.0.sdk/System/Library/SubFrameworks/CoreAIDelegates.framework/.../CoreAIDelegates.swiftinterface
// `tasks/JOURNAL.md` T-6.3 records the exact grep/read commands used to confirm the signatures
// below, which the prior session (blocked in `tasks/ASK.md`) had only been able to reconstruct
// by demangling exported symbols.

#if canImport(CoreAI)
import CoreAI
import Foundation
import MLX

/// Bridges `MLXArray` and Core AI's `NDArray`. fp16 only, matching T-6.1's E1 precision — the
/// `.aimodel` assets `Scripts/coreai/export_vae.py`/`export_nar_stack.py` produce are exported
/// `.half()`.
///
/// T-6.4b: both directions perform exactly **one** memcpy, straight into/out of the other side's
/// own backing storage — no intermediate Swift `[Float16]` array (T-6.4's original version did
/// `MLXArray.asArray` then `NDArray.init(scalars:)`, or `NDArray.View.withUnsafePointer` into
/// `Array(UnsafeBufferPointer(...))` then `MLXArray.init(_:_:)` — two copies each way). Neither
/// direction is truly zero-copy: `NDArray` and `MLXArray` each own separate backing memory with
/// no documented donation/aliasing contract in this SDK (`CoreAIRuntime.swiftinterface`'s
/// `NDArray.MutableRawView.init(mutableBytes:)`/`RawView.init(bytes:)` *look* like they could
/// alias a caller-supplied buffer, but nothing states whether Core AI retains, copies, or mutates
/// past the call — not worth the risk for a K/V buffer that's read on every one of a chunk's 64
/// velocity evaluations).
@available(macOS 27.0, iOS 27.0, *)
enum NDArrayBridge {
    /// `array`: any shape/dtype, cast to fp16, then copied once from `MLXArray`'s own contiguous
    /// buffer (`asData(access: .noCopyIfContiguous)` — zero-copy on this side, valid because a
    /// freshly-`eval`'d `asType` cast is always contiguous) straight into a freshly-allocated
    /// `NDArray`'s storage (`mutableView(as:).withUnsafeMutablePointer`, verified against
    /// `CoreAIRuntime.swiftinterface`).
    static func ndArray(from array: MLXArray) -> NDArray {
        let fp16 = array.asType(.float16)
        eval(fp16)
        let source = fp16.asData(access: .noCopyIfContiguous)
        var nd = NDArray(shape: array.shape, scalarType: .float16)
        nd.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
            source.data.withUnsafeBytes { bytes in
                UnsafeMutableRawPointer(pointer).copyMemory(
                    from: bytes.baseAddress!, byteCount: bytes.count)
            }
        }
        return nd
    }

    /// Reads a Core AI output `NDArray` back into an fp32 `MLXArray`. Per Apple's documentation
    /// for `InferenceFunction.run(inputs:states:outputViews:)`: "Any `NDArray` values in the
    /// returned outputs have a row-major contiguous layout" — so reading exactly
    /// `shape.reduce(1, *)` contiguous elements via `View.withUnsafePointer` is always valid for
    /// a `run()` output (never a partial/strided view); this helper is not meant for arbitrary
    /// `NDArray`s built elsewhere with custom strides. The pointer is handed straight to
    /// `MLXArray.init(_:UnsafeBufferPointer<T>,_:)` (`mlx_array_new_data` copies once into MLX's
    /// own storage), never through a Swift array.
    static func mlxArray(from ndArray: NDArray) throws -> MLXArray {
        let shape = ndArray.shape
        let count = shape.reduce(1, *)
        let view = ndArray.view(as: Float16.self)
        let fp16: MLXArray = try view.withUnsafePointer { pointer, _, _ in
            MLXArray(UnsafeBufferPointer(start: pointer, count: count), shape)
        }
        return fp16.asType(.float32)
    }

    /// Non-fp16 scalar inputs (T-6.4, E4): `NARStack.forward`'s `raw_t` (fp32 scalar) and
    /// `key_padding_mask` (bool) never round-trip through `MLXArray`, so they skip `ndArray(from:)`
    /// entirely — `NDArray.init<Scalar: BitwiseCopyable>(scalars:shape:)` takes any such value
    /// directly (verified against the same `.swiftinterface` files as the rest of this bridge).
    static func ndArray<Scalar: BitwiseCopyable>(scalars: [Scalar], shape: [Int]) -> NDArray {
        NDArray(scalars: scalars, shape: shape)
    }
}
#endif
