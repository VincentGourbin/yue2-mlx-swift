// NumpyIO.swift - .npy v1.0 read/write for int32 and float32 arrays (plan §1.6)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Reads and writes the `.npy` v1.0 format used for the port's numeric artifacts:
/// `semantic.npy` (`<i4`, 1-D) and `latent.npy` (`<f4`, `[T, 64]`).
public enum NumpyIO {
    // Raw byte 0x93 followed by ASCII "NUMPY" — NOT `"\u{93}NUMPY".utf8`, which encodes the
    // non-ASCII scalar U+0093 as two UTF-8 bytes (0xC2 0x93) instead of the one raw byte
    // numpy's format expects, silently shifting every offset that follows.
    private static let magic: [UInt8] = [0x93] + Array("NUMPY".utf8)
    private static let headerAlignment = 64

    // MARK: - Reading

    public static func readInt32(_ url: URL) throws -> [Int32] {
        let (descr, _, payload) = try readRaw(url)
        guard descr == "<i4" else {
            throw YuE2Error.invalidRequest("expected .npy dtype <i4, got \(descr)")
        }
        return payload.withUnsafeBytes { raw in
            raw.bindMemory(to: Int32.self).map(Int32.init(littleEndian:))
        }
    }

    public static func readFloat32(_ url: URL) throws -> (shape: [Int], data: [Float]) {
        let (descr, shape, payload) = try readRaw(url)
        guard descr == "<f4" else {
            throw YuE2Error.invalidRequest("expected .npy dtype <f4, got \(descr)")
        }
        let data = payload.withUnsafeBytes { raw in
            raw.bindMemory(to: UInt32.self).map { Float(bitPattern: UInt32(littleEndian: $0)) }
        }
        return (shape, data)
    }

    /// Reads a `<f4` `.npy` file straight into an `MLXArray` of the stored shape.
    public static func readArray(_ url: URL) throws -> MLXArray {
        let (shape, data) = try readFloat32(url)
        return MLXArray(data).reshaped(shape)
    }

    private static func readRaw(_ url: URL) throws -> (descr: String, shape: [Int], payload: Data) {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let head = file.readData(ofLength: 10)
        guard head.count == 10, Array(head.prefix(6)) == magic else {
            throw YuE2Error.invalidRequest("not a .npy file: \(url.path)")
        }
        let headerLen = Int(head[8]) | (Int(head[9]) << 8)
        guard let headerText = String(data: file.readData(ofLength: headerLen), encoding: .ascii) else {
            throw YuE2Error.invalidRequest("unreadable .npy header: \(url.path)")
        }
        let descr = try match(#"'descr'\s*:\s*'([^']+)'"#, in: headerText, file: url)
        let shapeText = try match(#"'shape'\s*:\s*\(([^)]*)\)"#, in: headerText, file: url)
        let shape = shapeText.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let payload = file.readDataToEndOfFile()
        return (descr, shape, payload)
    }

    private static func match(_ pattern: String, in text: String, file: URL) throws -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let result = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let range = Range(result.range(at: 1), in: text)
        else {
            throw YuE2Error.invalidRequest("malformed .npy header in \(file.path)")
        }
        return String(text[range])
    }

    // MARK: - Writing

    public static func writeInt32(_ values: [Int32], url: URL) throws {
        var payload = Data(capacity: values.count * 4)
        for value in values {
            var le = value.littleEndian
            payload.append(Data(bytes: &le, count: 4))
        }
        try write(descr: "<i4", shape: [values.count], payload: payload, url: url)
    }

    public static func writeFloat32(_ values: [Float], shape: [Int], url: URL) throws {
        guard shape.reduce(1, *) == values.count else {
            throw YuE2Error.invalidRequest("shape \(shape) does not match \(values.count) values")
        }
        var payload = Data(capacity: values.count * 4)
        for value in values {
            var le = value.bitPattern.littleEndian
            payload.append(Data(bytes: &le, count: 4))
        }
        try write(descr: "<f4", shape: shape, payload: payload, url: url)
    }

    private static func write(descr: String, shape: [Int], payload: Data, url: URL) throws {
        let shapeText = shape.count == 1 ? "(\(shape[0]),)" : "(\(shape.map(String.init).joined(separator: ", ")))"
        var dict = "{'descr': '\(descr)', 'fortran_order': False, 'shape': \(shapeText), }"
        let prefixLen = magic.count + 2 + 2 // magic + version + header-length field
        let unpadded = prefixLen + dict.utf8.count + 1 // +1 for the trailing '\n'
        let padding = (headerAlignment - unpadded % headerAlignment) % headerAlignment
        dict += String(repeating: " ", count: padding)
        dict += "\n"

        var file = Data(magic)
        file.append(contentsOf: [1, 0]) // version 1.0
        let headerLen = UInt16(dict.utf8.count).littleEndian
        withUnsafeBytes(of: headerLen) { file.append(contentsOf: $0) }
        file.append(contentsOf: dict.utf8)
        file.append(payload)

        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try file.write(to: url)
    }
}
