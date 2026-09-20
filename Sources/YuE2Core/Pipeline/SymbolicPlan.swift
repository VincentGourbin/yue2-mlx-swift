// SymbolicPlan.swift - saved planner output: score, prefix, SHA-256 manifest (pipeline.py)
// Copyright 2026 Vincent Gourbin

import CryptoKit
import Foundation

/// The planner's output: the request, the ABC score (if any), its tokenized ids, the resulting
/// AR prefix, and — when the ABC phase actually ran — its timing/truncation. Mirrors
/// `SymbolicPlan` in `pipeline.py`, including its save/load round trip and tamper detection.
public struct SymbolicPlan: Codable, Equatable {
    public let request: SongRequest
    public let abc: String?
    public let abcIDs: [Int]
    public let prefix: [Int]
    public let timing: GenerationTiming?
    public let truncated: Bool

    public init(
        request: SongRequest, abc: String?, abcIDs: [Int], prefix: [Int],
        timing: GenerationTiming? = nil, truncated: Bool = false
    ) {
        self.request = request
        self.abc = abc
        self.abcIDs = abcIDs
        self.prefix = prefix
        self.timing = timing
        self.truncated = truncated
    }

    private enum CodingKeys: String, CodingKey {
        case request, abc
        case abcIDs = "abc_ids"
        case prefix, timing, truncated
    }

    private static let requiredArtifacts: Set<String> = ["plan.json", "abc_tokens.npy", "prefix.npy"]

    /// Writes `score.abc` (if `abc != nil`), `abc_tokens.npy`, `prefix.npy`, `plan.json`, and a
    /// `plan_manifest.json` of SHA-256 digests over every artifact just written.
    public func save(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let abc {
            try Data(abc.utf8).write(to: directory.appendingPathComponent("score.abc"))
        }
        try NumpyIO.writeInt32(abcIDs.map(Int32.init), url: directory.appendingPathComponent("abc_tokens.npy"))
        try NumpyIO.writeInt32(prefix.map(Int32.init), url: directory.appendingPathComponent("prefix.npy"))

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: directory.appendingPathComponent("plan.json"))

        var names = Self.requiredArtifacts
        if abc != nil { names.insert("score.abc") }
        var hashes: [String: String] = [:]
        for name in names {
            hashes[name] = try Self.sha256(of: directory.appendingPathComponent(name))
        }
        try JSONSerialization.data(withJSONObject: hashes, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("plan_manifest.json"))
    }

    /// Restores an exact planner output, verifying every artifact's hash first (piège n°16:
    /// this is the port's one legitimate reason to re-read a "real" file byte for byte — the
    /// manifest exists specifically to catch hand-edited/corrupted plan directories).
    public static func load(from directory: URL) throws -> SymbolicPlan {
        guard let manifestData = try? Data(contentsOf: directory.appendingPathComponent("plan_manifest.json")),
              let hashes = try? JSONSerialization.jsonObject(with: manifestData) as? [String: String]
        else {
            throw YuE2Error.invalidRequest("Missing or unreadable plan manifest")
        }
        guard requiredArtifacts.isSubset(of: Set(hashes.keys)) else {
            throw YuE2Error.invalidRequest("Incomplete saved plan")
        }
        let allowed = requiredArtifacts.union(["score.abc"])
        for (name, digest) in hashes {
            let fileURL = directory.appendingPathComponent(name)
            let attributes = try? FileManager.default.attributesOfItem(atPath: fileURL.path)
            guard allowed.contains(name), (attributes?[.type] as? FileAttributeType) != .typeSymbolicLink else {
                throw YuE2Error.invalidRequest("Invalid plan artifact: \(name)")
            }
            guard try sha256(of: fileURL) == digest else {
                throw YuE2Error.invalidRequest("Saved plan changed; supply modified ABC as an external planner input")
            }
        }

        let plan = try JSONDecoder().decode(
            SymbolicPlan.self, from: Data(contentsOf: directory.appendingPathComponent("plan.json")))
        let abcIDsOnDisk = try NumpyIO.readInt32(directory.appendingPathComponent("abc_tokens.npy")).map(Int.init)
        let prefixOnDisk = try NumpyIO.readInt32(directory.appendingPathComponent("prefix.npy")).map(Int.init)
        guard abcIDsOnDisk == plan.abcIDs, prefixOnDisk == plan.prefix else {
            throw YuE2Error.invalidRequest("Saved plan token array mismatch")
        }
        if let abc = plan.abc {
            guard hashes["score.abc"] != nil,
                  let scoreOnDisk = try? Data(contentsOf: directory.appendingPathComponent("score.abc")),
                  scoreOnDisk == Data(abc.utf8)
            else {
                throw YuE2Error.invalidRequest("Saved ABC text mismatch")
            }
        }
        return plan
    }

    private static func sha256(of url: URL) throws -> String {
        let digest = SHA256.hash(data: try Data(contentsOf: url))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
