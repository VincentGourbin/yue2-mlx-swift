// SymbolicPlan.swift - saved planner output: score, prefix, SHA-256 manifest (pipeline.py)
// Copyright 2026 Vincent Gourbin

import CryptoKit
import Foundation

/// The planner's output: the request, the ABC score (if any), its tokenized ids, the resulting
/// AR prefix, and — when the ABC phase actually ran — its timing/truncation. Mirrors
/// `SymbolicPlan` in `pipeline.py`, including its save/load round trip and tamper detection.
/// `plan.json` also carries the score's `layout` (ASK Q16), which `pipeline.py` ignores.
public struct SymbolicPlan: Codable, Equatable {
    public let request: SongRequest
    public let abc: String?
    public let abcIDs: [Int]
    public let prefix: [Int]
    public let timing: GenerationTiming?
    public let truncated: Bool
    /// Sections, first and last sung bar, nominal seconds of `abc`; `nil` without a score or
    /// when it does not parse.
    public let layout: ABCScore.Layout?

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
        self.layout = abc.flatMap(ABCScore.layout(of:))
    }

    private enum CodingKeys: String, CodingKey {
        case request, abc
        case abcIDs = "abc_ids"
        case prefix, timing, truncated, layout
    }

    /// `layout` is derived data: always recomputed from the score, so plans saved before it existed get one.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            request: try c.decode(SongRequest.self, forKey: .request), abc: try c.decodeIfPresent(String.self, forKey: .abc),
            abcIDs: try c.decode([Int].self, forKey: .abcIDs), prefix: try c.decode([Int].self, forKey: .prefix),
            timing: try c.decodeIfPresent(GenerationTiming.self, forKey: .timing),
            truncated: try c.decode(Bool.self, forKey: .truncated))
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

extension SymbolicPlan {
    /// This plan with another ABC score: same request, prefix rebuilt around the new score. The
    /// score's tokens are kept as planned up to its first changed line, so an edit after a point
    /// leaves every token before it as the planner wrote it (ASK Q18). Use it to write notes into
    /// a section the plan left silent, before the semantic stage or with `regenerate(score:)`
    /// after it. Needs a plan with a score (`cot` full or melody).
    public func replacingScore(_ abc: String, tokenizer: YuE2Tokenizer) throws -> SymbolicPlan {
        guard abc != self.abc else { return self }
        guard request.cot != .off else {
            throw YuE2Error.invalidRequest("replacingScore needs a plan with a score (cot full or melody)")
        }
        let ids = Self.spliceScoreIDs(old: self.abc ?? "", oldIDs: abcIDs, new: abc, tokenizer: tokenizer)
        var scored = request
        if scored.abc != nil { scored.abc = abc }
        return SymbolicPlan(
            request: scored, abc: abc, abcIDs: ids,
            prefix: try Prefixes.tokenPrefixes(request: scored, tokenizer: tokenizer, abcIDs: ids))
    }

    /// Token ids of `new`: `oldIDs` kept up to the last token that ends before the first changed
    /// line of `old`, the rest encoded afresh. Falls back to encoding `new` whole when `oldIDs`
    /// do not spell `old` byte for byte.
    static func spliceScoreIDs(old: String, oldIDs: [Int], new: String, tokenizer: YuE2Tokenizer) -> [Int] {
        let oldBytes = Array(old.utf8), newBytes = Array(new.utf8)
        guard oldIDs.reduce(0, { $0 + tokenizer.byteCount(of: $1) }) == oldBytes.count,
              tokenizer.decode(oldIDs) == old else { return tokenizer.encode(new) }
        var common = 0
        while common < min(oldBytes.count, newBytes.count), oldBytes[common] == newBytes[common] { common += 1 }
        // Back to the start of the changed line: the tokenizer never merges across a newline there.
        while common > 0, oldBytes[common - 1] != 0x0A { common -= 1 }
        var kept = 0, offset = 0
        for id in oldIDs {
            let n = tokenizer.byteCount(of: id)
            if offset + n > common { break }
            offset += n; kept += 1
        }
        guard let tail = String(bytes: newBytes[offset...], encoding: .utf8) else { return tokenizer.encode(new) }
        return Array(oldIDs.prefix(kept)) + tokenizer.encode(tail)
    }
}
