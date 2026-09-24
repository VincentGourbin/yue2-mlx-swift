// ModelDownloader.swift - resumable HuggingFace downloads for YuE2 checkpoints
// Copyright 2026 Vincent Gourbin

import Foundation
import CryptoKit

/// Snapshot of one file's download progress within a `ModelDownloader.download` call.
public struct DownloadProgress: Sendable, Equatable {
    public let file: String
    public let fileIndex: Int
    public let fileCount: Int
    public let writtenBytes: Int64
    public let totalBytes: Int64
}

/// Downloads a `YuE2Model`'s files from HuggingFace and verifies them against
/// `weights_manifest.json`. Each file streams to a `<name>.part` sibling with HTTP
/// `Range` resume, then is atomically renamed into place once complete.
public actor ModelDownloader {
    private let modelsDir: URL
    private let token: String?

    public init(modelsDir: URL, token: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]) {
        self.modelsDir = modelsDir
        self.token = token
    }

    /// Downloads every file of `model` that is not already present, skipping files
    /// that already exist at their final destination.
    public func download(_ model: YuE2Model, progress: @Sendable @escaping (DownloadProgress) -> Void = { _ in }) async throws {
        let dir = modelsDir.appendingPathComponent(model.directoryName)
        let files = model.files
        for (index, file) in files.enumerated() {
            let destination = dir.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destination.path) { continue }
            try await downloadOne(repoID: model.repoID, remotePath: file, to: destination) { written, total in
                progress(DownloadProgress(file: file, fileIndex: index, fileCount: files.count, writtenBytes: written, totalBytes: total))
            }
        }
    }

    /// Existence of every expected file, plus `"model.safetensors.sha256"` when
    /// `weights_manifest.json` and the weights file are both present.
    public func verify(_ model: YuE2Model) throws -> [String: Bool] {
        let dir = modelsDir.appendingPathComponent(model.directoryName)
        var result: [String: Bool] = [:]
        for file in model.files {
            result[file] = FileManager.default.fileExists(atPath: dir.appendingPathComponent(file).path)
        }
        let weightsURL = dir.appendingPathComponent("model.safetensors")
        if let expected = try? Self.expectedSHA256(manifestURL: dir.appendingPathComponent("weights_manifest.json"), file: "model.safetensors"),
           FileManager.default.fileExists(atPath: weightsURL.path) {
            result["model.safetensors.sha256"] = try Self.sha256Hex(of: weightsURL).caseInsensitiveCompare(expected) == .orderedSame
        }
        return result
    }

    private func downloadOne(
        repoID: String, remotePath: String, to destination: URL,
        progress: @Sendable @escaping (Int64, Int64) -> Void
    ) async throws {
        let partURL = destination.appendingPathExtension("part")
        guard let url = Self.remoteURL(repoID: repoID, remotePath: remotePath) else {
            throw YuE2Error.invalidRequest("bad URL for \(repoID)/\(remotePath)")
        }
        var request = URLRequest(url: url)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let resumeOffset = (try? FileManager.default.attributesOfItem(atPath: partURL.path)[.size] as? Int64) ?? nil ?? 0
        if resumeOffset > 0 { request.setValue("bytes=\(resumeOffset)-", forHTTPHeaderField: "Range") }

        let delegate = ChunkedDownloadDelegate(partURL: partURL, resumeOffset: resumeOffset, progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            delegate.completion = { continuation.resume(with: $0) }
            session.dataTask(with: request).resume()
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: partURL, to: destination)
    }

    /// The exact `resolve/main` URL a file downloads from — exposed for testing without
    /// making a network request.
    public static func remoteURL(repoID: String, remotePath: String) -> URL? {
        guard let escaped = remotePath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else { return nil }
        return URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(escaped)")
    }

    private static func expectedSHA256(manifestURL: URL, file: String) throws -> String {
        let data = try Data(contentsOf: manifestURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let files = json["files"] as? [String: Any],
              let entry = files[file] as? [String: Any],
              let sha = entry["sha256"] as? String else {
            throw YuE2Error.invalidRequest("malformed weights_manifest.json")
        }
        return sha
    }

    private static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// Bridges `URLSessionDataDelegate` chunk callbacks to incremental writes on `partURL`.
/// Writing every chunk as it arrives (rather than waiting for a single completed temp
/// file, as `URLSessionDownloadTask` does) means a dropped connection leaves a resumable
/// `.part` file instead of losing all progress on a multi-gigabyte checkpoint.
private final class ChunkedDownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let partURL: URL
    private var handle: FileHandle?
    private var written: Int64
    private var total: Int64 = 0
    private let progress: @Sendable (Int64, Int64) -> Void
    var completion: ((Result<Void, Error>) -> Void)?

    init(partURL: URL, resumeOffset: Int64, progress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.partURL = partURL
        self.written = resumeOffset
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            completion?(.failure(YuE2Error.missingFile("HTTP \(status) for \(partURL.deletingPathExtension().lastPathComponent)")))
            completionHandler(.cancel)
            return
        }
        let resumed = http.statusCode == 206
        if !resumed {
            written = 0
            if FileManager.default.fileExists(atPath: partURL.path) {
                try? FileManager.default.removeItem(at: partURL)
            }
        }
        total = http.value(forHTTPHeaderField: "Content-Length").flatMap(Int64.init).map { $0 + written } ?? 0
        if !FileManager.default.fileExists(atPath: partURL.path) {
            FileManager.default.createFile(atPath: partURL.path, contents: nil)
        }
        handle = try? FileHandle(forWritingTo: partURL)
        if resumed { _ = try? handle?.seekToEnd() }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let handle else { return }
        try? handle.write(contentsOf: data)
        written += Int64(data.count)
        progress(written, total)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle?.close()
        if let error {
            completion?(.failure(error))
        } else {
            completion?(.success(()))
        }
    }
}
