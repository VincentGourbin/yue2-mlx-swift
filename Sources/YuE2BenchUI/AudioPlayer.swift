// AudioPlayer.swift - AVAudioPlayer wrapper for the last generated WAV (T-4.7, O10)
// Copyright 2026 Vincent Gourbin

import AVFoundation
import SwiftUI

/// Plays one WAV file at a time (the pipeline's most recent `audio.wav`). `@Observable` so
/// `AudioPlayerView` re-renders on play/pause/progress without a delegate-to-callback dance.
@Observable
@MainActor
final class AudioPlayerModel: NSObject, AVAudioPlayerDelegate {
    private(set) var isPlaying = false
    private(set) var duration: TimeInterval = 0
    private(set) var currentTime: TimeInterval = 0
    private(set) var url: URL?

    private var player: AVAudioPlayer?
    private var ticker: Timer?

    func load(url: URL) {
        stop()
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            self.player = player
            self.url = url
            duration = player.duration
            currentTime = 0
        } catch {
            self.player = nil
            self.url = nil
            duration = 0
        }
    }

    func togglePlayPause() {
        guard let player else { return }
        if player.isPlaying {
            player.pause()
            isPlaying = false
            ticker?.invalidate()
        } else {
            player.play()
            isPlaying = true
            startTicker()
        }
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
        currentTime = 0
        ticker?.invalidate()
        ticker = nil
    }

    func seek(to time: TimeInterval) {
        player?.currentTime = time
        currentTime = time
    }

    private func startTicker() {
        ticker?.invalidate()
        ticker = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let player = self.player else { return }
                self.currentTime = player.currentTime
            }
        }
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            self?.isPlaying = false
            self?.currentTime = 0
            self?.ticker?.invalidate()
        }
    }
}

/// Minimal transport: play/pause + a scrub slider, for the last run's `audio.wav`.
struct AudioPlayerView: View {
    @Bindable var model: AudioPlayerModel

    var body: some View {
        HStack(spacing: 12) {
            Button(action: model.togglePlayPause) {
                Image(systemName: model.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 28))
            }
            .buttonStyle(.plain)
            .disabled(model.url == nil)

            Slider(
                value: Binding(
                    get: { model.currentTime },
                    set: { model.seek(to: $0) }),
                in: 0...(max(model.duration, 0.01))
            )
            .disabled(model.url == nil)

            Text("\(format(model.currentTime)) / \(format(model.duration))")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private func format(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite else { return "0:00" }
        let s = Int(seconds)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
