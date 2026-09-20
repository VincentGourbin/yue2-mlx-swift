// YuE2BenchUIApp.swift - TabView: Génération (+ Mesures panel) / Historique (T-4.7, O10)
// Copyright 2026 Vincent Gourbin

import AppKit
import SwiftUI
import YuE2Core

/// A SwiftPM executable target is not an `.app` bundle, so macOS gives it no activation
/// policy by default — the window is created but stays hidden. Force "regular app" and
/// activate at launch (same fix as `gemma-4-bench-ui`/`qwen38-bench-ui`).
final class BenchAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        for window in NSApp.windows {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct YuE2BenchUIApp: App {
    @NSApplicationDelegateAdaptor(BenchAppDelegate.self) var appDelegate
    @State private var vm = GenerationViewModel()

    var body: some Scene {
        WindowGroup {
            TabView {
                GenerationView(vm: vm)
                    .tabItem { Label("Génération", systemImage: "waveform") }
                HistoryView(entries: vm.history)
                    .tabItem { Label("Historique", systemImage: "clock.arrow.circlepath") }
            }
            .frame(minWidth: 900, minHeight: 640)
        }
    }
}
