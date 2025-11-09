import SwiftUI
import UIKit

struct DetectionOverlayView: View {
    let overlays: [DetectionOverlay]

    var body: some View {
        ZStack {
            ForEach(overlays) { overlay in
                let rect = overlay.rect
                ZStack {
                    if let uiImage = overlay.overlayImage {
                        Image(uiImage: uiImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: rect.width, height: rect.height)
                            .clipped()
                    }

                    RoundedRectangle(cornerRadius: 8)
                        .stroke(borderColor(for: overlay.submissionState), lineWidth: 2)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(backgroundFill(for: overlay))
                        )
                }
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
                .overlay(alignment: .topLeading) {
                    infoCard(for: overlay)
                        .offset(x: 0, y: -rect.height / 2 - 12)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func backgroundFill(for overlay: DetectionOverlay) -> Color {
        if overlay.overlayImage != nil {
            return Color.black.opacity(0.001)
        }
        return borderColor(for: overlay.submissionState).opacity(0.15)
    }

    private func infoCard(for overlay: DetectionOverlay) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(overlay.label) \(String(format: "%.0f%%", overlay.confidence * 100))")
                .font(.caption)
                .bold()
            Text(stateLabel(for: overlay.submissionState))
                .font(.caption2)
                .foregroundColor(.white.opacity(0.7))
        }
        .padding(6)
        .background(.black.opacity(0.6))
        .cornerRadius(6)
    }

    private func borderColor(for state: SubmissionState) -> Color {
        switch state {
        case .processed:
            return .green
        case .failed:
            return .red
        case .uploading, .awaitingServer:
            return .orange
        case .idle:
            return .cyan
        }
    }

    private func stateLabel(for state: SubmissionState) -> String {
        switch state {
        case .idle:
            return "Pending"
        case .uploading:
            return "Uploading"
        case .awaitingServer:
            return "Waiting"
        case .processed:
            return "Anchored"
        case .failed(let message):
            return "Failed: \(message)"
        }
    }
}
