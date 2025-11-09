import SwiftUI
import AVFoundation

struct CameraScreen: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var detectionCoordinator: DetectionCoordinator
    @State private var viewSize: CGSize = .zero

    let lensMode: String

    init(lensMode: String) {
        self.lensMode = lensMode
        _detectionCoordinator = StateObject(wrappedValue: DetectionCoordinator(lensMode: lensMode))
    }

    var body: some View {
        ZStack {
            GeometryReader { proxy in
                ZStack {
                    ARCameraView(coordinator: detectionCoordinator)
                        .ignoresSafeArea()
                        .onAppear {
                            viewSize = proxy.size
                            detectionCoordinator.updateViewportSize(proxy.size)
                            detectionCoordinator.setActive(true)
                        }
                        .onDisappear {
                            detectionCoordinator.setActive(false)
                        }

                    if detectionCoordinator.usesInfoPipeline {
                        InfoOverlayView(overlays: detectionCoordinator.infoOverlays,
                                        accentColor: detectionCoordinator.infoAccentColor)
                            .frame(width: proxy.size.width, height: proxy.size.height)
                    } else {
                        DetectionOverlayView(overlays: detectionCoordinator.overlays)
                            .frame(width: proxy.size.width, height: proxy.size.height)
                    }
                }
                .onChange(of: proxy.size) { newSize in
                    viewSize = newSize
                    detectionCoordinator.updateViewportSize(newSize)
                }
            }

            VStack {
                HStack {
                    Button(action: { dismiss() }) {
                        Text("‹ Back")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundColor(Color(hex: "#f8fafc"))
                            .padding(.vertical, 8)
                            .padding(.horizontal, 16)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(999)
                    }
                    Spacer()
                    Text(lensMode.capitalized)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.white)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(12)
                }
                .padding(.top, 24)
                .padding(.horizontal, 16)

                Spacer()
            }

            if let banner = detectionCoordinator.statusBanner {
                VStack {
                    HStack {
                        Text(banner.message)
                            .foregroundColor(.white)
                            .font(.system(size: 14))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                    }
                    .frame(maxWidth: .infinity)
                    .background(banner.tone == .error ? Color.red.opacity(0.9) : Color.blue.opacity(0.85))
                    .cornerRadius(12)
                    .padding(.horizontal, 16)
                    .padding(.top, 72)
                    Spacer()
                }
                .transition(.opacity)
            }
        }
        .background(Color.black)
        .navigationBarHidden(true)
    }
}

enum StatusTone {
    case info
    case error
}
