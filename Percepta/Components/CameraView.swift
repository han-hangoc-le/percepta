import SwiftUI
import AVFoundation

// MARK: - Props Equivalent
struct CameraView: View {
    let onCapture: (Data) async -> Void
    let isCapturing: Bool
    
    @StateObject private var cameraModel = CameraModel()
    
    var body: some View {
        ZStack {
            switch cameraModel.authorizationStatus {
            case .notDetermined:
                Color.black
                    .overlay(ProgressView().tint(.white))
                    .onAppear {
                        Task { await cameraModel.requestPermission() }
                    }
                
            case .denied:
                PermissionDeniedView(onRequestPermission: {
                    Task { await cameraModel.requestPermission() }
                })
                
            case .authorized:
                ZStack {
                    CameraPreview(session: cameraModel.session)
                        .ignoresSafeArea()
                        .onAppear { cameraModel.startSession() }
                        .onDisappear { cameraModel.stopSession() }
                    
                    VStack {
                        Spacer()
                        ShutterButton(
                            disabled: isCapturing,
                            onTap: {
                                Task { await cameraModel.capturePhoto(onCapture: onCapture) }
                            }
                        )
                        .padding(.bottom, 40)
                    }
                }
                
            default:
                Color.black
            }
        }
        .background(Color.black)
    }
}

