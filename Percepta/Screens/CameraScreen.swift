import SwiftUI
import AVFoundation

struct CameraScreen: View {
    @Environment(\.dismiss) var dismiss
    @State private var isCapturing: Bool = false
    @State private var statusMessage: String? = nil
    @State private var statusTone: StatusTone = .info

    var lensMode: String

    var body: some View {
        ZStack {
            // ✅ Replaces the placeholder with full CameraView
            CameraView(onCapture: handleCapture, isCapturing: isCapturing)
                .ignoresSafeArea()
            
            VStack {
                // 🔹 Top bar (Back button)
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
                }
                .padding(.top, 24)
                .padding(.horizontal, 16)

                Spacer()
            }
            
            // 🔹 Status Banner
            if let message = statusMessage {
                VStack {
                    HStack {
                        Text(message)
                            .foregroundColor(.white)
                            .font(.system(size: 14))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                    }
                    .frame(maxWidth: .infinity)
                    .background(statusTone == .error
                                ? Color.red.opacity(0.9)
                                : Color.blue.opacity(0.85))
                    .cornerRadius(12)
                    .padding(.horizontal, 16)
                    .padding(.top, 72)
                    Spacer()
                }
                .animation(.easeInOut, value: statusMessage)
            }
            
            // 🔹 Loading overlay when analyzing frame
            if isCapturing {
                ZStack {
                    Color.black.opacity(0.6).ignoresSafeArea()
                    VStack {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.5)
                        Text("Analyzing frame...")
                            .font(.system(size: 16))
                            .foregroundColor(Color(hex: "#f8fafc"))
                            .padding(.top, 12)
                    }
                }
                .transition(.opacity)
            }
        }
        .background(Color.black)
        .navigationBarHidden(true)
    }

    // MARK: - Logic
    func handleCapture(imageData: Data) async {
        isCapturing = true
        statusTone = .info
        statusMessage = "Analyzing frame..."
        
        do {
            // Convert to Base64 just like in React Native
            let base64String = imageData.base64EncodedString()
            
            // Call your Flask API
            let result = try await APIService.shared.detectObjects(
                base64String: base64String,
                lensMode: lensMode
            )
            
            if result.usingMock, let error = result.error {
                statusTone = .error
                statusMessage = APIService.shared.mapErrorToMessage(ApiError(info: error))

            } else {
                statusTone = .info
                statusMessage = result.message ?? "Frame processed successfully by the backend."
            }
        } catch {
            statusTone = .error
            statusMessage = APIService.shared.mapErrorToMessage(error)
        }
        
        isCapturing = false
    }
}

// MARK: - Supporting Enum
enum StatusTone {
    case info
    case error
}
