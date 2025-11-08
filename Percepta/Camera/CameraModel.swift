import AVFoundation
import UIKit
import Combine

@MainActor
final class CameraModel: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    @Published var authorizationStatus: AVAuthorizationStatus = .notDetermined
    @Published var session = AVCaptureSession()
    
    private var photoOutput = AVCapturePhotoOutput()
    private var captureCompletion: ((Data) -> Void)?
    
    override init() {
        super.init()
        authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
    }
    
    func requestPermission() async {
        let current = AVCaptureDevice.authorizationStatus(for: .video)
        
        if current == .notDetermined {
            // 🔹 This line triggers the iOS permission popup
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            await MainActor.run {
                self.authorizationStatus = granted ? .authorized : .denied
            }
        } else {
            // 🔹 Update status if permission was already decided
            await MainActor.run {
                self.authorizationStatus = current
            }
        }
    }
    
    func startSession() {
        guard authorizationStatus == .authorized else { return }
        configureSession()
        session.startRunning()
    }
    
    func stopSession() {
        session.stopRunning()
    }
    
    private func configureSession() {
        session.beginConfiguration()
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input)
        else { return }
        
        session.addInput(input)
        
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
        }
        
        session.commitConfiguration()
    }
    
    func capturePhoto(onCapture: @escaping (Data) async -> Void) {
        captureCompletion = { data in
            Task { await onCapture(data) }
        }
        let settings = AVCapturePhotoSettings()
        settings.isHighResolutionPhotoEnabled = false
        settings.isAutoStillImageStabilizationEnabled = true
        settings.isAutoVirtualDeviceFusionEnabled = true
        settings.photoQualityPrioritization = .speed
        settings.isAutoRedEyeReductionEnabled = true
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    // MARK: - Delegate
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let error = error {
            print("Capture failed: \(error)")
            return
        }
        if let data = photo.fileDataRepresentation() {
            captureCompletion?(data)
        }
    }
}
