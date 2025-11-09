import SwiftUI
import UIKit

enum SubmissionState: Equatable {
    case idle
    case uploading
    case awaitingServer
    case processed
    case failed(String)

    var isError: Bool {
        if case .failed = self { return true }
        return false
    }
}

struct DetectionOverlay: Identifiable {
    let id: UUID
    let rect: CGRect
    let label: String
    let confidence: Float
    let submissionState: SubmissionState
    let overlayImage: UIImage?
}

enum InfoOverlayStatus: Equatable {
    case idle
    case loading
    case loaded
    case failed(String)
}

struct InfoOverlay: Identifiable {
    let id: UUID
    let normalizedBoundingBox: CGRect
    var rect: CGRect
    let label: String
    let confidence: Float
    var contourPath: CGPath?
    var equation: String?
    var explanation: String?
    var status: InfoOverlayStatus = .idle
}

struct StatusBanner: Identifiable {
    let id = UUID()
    let message: String
    let tone: StatusTone
}
