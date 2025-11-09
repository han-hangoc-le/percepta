import CoreML
import Vision
import UIKit

struct YOLOPrediction: Identifiable {
    let id: UUID
    let identifier: String
    let confidence: Float
    /// Normalized bounding box in Vision space (origin at lower-left).
    let boundingBox: CGRect

    init(id: UUID = UUID(), identifier: String, confidence: Float, boundingBox: CGRect) {
        self.id = id
        self.identifier = identifier
        self.confidence = confidence
        self.boundingBox = boundingBox
    }

    func withBoundingBox(_ boundingBox: CGRect) -> YOLOPrediction {
        YOLOPrediction(id: id, identifier: identifier, confidence: confidence, boundingBox: boundingBox)
    }
}

enum YOLODetectorError: Error {
    case modelNotFound(String)
    case invalidResults
}

final class YOLODetector {
    private let request: VNCoreMLRequest
    private let requestQueue = DispatchQueue(label: "com.percepta.yolo.detector", qos: .userInitiated)

    init(modelFileName: String = "YOLOv8n") throws {
        guard let modelURL = YOLODetector.locateModel(named: modelFileName) else {
            throw YOLODetectorError.modelNotFound(modelFileName)
        }
        let compiledModel = try MLModel(contentsOf: modelURL)
        let visionModel = try VNCoreMLModel(for: compiledModel)
        self.request = VNCoreMLRequest(model: visionModel)
        self.request.imageCropAndScaleOption = .scaleFill
    }

    func detect(on pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation) async throws -> [YOLOPrediction] {
        try await withCheckedThrowingContinuation { continuation in
            requestQueue.async {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                    orientation: orientation,
                                                    options: [:])
                do {
                    try handler.perform([self.request])
                    guard let results = self.request.results as? [VNRecognizedObjectObservation] else {
                        throw YOLODetectorError.invalidResults
                    }

                    let predictions: [YOLOPrediction] = results.compactMap { observation in
                        guard let best = observation.labels.first else { return nil }
                        return YOLOPrediction(identifier: best.identifier,
                                              confidence: best.confidence,
                                              boundingBox: observation.boundingBox)
                    }
                    continuation.resume(returning: predictions)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func locateModel(named name: String) -> URL? {
        let bundle = Bundle.main
        if let url = bundle.url(forResource: name, withExtension: "mlmodelc") {
            return url
        }

        let variants = [name.lowercased(), name.uppercased(), name.capitalized]
        for variant in variants where variant != name {
            if let url = bundle.url(forResource: variant, withExtension: "mlmodelc") {
                return url
            }
        }

        return bundle.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)?.first
    }
}
