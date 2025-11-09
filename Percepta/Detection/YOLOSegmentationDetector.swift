import Foundation
import CoreML
import Vision
import CoreGraphics

struct YOLOSegmentationPrediction: Identifiable {
    let id: UUID
    let label: String
    let confidence: Float
    let boundingBox: CGRect
    let contourPath: CGPath?
}

final class YOLOSegmentationDetector {
    private struct RawDetection {
        let id: UUID
        let classIndex: Int
        let confidence: Float
        let boundingBox: CGRect
        let maskCoefficients: [Float]
    }

    private let request: VNCoreMLRequest
    private let requestQueue = DispatchQueue(label: "com.percepta.yolo.segmentation", qos: .userInitiated)
    private let labels: [String]

    private let inputWidth: Float = 640
    private let inputHeight: Float = 640
    private let maskDimensions = 32
    private let maskThreshold: Float = 0.5
    private let confidenceThreshold: Float = 0.35
    private let iouThreshold: Float = 0.45
    private let maxDetections = 12

    init(modelFileName: String = "yolov8n-seg") throws {
        guard let modelURL = Bundle.main.url(forResource: modelFileName, withExtension: "mlmodelc") else {
            throw YOLODetectorError.modelNotFound(modelFileName)
        }

        let coreModel = try MLModel(contentsOf: modelURL)
        labels = YOLOSegmentationDetector.loadLabels(from: coreModel)
        let visionModel = try VNCoreMLModel(for: coreModel)
        request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill
    }

    func detect(on pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation) async throws -> [YOLOSegmentationPrediction] {
        try await withCheckedThrowingContinuation { continuation in
            requestQueue.async {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                    orientation: orientation,
                                                    options: [:])
                do {
                    try handler.perform([self.request])
                    let observations = self.request.results?.compactMap { $0 as? VNCoreMLFeatureValueObservation } ?? []
                    let featureMap = Dictionary(uniqueKeysWithValues: observations.map { ($0.featureName, $0.featureValue) })

                    guard let rawDetections = featureMap["var_1053"]?.multiArrayValue,
                          let proto = featureMap["p"]?.multiArrayValue else {
                        throw YOLODetectorError.invalidResults
                    }

                    let predictions = self.decodeDetections(detections: rawDetections, proto: proto)
                    continuation.resume(returning: predictions)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func decodeDetections(detections: MLMultiArray,
                                  proto: MLMultiArray) -> [YOLOSegmentationPrediction] {
        guard detections.shape.count == 3 else { return [] }
        let channels = detections.shape[1].intValue
        let anchors = detections.shape[2].intValue
        let classCount = min(labels.count, max(0, channels - 4 - maskDimensions))

        let data = detections.dataPointer.bindMemory(to: Float32.self, capacity: detections.count)
        let channelStride = anchors

        var rawDetections: [RawDetection] = []
        for anchor in 0..<anchors {
            let x = data[anchor + 0 * channelStride]
            let y = data[anchor + 1 * channelStride]
            let width = data[anchor + 2 * channelStride]
            let height = data[anchor + 3 * channelStride]

            var bestClass = 0
            var bestScore: Float = 0
            for classIndex in 0..<classCount {
                let score = sigmoid(data[anchor + (4 + classIndex) * channelStride])
                if score > bestScore {
                    bestScore = score
                    bestClass = classIndex
                }
            }

            guard bestScore >= confidenceThreshold else { continue }

            var coefficients: [Float] = []
            let maskOffset = 4 + classCount
            coefficients.reserveCapacity(maskDimensions)
            for maskIndex in 0..<maskDimensions {
                coefficients.append(data[anchor + (maskOffset + maskIndex) * channelStride])
            }

            let rect = CGRect(
                x: CGFloat((x - width / 2) / inputWidth),
                y: CGFloat((y - height / 2) / inputHeight),
                width: CGFloat(width / inputWidth),
                height: CGFloat(height / inputHeight)
            ).clampedToUnitSquare()

            guard rect.width > 0, rect.height > 0 else { continue }

            let detection = RawDetection(id: UUID(),
                                         classIndex: bestClass,
                                         confidence: bestScore,
                                         boundingBox: rect,
                                         maskCoefficients: coefficients)
            rawDetections.append(detection)
        }

        let topDetections = performNMS(on: rawDetections).prefix(maxDetections)
        return buildPredictions(from: Array(topDetections), proto: proto)
    }

    private func buildPredictions(from detections: [RawDetection],
                                  proto: MLMultiArray) -> [YOLOSegmentationPrediction] {
        let protoPointer = proto.dataPointer.bindMemory(to: Float32.self, capacity: proto.count)
        let protoHeight = proto.shape[2].intValue
        let protoWidth = proto.shape[3].intValue
        let protoArea = protoHeight * protoWidth

        return detections.map { detection in
            let contour = contourPath(for: detection,
                                      protoPointer: protoPointer,
                                      protoWidth: protoWidth,
                                      protoHeight: protoHeight,
                                      protoArea: protoArea)

            return YOLOSegmentationPrediction(id: detection.id,
                                              label: labels[safe: detection.classIndex] ?? "object",
                                              confidence: detection.confidence,
                                              boundingBox: detection.boundingBox,
                                              contourPath: contour)
        }
    }

    private func contourPath(for detection: RawDetection,
                             protoPointer: UnsafePointer<Float32>,
                             protoWidth: Int,
                             protoHeight: Int,
                             protoArea: Int) -> CGPath? {
        let minX = max(0, Int(Float(detection.boundingBox.minX) * Float(protoWidth)))
        let maxX = min(protoWidth, Int(ceil(Float(detection.boundingBox.maxX) * Float(protoWidth))))
        let minY = max(0, Int(Float(detection.boundingBox.minY) * Float(protoHeight)))
        let maxY = min(protoHeight, Int(ceil(Float(detection.boundingBox.maxY) * Float(protoHeight))))

        let width = max(1, maxX - minX)
        let height = max(1, maxY - minY)
        var maskBuffer = [UInt8](repeating: 0, count: width * height)

        for y in 0..<height {
            for x in 0..<width {
                let protoX = minX + x
                let protoY = minY + y
                let protoIndex = protoY * protoWidth + protoX

                var value: Float = 0
                for maskIndex in 0..<maskDimensions {
                    let coefficient = detection.maskCoefficients[maskIndex]
                    let protoValue = protoPointer[maskIndex * protoArea + protoIndex]
                    value += coefficient * protoValue
                }

                let normalized = sigmoid(value)
                maskBuffer[y * width + x] = normalized > maskThreshold ? 255 : 0
            }
        }

        guard let maskImage = cgImage(from: maskBuffer, width: width, height: height) else {
            return nil
        }

        let request = VNDetectContoursRequest()
        request.revision = 1 // use earliest revision available across iOS versions
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = false
        request.maximumImageDimension = max(width, height)

        let handler = VNImageRequestHandler(cgImage: maskImage, options: [:])
        do {
            try handler.perform([request])
            guard let observation = request.results?.first else { return nil }
            return observation.normalizedPath
        } catch {
            return nil
        }
    }

    private func performNMS(on detections: [RawDetection]) -> [RawDetection] {
        var remaining = detections.sorted { $0.confidence > $1.confidence }
        var keep: [RawDetection] = []

        while let best = remaining.first {
            keep.append(best)
            remaining.removeFirst()
            remaining.removeAll { candidate in
                best.boundingBox.intersectionOverUnion(with: candidate.boundingBox) >= CGFloat(iouThreshold)
            }
        }

        return keep
    }

    private func cgImage(from buffer: [UInt8], width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(buffer) as CFData) else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        return CGImage(width: width,
                       height: height,
                       bitsPerComponent: 8,
                       bitsPerPixel: 8,
                       bytesPerRow: width,
                       space: colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: provider,
                       decode: nil,
                       shouldInterpolate: false,
                       intent: .defaultIntent)
    }

    private func sigmoid(_ value: Float32) -> Float32 {
        let exponent = expf(-value)
        return 1 / (1 + exponent)
    }

    private static func loadLabels(from model: MLModel) -> [String] {
        if let userDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
           let namesString = userDefined["names"] {
            let jsonReady = namesString
                .replacingOccurrences(of: "'", with: "\"")
                .replacingOccurrences(of: " ", with: "")
            if let data = jsonReady.data(using: .utf8),
               let dictionary = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
                let sorted = dictionary.sorted { (lhs, rhs) -> Bool in
                    let leftKey = Int(lhs.key) ?? 0
                    let rightKey = Int(rhs.key) ?? 0
                    return leftKey < rightKey
                }
                return sorted.map { $0.value }
            }
        }

        return YOLOSegmentationDetector.defaultClassNames
    }

    private static let defaultClassNames: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}
