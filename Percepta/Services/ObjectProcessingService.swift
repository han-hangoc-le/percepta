import UIKit

struct ObjectUploadRequest {
    let trackId: UUID
    let label: String
    let confidence: Float
    let boundingBox: CGRect
    let imageData: Data
}

struct ProcessedObjectResult {
    let trackId: UUID
    let image: UIImage
    let message: String?
}

struct LensFactResult {
    let trackId: UUID
    let equation: String
    let explanation: String
}

actor ObjectProcessingService {
    func process(request: ObjectUploadRequest, lensMode: String) async throws -> ProcessedObjectResult {
        let base64 = request.imageData.base64EncodedString()
        let payload = ObjectUploadPayload(
            clientObjectId: request.trackId.uuidString,
            lensMode: lensMode,
            label: request.label,
            confidence: request.confidence,
            boundingBox: BoundingBox(x: Double(request.boundingBox.origin.x),
                                     y: Double(request.boundingBox.origin.y),
                                     width: Double(request.boundingBox.width),
                                     height: Double(request.boundingBox.height)),
            imageBase64: base64
        )

        let response = try await APIService.shared.processObject(payload: payload)

        guard let imageData = Data(base64Encoded: response.annotatedImageBase64),
              let image = UIImage(data: imageData) else {
            throw ApiError(info: ApiErrorInfo(type: "invalid-response",
                                              message: "Unable to decode annotated image."))
        }

        return ProcessedObjectResult(trackId: request.trackId,
                                     image: image,
                                     message: response.message)
    }

    func fetchFacts(request: ObjectUploadRequest, lensMode: String) async throws -> LensFactResult {
        let base64 = request.imageData.base64EncodedString()

        let payload = ObjectUploadPayload(
            clientObjectId: request.trackId.uuidString,
            lensMode: lensMode,
            label: request.label,
            confidence: request.confidence,
            boundingBox: BoundingBox(x: Double(request.boundingBox.origin.x),
                                     y: Double(request.boundingBox.origin.y),
                                     width: Double(request.boundingBox.width),
                                     height: Double(request.boundingBox.height)),
            imageBase64: base64
        )

        let response = try await APIService.shared.fetchLensFacts(payload: payload)

        return LensFactResult(trackId: request.trackId,
                              equation: response.equation ?? "No equation generated.",
                              explanation: response.explanation ?? "")
    }
}
