import Foundation

// MARK: - Models
struct DetectionResponsePayload: Codable {
    var objects: [DetectedObject]?
    var lensMode: String?
    var message: String?
}

struct ObjectUploadPayload: Codable {
    var clientObjectId: String
    var lensMode: String
    var label: String
    var confidence: Float
    var boundingBox: BoundingBox
    var imageBase64: String
}

struct ObjectProcessingResponse: Codable {
    var clientObjectId: String
    var annotatedImageBase64: String
    var message: String?
}

struct DetectionResult {
    var objects: [DetectedObject]
    var lensMode: String
    var message: String?
    var usingMock: Bool
    var error: ApiErrorInfo?
    var rawResponse: DetectionResponsePayload
}

struct DetectedObject: Codable, Identifiable {
    var id: String
    var label: String
    var confidence: Double
    var boundingBox: BoundingBox
}

struct BoundingBox: Codable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
}

struct HealthResponse: Codable {
    var status: String
}

// MARK: - Error Info
struct ApiErrorInfo: Error {
    var type: String
    var message: String
    var status: Int?
    var code: String?
    var originalError: Error?
}

struct ApiError: Error {
    let info: ApiErrorInfo
}

// MARK: - API Service
final class APIService {
    static let shared = APIService()
    private init() {}

    // MARK: - Configuration
    private let defaultPort = "5000"
    private let defaultPath = "/api"
    private let defaultTimeout: TimeInterval = 25
    private let maxRetries = 0

    private var overrideBaseUrl: String? = "http://10.25.19.251:5050"

    private func buildUrl(from host: String) -> String {
        return "http://\(host):\(defaultPort)\(defaultPath)"
    }

    private func baseCandidates() -> [String] {
        var candidates: [String] = []
        var seen: Set<String> = []

        func append(_ value: String) {
            guard !seen.contains(value) else { return }
            candidates.append(value)
            seen.insert(value)
        }

        if let overrideBaseUrl {
            append(ensureApiPath(for: overrideBaseUrl))
        }
        append(buildUrl(from: "127.0.0.1"))
        append(buildUrl(from: "localhost"))
        append(buildUrl(from: "10.0.2.2")) // Android emulator style

        return candidates
    }

    private func ensureApiPath(for base: String) -> String {
        let trimmed = base.hasSuffix("/") ? String(base.dropLast()) : base
        if trimmed.lowercased().contains("\(defaultPath)/") || trimmed.lowercased().hasSuffix(defaultPath) {
            return trimmed
        }
        return "\(trimmed)\(defaultPath)"
    }

    private func request<T: Decodable>(_ endpoint: String,
                                       method: String = "GET",
                                       body: [String: Any]? = nil,
                                       retries: Int = 1,
                                       responseType: T.Type) async throws -> T {
        let candidates = baseCandidates()
        var lastError: Error?

        for base in candidates {
            guard let url = URL(string: "\(base)\(endpoint)") else { continue }
            var request = URLRequest(url: url)
            request.httpMethod = method
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = defaultTimeout
            if let body {
                request.httpBody = try? JSONSerialization.data(withJSONObject: body)
            }

            for attempt in 0...retries {
                do {
#if DEBUG
                    print("[API] \(method) \(url.absoluteString) attempt \(attempt + 1)")
#endif
                    let (data, response) = try await URLSession.shared.data(for: request)
                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw ApiError(info: ApiErrorInfo(type: "invalid-response",
                                                          message: "No HTTP response"))
                    }

                    if (200..<300).contains(httpResponse.statusCode) {
                        return try JSONDecoder().decode(T.self, from: data)
                    }

                    let errorMessage = APIService.extractServerMessage(from: data)
                    let type = (400..<500).contains(httpResponse.statusCode) ? "client" : "server"
                    throw ApiError(info: ApiErrorInfo(type: type,
                                                      message: errorMessage.isEmpty
                                                        ? "Server returned \(httpResponse.statusCode)"
                                                        : errorMessage,
                                                      status: httpResponse.statusCode))
                } catch {
#if DEBUG
                    print("[API] \(method) \(url.absoluteString) failed: \(error.localizedDescription)")
#endif
                    lastError = error
                    if attempt < retries { continue }
                }
            }
        }

        throw toApiError(lastError ?? NSError(domain: "network", code: -1))
    }

    // MARK: - Public API Calls
    func detectObjects(base64String: String, lensMode: String) async throws -> DetectionResult {
        do {
            let payload: DetectionResponsePayload = try await request(
                "/detect",
                method: "POST",
                body: [
                    "imageBase64": base64String,
                    "lensMode": lensMode
                ],
                retries: maxRetries,
                responseType: DetectionResponsePayload.self
            )

            guard let objects = payload.objects else {
                throw ApiError(info: ApiErrorInfo(type: "invalid-response",
                                                  message: "Missing objects array"))
            }

            return DetectionResult(
                objects: objects,
                lensMode: payload.lensMode ?? lensMode,
                message: payload.message,
                usingMock: false,
                error: nil,
                rawResponse: payload
            )
        } catch {
            let apiError = toApiError(error)
            switch apiError.info.type {
            case "network", "timeout", "server", "invalid-response":
                let mock = getMockDetectionResponse(lensMode: lensMode)
                return DetectionResult(
                    objects: mock.objects,
                    lensMode: mock.lensMode,
                    message: mock.message,
                    usingMock: true,
                    error: apiError.info,
                    rawResponse: mock.rawResponse
                )
            default:
                throw apiError
            }
        }
    }

    func processObject(payload: ObjectUploadPayload) async throws -> ObjectProcessingResponse {
        do {
            return try await request(
                "/objects",
                method: "POST",
                body: [
                    "clientObjectId": payload.clientObjectId,
                    "lensMode": payload.lensMode,
                    "label": payload.label,
                    "confidence": payload.confidence,
                    "boundingBox": [
                        "x": payload.boundingBox.x,
                        "y": payload.boundingBox.y,
                        "width": payload.boundingBox.width,
                        "height": payload.boundingBox.height
                    ],
                    "imageBase64": payload.imageBase64
                ],
                retries: maxRetries,
                responseType: ObjectProcessingResponse.self
            )
        } catch {
            throw toApiError(error)
        }
    }

    func checkHealth() async throws -> HealthResponse {
        do {
            return try await request("/health", method: "GET", retries: 0, responseType: HealthResponse.self)
        } catch {
            throw toApiError(error)
        }
    }

    // MARK: - Helpers
    func getMockDetectionResponse(lensMode: String)
        -> (objects: [DetectedObject], lensMode: String, message: String, rawResponse: DetectionResponsePayload) {
        let objects = [
            DetectedObject(id: "mock-1", label: "Notebook", confidence: 0.92,
                           boundingBox: BoundingBox(x: 0.18, y: 0.32, width: 0.28, height: 0.24)),
            DetectedObject(id: "mock-2", label: "Coffee Mug", confidence: 0.87,
                           boundingBox: BoundingBox(x: 0.55, y: 0.38, width: 0.22, height: 0.3))
        ]

        let payload = DetectionResponsePayload(objects: objects,
                                               lensMode: lensMode,
                                               message: "Mock response generated on-device.")

        return (objects, lensMode, "Mock data for \(lensMode) lens.", payload)
    }

    private func toApiError(_ error: Error) -> ApiError {
        if let apiError = error as? ApiError {
            return apiError
        }

        if let urlError = error as? URLError {
            var type = "unknown"
            if urlError.code == .timedOut {
                type = "timeout"
            } else if urlError.code == .notConnectedToInternet {
                type = "network"
            }
            return ApiError(info: ApiErrorInfo(type: type,
                                               message: urlError.localizedDescription,
                                               code: "\(urlError.code.rawValue)",
                                               originalError: error))
        }

        return ApiError(info: ApiErrorInfo(type: "unknown",
                                           message: error.localizedDescription,
                                           originalError: error))
    }

    private static func extractServerMessage(from data: Data) -> String {
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let details = json["details"] as? String, !details.isEmpty {
                return details
            }
            if let error = json["error"] as? String, !error.isEmpty {
                return error
            }
            if let message = json["message"] as? String, !message.isEmpty {
                return message
            }
        }
        return String(data: data, encoding: .utf8) ?? ""
    }

    func mapErrorToMessage(_ error: Error) -> String {
        let info: ApiErrorInfo
        if let apiError = error as? ApiError {
            info = apiError.info
        } else {
            info = ApiErrorInfo(type: "unknown", message: error.localizedDescription)
        }

        switch info.type {
        case "network":
            let detail = info.message.isEmpty ? "" : " (\(info.message))"
            return "Can't connect to server\(detail)."
        case "timeout":
            let detail = info.message.isEmpty ? "" : " (\(info.message))"
            return "Request took too long\(detail). Please try again."
        case "server":
            return info.message.isEmpty ? "Server is having issues. Please try again later." : info.message
        case "invalid-response":
            return info.message.isEmpty ? "Received an unexpected response from the server." : info.message
        case "client":
            return info.message
        default:
            return info.message.isEmpty ? "Something went wrong. Please try again." : info.message
        }
    }

    func debugDescription(for error: Error) -> String {
        if let apiError = error as? ApiError {
            let info = apiError.info
            var parts: [String] = []
            parts.append("[\(info.type)]")
            if let status = info.status {
                parts.append("status \(status)")
            }
            if let code = info.code {
                parts.append("code \(code)")
            }
            if !info.message.isEmpty {
                parts.append(info.message)
            }
            return parts.joined(separator: " ")
        }
        return error.localizedDescription
    }
}
