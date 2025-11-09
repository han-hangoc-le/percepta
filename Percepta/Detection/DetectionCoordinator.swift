import Foundation
import ARKit
import RealityKit
import SwiftUI
import Combine
import UIKit
import simd

private enum LensPipeline {
    case hologram
    case infoOverlay
}

@MainActor
final class DetectionCoordinator: NSObject, ObservableObject, ARSessionDelegate {
    @Published private(set) var overlays: [DetectionOverlay] = []
    @Published private(set) var infoOverlays: [InfoOverlay] = []
    @Published var statusBanner: StatusBanner?

    let lensMode: String
    var usesInfoPipeline: Bool { pipeline == .infoOverlay }
    var infoAccentColor: Color { infoAccentColorValue }

    private let pipeline: LensPipeline
    private let detector: YOLODetector
    private let segmentationDetector: YOLOSegmentationDetector?
    private let tracker = ObjectTracker()
    private let processingService = ObjectProcessingService()
    private let infoAccentColorValue: Color

    private weak var arView: ARView?
    private var viewportSize: CGSize = .zero
    private var anchorMap: [UUID: AnchorEntity] = [:]
    private var objectStates: [UUID: ObjectState] = [:]

    private var detectionTask: Task<Void, Never>?
    private var latestSnapshot: FrameSnapshot?
    private var sessionConfigured = false
    private var isActive = false
    private let selectionPaddingFraction: CGFloat = 0.15
    private let tapInfluenceFraction: CGFloat = 0.2
    private lazy var worldConfiguration: ARWorldTrackingConfiguration = {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .automatic
        configuration.sceneReconstruction = .mesh
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        return configuration
    }()

    init(lensMode: String, modelFileName: String = "YOLOv8n") {
        self.lensMode = lensMode
        self.pipeline = DetectionCoordinator.pipeline(for: lensMode)
        self.infoAccentColorValue = DetectionCoordinator.accentColor(for: lensMode)

        do {
            detector = try YOLODetector(modelFileName: modelFileName)
        } catch {
            fatalError("Failed to load YOLO model: \(error)")
        }

        if pipeline == .infoOverlay {
            do {
                segmentationDetector = try YOLOSegmentationDetector()
            } catch {
                print("⚠️ Failed to load segmentation model: \(error.localizedDescription)")
                segmentationDetector = nil
            }
        } else {
            segmentationDetector = nil
        }
    }

    deinit {
        detectionTask?.cancel()
    }

    func attach(arView: ARView) {
        self.arView = arView
        if !sessionConfigured {
            configureSession(for: arView, reset: true)
            sessionConfigured = true
        }
    }

    func updateViewportSize(_ size: CGSize) {
        viewportSize = size
        refreshInfoOverlayLayouts()
    }

    func setActive(_ active: Bool) {
        isActive = active
        guard let arView else { return }

        if active {
            if sessionConfigured {
                arView.session.run(worldConfiguration, options: [])
            } else {
                configureSession(for: arView, reset: true)
                sessionConfigured = true
            }
        } else {
            detectionTask?.cancel()
            arView.session.pause()
            latestSnapshot = nil
            infoOverlays.removeAll()
        }
    }

    private func configureSession(for arView: ARView, reset: Bool) {
        arView.automaticallyConfigureSession = false
        let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
        arView.session.run(worldConfiguration, options: options)
    }

    private func runAnchoredPipeline(snapshot: FrameSnapshot, normalizedPoint: CGPoint?) {
        statusBanner = StatusBanner(message: "Analyzing selection...", tone: .info)
        let detector = detector

        detectionTask?.cancel()
        detectionTask = Task(priority: .userInitiated) { [weak self, snapshot, normalizedPoint] in
            guard let self = self else { return }
            do {
                let predictions = try await detector.detect(on: snapshot.pixelBuffer,
                                                            orientation: snapshot.orientation)

                guard let selection = DetectionCoordinator.selectPrediction(near: normalizedPoint,
                                                                            from: predictions) else {
                    await self.publishError("Couldn't find an object at that location.")
                    return
                }

                let adjustedPrediction = selection.withBoundingBox(
                    await self.selectionBoundingBox(from: selection.boundingBox, tapPoint: normalizedPoint)
                )

                await MainActor.run {
                    self.tracker.reset()
                }

                let uploadRequests = await self.prepareFrameUpdate(predictions: [adjustedPrediction],
                                                                   snapshot: snapshot)

                if uploadRequests.isEmpty {
                    await self.publishError("Failed to prepare the selection for upload.")
                    return
                }

                await MainActor.run {
                    self.statusBanner = StatusBanner(message: "Uploading selection...", tone: .info)
                }

                for request in uploadRequests {
                    await self.upload(request: request)
                }
            } catch is CancellationError {
                return
            } catch {
                await self.publishError("Detection failed: \(error.localizedDescription)")
            }
        }
    }

    private func runInfoPipeline(snapshot: FrameSnapshot, normalizedPoint: CGPoint?) {
        statusBanner = StatusBanner(message: "Scanning objects...", tone: .info)

        detectionTask?.cancel()
        detectionTask = Task(priority: .userInitiated) { [weak self, snapshot, normalizedPoint] in
            guard let self = self else { return }
            guard let segmentationDetector = self.segmentationDetector else {
                await self.publishError("Segmentation model unavailable.")
                return
            }

            do {
                let predictions = try await segmentationDetector.detect(on: snapshot.pixelBuffer,
                                                                        orientation: snapshot.orientation)
                guard !predictions.isEmpty else {
                    await self.publishError("No objects detected in this frame.")
                    return
                }

                let lightweight = predictions.map {
                    YOLOPrediction(id: $0.id,
                                   identifier: $0.label,
                                   confidence: $0.confidence,
                                   boundingBox: $0.boundingBox)
                }

                guard let selection = DetectionCoordinator.selectPrediction(near: normalizedPoint,
                                                                            from: lightweight),
                      let chosen = predictions.first(where: { $0.id == selection.id }) else {
                    await self.publishError("Couldn't find an object at that location.")
                    return
                }

                let overlayId = chosen.id
                let viewRect = await MainActor.run { self.convertToViewRect(chosen.boundingBox) }

                var overlay = InfoOverlay(id: overlayId,
                                          normalizedBoundingBox: chosen.boundingBox,
                                          rect: viewRect,
                                          label: chosen.label,
                                          confidence: chosen.confidence,
                                          contourPath: chosen.contourPath,
                                          equation: "Generating equation...",
                                          explanation: "Fetching explanation...",
                                          status: .loading)

                await MainActor.run {
                    self.infoOverlays = [overlay]
                }

                guard let imageData = ImageCropper.jpegData(from: snapshot.pixelBuffer,
                                                            boundingBox: chosen.boundingBox,
                                                            orientation: snapshot.orientation) else {
                    await self.publishError("Failed to crop selection for upload.")
                    return
                }

                let uploadRequest = ObjectUploadRequest(trackId: chosen.id,
                                                        label: chosen.label,
                                                        confidence: chosen.confidence,
                                                        boundingBox: chosen.boundingBox,
                                                        imageData: imageData)

                let facts = try await self.processingService.fetchFacts(request: uploadRequest,
                                                                        lensMode: self.lensMode)

                await MainActor.run {
                    self.applyFacts(facts)
                    self.statusBanner = StatusBanner(message: "Overlay updated with insights.", tone: .info)
                }
            } catch is CancellationError {
                return
            } catch {
                await self.publishError("Segmentation failed: \(error.localizedDescription)")
                await MainActor.run {
                    self.markInfoOverlayFailed("Unable to fetch lens data.")
                }
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isActive else { return }
        let orientation = CGImagePropertyOrientation(deviceOrientation: UIDevice.current.orientation)
        latestSnapshot = FrameSnapshot(pixelBuffer: frame.capturedImage,
                                       orientation: orientation,
                                       timestamp: frame.timestamp)
    }

    func handleTap(at viewPoint: CGPoint) {
        guard isActive else {
            publishError("Start the camera before selecting an object.")
            return
        }

        guard viewportSize != .zero else {
            publishError("Camera view is still sizing. Please try again.")
            return
        }

        guard let snapshot = latestSnapshot else {
            publishError("Need a fresh camera frame. Move the device slightly and tap again.")
            return
        }

        let normalizedPoint = normalizedVisionPoint(fromViewPoint: viewPoint)

        switch pipeline {
        case .hologram:
            runAnchoredPipeline(snapshot: snapshot, normalizedPoint: normalizedPoint)
        case .infoOverlay:
            runInfoPipeline(snapshot: snapshot, normalizedPoint: normalizedPoint)
        }
    }

    @MainActor
    private func prepareFrameUpdate(predictions: [YOLOPrediction],
                                    snapshot: FrameSnapshot) -> [ObjectUploadRequest] {
        let tracks = tracker.update(predictions: predictions, timestamp: snapshot.timestamp)
        ensureWorldTransforms(for: tracks)
        updateOverlays(with: tracks)
        removeAnchorsIfNeeded(validTrackIds: Set(tracks.map(\.id)))

        var uploadRequests: [ObjectUploadRequest] = []
        for track in tracks {
            let state = objectStates[track.id] ?? ObjectState()
            if shouldSubmit(track: track, state: state) {
                if let imageData = ImageCropper.jpegData(from: snapshot.pixelBuffer,
                                                         boundingBox: track.boundingBox,
                                                         orientation: snapshot.orientation) {
                    let request = ObjectUploadRequest(trackId: track.id,
                                                      label: track.label,
                                                      confidence: track.confidence,
                                                      boundingBox: track.boundingBox,
                                                      imageData: imageData)
                    uploadRequests.append(request)

                    var updatedState = state
                    updatedState.submissionState = .uploading
                    updatedState.lastSubmittedBox = track.boundingBox
                    updatedState.overlayImage = nil
                    objectStates[track.id] = updatedState
                } else {
                    var failedState = state
                    failedState.submissionState = .failed("Crop failed")
                    failedState.overlayImage = nil
                    objectStates[track.id] = failedState
                }
            }
        }

        return uploadRequests
    }

    private func refreshInfoOverlayLayouts() {
        guard usesInfoPipeline, viewportSize != .zero, !infoOverlays.isEmpty else { return }
        infoOverlays = infoOverlays.map { overlay in
            var updated = overlay
            updated.rect = convertToViewRect(overlay.normalizedBoundingBox)
            return updated
        }
    }

    private func updateOverlays(with tracks: [ObjectTrack]) {
        guard viewportSize != .zero, pipeline == .hologram else { return }
        overlays = tracks.map { track in
            let rect = convertToViewRect(track.boundingBox)
            let submissionState = objectStates[track.id]?.submissionState ?? .idle
            let overlayImage = objectStates[track.id]?.overlayImage
            return DetectionOverlay(id: track.id,
                                    rect: rect,
                                    label: track.label,
                                    confidence: track.confidence,
                                    submissionState: submissionState,
                                    overlayImage: overlayImage)
        }
    }

    private func convertToViewRect(_ boundingBox: CGRect) -> CGRect {
        var rect = boundingBox
        rect.origin.y = 1 - rect.origin.y - rect.height
        rect.origin.x *= viewportSize.width
        rect.origin.y *= viewportSize.height
        rect.size.width *= viewportSize.width
        rect.size.height *= viewportSize.height
        return rect
    }

    private func ensureWorldTransforms(for tracks: [ObjectTrack]) {
        guard let arView else { return }
        for track in tracks {
            if tracker.track(for: track.id)?.worldTransform == nil {
                let _ = resolveWorldTransform(for: track, in: arView)
            }
        }
    }

    private func resolveWorldTransform(for track: ObjectTrack, in arView: ARView) -> simd_float4x4? {
        let screenPoint = CGPoint(x: track.boundingBox.midX * viewportSize.width,
                                  y: (1 - track.boundingBox.midY) * viewportSize.height)

        if let raycastResult = arView.raycast(from: screenPoint,
                                              allowing: .estimatedPlane,
                                              alignment: .any).first {
            tracker.updateWorldTransform(raycastResult.worldTransform, for: track.id)
            return raycastResult.worldTransform
        }

        if let query = arView.makeRaycastQuery(from: screenPoint,
                                               allowing: .estimatedPlane,
                                               alignment: .any),
           let alternateResult = arView.session.raycast(query).first {
            tracker.updateWorldTransform(alternateResult.worldTransform, for: track.id)
            return alternateResult.worldTransform
        }

        return nil
    }

    private func shouldSubmit(track: ObjectTrack, state: ObjectState) -> Bool {
        switch state.submissionState {
        case .uploading, .awaitingServer:
            return false
        case .processed:
            guard let lastBox = state.lastSubmittedBox else { return false }
            let iou = lastBox.intersectionOverUnion(with: track.boundingBox)
            return iou < 0.3
        case .failed:
            return true
        case .idle:
            return true
        }
    }

    private func upload(request: ObjectUploadRequest) async {
        do {
            let result = try await processingService.process(request: request, lensMode: lensMode)
            await MainActor.run {
                var state = objectStates[request.trackId] ?? ObjectState()
                state.submissionState = .processed
                state.overlayImage = result.image
                objectStates[request.trackId] = state
                statusBanner = StatusBanner(message: result.message ?? "Received annotated asset.",
                                            tone: .info)
                placeAnchor(for: request.trackId, image: result.image)
                refreshOverlay(for: request.trackId)
            }
        } catch {
            await MainActor.run {
                var state = objectStates[request.trackId] ?? ObjectState()
                let friendly = APIService.shared.mapErrorToMessage(error)
                let detailed = APIService.shared.debugDescription(for: error)
                let combined = friendly == detailed ? friendly : "\(friendly)\n\(detailed)"
                state.submissionState = .failed(combined)
                state.overlayImage = nil
                objectStates[request.trackId] = state
                statusBanner = StatusBanner(message: "Upload failed: \(combined)",
                                            tone: .error)
                refreshOverlay(for: request.trackId)
            }
        }
    }

    private func refreshOverlay(for trackId: UUID) {
        guard let index = overlays.firstIndex(where: { $0.id == trackId }) else { return }
        let submissionState = objectStates[trackId]?.submissionState ?? .idle
        let overlayImage = objectStates[trackId]?.overlayImage
        let overlay = overlays[index]
        overlays[index] = DetectionOverlay(id: overlay.id,
                                           rect: overlay.rect,
                                           label: overlay.label,
                                           confidence: overlay.confidence,
                                           submissionState: submissionState,
                                           overlayImage: overlayImage)
    }

    private func applyFacts(_ facts: LensFactResult) {
        guard let index = infoOverlays.firstIndex(where: { $0.id == facts.trackId }) else { return }
        infoOverlays[index].equation = facts.equation
        infoOverlays[index].explanation = facts.explanation
        infoOverlays[index].status = .loaded
    }

    private func markInfoOverlayFailed(_ message: String) {
        guard !infoOverlays.isEmpty else { return }
        infoOverlays = infoOverlays.map { overlay in
            var updated = overlay
            updated.status = .failed(message)
            if (overlay.explanation?.isEmpty ?? true) {
                updated.explanation = message
            }
            return updated
        }
    }

    private func placeAnchor(for trackId: UUID, image: UIImage) {
        guard let arView else { return }

        if let existingAnchor = anchorMap[trackId] {
            update(anchor: existingAnchor, with: image)
            return
        }

        guard let track = tracker.track(for: trackId) else { return }
        let worldTransform = track.worldTransform ?? resolveWorldTransform(for: track, in: arView)
        guard let transform = worldTransform else { return }

        let anchor = AnchorEntity(world: transform.translation)
        anchor.addChild(makeBillboard(for: image))
        arView.scene.addAnchor(anchor)
        anchorMap[trackId] = anchor
    }

    private func makeBillboard(for image: UIImage) -> ModelEntity {
        let aspect = Float(image.size.height / max(image.size.width, 0.0001))
        let width: Float = 0.25
        let height = width * aspect
        let mesh = MeshResource.generatePlane(width: width, depth: height)

        var material = UnlitMaterial()
        if let cgImage = image.cgImage,
           let texture = try? TextureResource.generate(from: cgImage, options: .init(semantic: .color)) {
            material.baseColor = .texture(texture)
        } else {
            material.baseColor = .color(.white)
        }

        let model = ModelEntity(mesh: mesh, materials: [material])
        model.transform.rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
        return model
    }

    private func update(anchor: AnchorEntity, with image: UIImage) {
        guard let model = anchor.children.compactMap({ $0 as? ModelEntity }).first else { return }

        if let cgImage = image.cgImage,
           let texture = try? TextureResource.generate(from: cgImage, options: .init(semantic: .color)) {
            var material = UnlitMaterial()
            material.baseColor = .texture(texture)
            model.model?.materials = [material]
        }
    }

    private func removeAnchorsIfNeeded(validTrackIds: Set<UUID>) {
        guard let arView else { return }
        for (id, anchor) in anchorMap where !validTrackIds.contains(id) {
            arView.scene.removeAnchor(anchor)
            anchorMap.removeValue(forKey: id)
            objectStates.removeValue(forKey: id)
        }
    }

    private func selectionBoundingBox(from base: CGRect, tapPoint: CGPoint?) -> CGRect {
        var rect = base.insetBy(dx: -base.width * selectionPaddingFraction,
                                dy: -base.height * selectionPaddingFraction)

        if let point = tapPoint {
            let influence = max(max(base.width, base.height) * (1 + selectionPaddingFraction),
                                tapInfluenceFraction)
            let tapRect = CGRect(x: point.x - influence / 2,
                                 y: point.y - influence / 2,
                                 width: influence,
                                 height: influence)
            rect = rect.union(tapRect)
        }

        return rect.clampedToUnitSquare()
    }

    private func normalizedVisionPoint(fromViewPoint point: CGPoint) -> CGPoint? {
        guard viewportSize.width > 0, viewportSize.height > 0 else { return nil }
        let normalizedX = max(0, min(1, point.x / viewportSize.width))
        let normalizedY = max(0, min(1, 1 - (point.y / viewportSize.height)))
        return CGPoint(x: normalizedX, y: normalizedY)
    }

    private static func selectPrediction(near normalizedPoint: CGPoint?,
                                         from predictions: [YOLOPrediction]) -> YOLOPrediction? {
        guard !predictions.isEmpty else { return nil }
        guard let point = normalizedPoint else {
            return predictions.max(by: { $0.confidence < $1.confidence })
        }

        let containing = predictions.filter { $0.boundingBox.contains(point) }
        if let best = containing.max(by: { $0.confidence < $1.confidence }) {
            return best
        }

        return predictions.min { lhs, rhs in
            let lhsCenter = CGPoint(x: lhs.boundingBox.midX, y: lhs.boundingBox.midY)
            let rhsCenter = CGPoint(x: rhs.boundingBox.midX, y: rhs.boundingBox.midY)
            let lhsDistance = distanceSquared(lhsCenter, point)
            let rhsDistance = distanceSquared(rhsCenter, point)

            if lhsDistance == rhsDistance {
                return lhs.confidence > rhs.confidence
            }
            return lhsDistance < rhsDistance
        }
    }

    private static func distanceSquared(_ lhs: CGPoint, _ rhs: CGPoint) -> CGFloat {
        let dx = lhs.x - rhs.x
        let dy = lhs.y - rhs.y
        return (dx * dx) + (dy * dy)
    }

    private static func pipeline(for lensMode: String) -> LensPipeline {
        let infoModes: Set<String> = ["biologist", "artist", "ecologist", "cultural"]
        return infoModes.contains(lensMode.lowercased()) ? .infoOverlay : .hologram
    }

    private static func accentColor(for lensMode: String) -> Color {
        if let lens = LENS_MODES.first(where: { $0.id == lensMode }) {
            return Color(hex: lens.color)
        }
        return Color.cyan
    }

    private func publishError(_ message: String) {
        statusBanner = StatusBanner(message: message, tone: .error)
    }
}

private struct FrameSnapshot: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let orientation: CGImagePropertyOrientation
    let timestamp: TimeInterval
}

private struct ObjectState {
    var submissionState: SubmissionState = .idle
    var lastSubmittedBox: CGRect?
    var overlayImage: UIImage?
}
