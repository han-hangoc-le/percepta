import Foundation
import CoreGraphics
import simd

struct ObjectTrack: Identifiable {
    let id: UUID
    var label: String
    var confidence: Float
    /// Normalized bounding box (Vision coordinate space).
    var boundingBox: CGRect
    var lastSeenTimestamp: TimeInterval
    var worldTransform: simd_float4x4?
}

final class ObjectTracker {
    private var tracks: [UUID: ObjectTrack] = [:]
    private let iouThreshold: CGFloat
    private let expirationInterval: TimeInterval

    init(iouThreshold: CGFloat = 0.4, expirationInterval: TimeInterval = 1.2) {
        self.iouThreshold = iouThreshold
        self.expirationInterval = expirationInterval
    }

    func update(predictions: [YOLOPrediction], timestamp: TimeInterval) -> [ObjectTrack] {
        var updatedTracks: [UUID: ObjectTrack] = [:]
        var unmatchedPredictions = predictions

        // Try matching existing tracks with new predictions
        for track in tracks.values {
            var bestMatchIndex: Int?
            var bestIoU: CGFloat = 0

            for (index, prediction) in unmatchedPredictions.enumerated() {
                let iou = track.boundingBox.intersectionOverUnion(with: prediction.boundingBox)
                if iou > bestIoU {
                    bestIoU = iou
                    bestMatchIndex = index
                }
            }

            if let matchIndex = bestMatchIndex, bestIoU >= iouThreshold {
                var updatedTrack = track
                let prediction = unmatchedPredictions.remove(at: matchIndex)
                updatedTrack.boundingBox = prediction.boundingBox
                updatedTrack.confidence = prediction.confidence
                updatedTrack.label = prediction.identifier
                updatedTrack.lastSeenTimestamp = timestamp
                updatedTracks[updatedTrack.id] = updatedTrack
            } else if timestamp - track.lastSeenTimestamp <= expirationInterval {
                updatedTracks[track.id] = track
            }
        }

        // Any remaining predictions start new tracks
        for prediction in unmatchedPredictions {
            let newTrack = ObjectTrack(id: UUID(),
                                       label: prediction.identifier,
                                       confidence: prediction.confidence,
                                       boundingBox: prediction.boundingBox,
                                       lastSeenTimestamp: timestamp,
                                       worldTransform: nil)
            updatedTracks[newTrack.id] = newTrack
        }

        tracks = updatedTracks
        return tracks.values.sorted { $0.lastSeenTimestamp < $1.lastSeenTimestamp }
    }

    func remove(trackId: UUID) {
        tracks.removeValue(forKey: trackId)
    }

    func allTracks() -> [ObjectTrack] {
        Array(tracks.values)
    }

    func track(for id: UUID) -> ObjectTrack? {
        tracks[id]
    }

    func updateWorldTransform(_ transform: simd_float4x4?, for trackId: UUID) {
        guard var track = tracks[trackId] else { return }
        track.worldTransform = transform
        tracks[trackId] = track
    }

    func reset() {
        tracks.removeAll()
    }
}
