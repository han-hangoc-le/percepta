import CoreGraphics

extension CGRect {
    func intersectionOverUnion(with other: CGRect) -> CGFloat {
        let intersectionRect = self.intersection(other)
        guard !intersectionRect.isNull else { return 0 }

        let intersectionArea = intersectionRect.width * intersectionRect.height
        let unionArea = (width * height) + (other.width * other.height) - intersectionArea

        guard unionArea > 0 else { return 0 }
        return intersectionArea / unionArea
    }

    func clampedToUnitSquare() -> CGRect {
        let minX = max(0, min(1, origin.x))
        let minY = max(0, min(1, origin.y))
        let maxX = max(minX, min(1, origin.x + width))
        let maxY = max(minY, min(1, origin.y + height))
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}
