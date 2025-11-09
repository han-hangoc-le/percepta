import CoreImage
import UIKit

enum ImageCropper {
    private static let context = CIContext(options: nil)

    static func jpegData(from pixelBuffer: CVPixelBuffer,
                         boundingBox: CGRect,
                         orientation: CGImagePropertyOrientation,
                         compression: CGFloat = 0.9) -> Data? {
        let baseImage = CIImage(cvPixelBuffer: pixelBuffer)
        let orientedImage = baseImage.oriented(forExifOrientation: Int32(orientation.rawValue))
        let extent = orientedImage.extent.integral

        guard extent.width > 0, extent.height > 0 else { return nil }

        var rect = CGRect(
            x: extent.origin.x + boundingBox.origin.x * extent.width,
            y: extent.origin.y + (1 - boundingBox.origin.y - boundingBox.height) * extent.height,
            width: boundingBox.width * extent.width,
            height: boundingBox.height * extent.height
        ).integral

        rect = rect.intersection(extent)
        guard !rect.isNull,
              let cgImage = context.createCGImage(orientedImage, from: rect) else { return nil }

        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: compression)
    }
}
