import UIKit
import ImageIO

extension CGImagePropertyOrientation {
    init(deviceOrientation: UIDeviceOrientation) {
        switch deviceOrientation {
        case .portraitUpsideDown:
            self = .left
        case .landscapeLeft:
            self = .up
        case .landscapeRight:
            self = .down
        case .portrait:
            fallthrough
        default:
            self = .right
        }
    }
}
