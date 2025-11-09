import SwiftUI
import RealityKit
import ARKit
import UIKit

struct ARCameraView: UIViewRepresentable {
    @ObservedObject var coordinator: DetectionCoordinator

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        coordinator.attach(arView: view)
        view.session.delegate = coordinator

        let tapRecognizer = UITapGestureRecognizer(target: context.coordinator,
                                                   action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(tapRecognizer)

        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    final class Coordinator: NSObject {
        private let parent: ARCameraView

        init(parent: ARCameraView) {
            self.parent = parent
        }

        @objc
        func handleTap(_ gesture: UITapGestureRecognizer) {
            guard gesture.state == .ended,
                  let targetView = gesture.view else { return }
            let location = gesture.location(in: targetView)
            parent.coordinator.handleTap(at: location)
        }
    }
}
