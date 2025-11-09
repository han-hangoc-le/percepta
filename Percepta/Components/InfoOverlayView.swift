import SwiftUI

struct InfoOverlayView: View {
    let overlays: [InfoOverlay]
    let accentColor: Color

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                ForEach(overlays) { overlay in
                    InfoOverlayRenderable(overlay: overlay,
                                          accentColor: accentColor,
                                          canvasSize: proxy.size)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

private struct InfoOverlayRenderable: View {
    let overlay: InfoOverlay
    let accentColor: Color
    let canvasSize: CGSize

    @State private var cardSize: CGSize = CGSize(width: 260, height: 140)

    var body: some View {
        let rect = overlay.rect
        let cardOrigin = computeCardOrigin(for: rect, canvas: canvasSize, cardSize: cardSize)
        let cardCenter = CGPoint(x: cardOrigin.x + cardSize.width / 2,
                                 y: cardOrigin.y + cardSize.height / 2)
        let arrowStart = CGPoint(x: cardOrigin.x + cardSize.width / 2,
                                 y: cardOrigin.y + cardSize.height)
        let arrowEnd = CGPoint(x: rect.midX, y: rect.minY)

        ZStack(alignment: .topLeading) {
            if let path = overlay.contourPath {
                ContourShape(contour: path, rect: rect)
                    .stroke(accentColor.opacity(0.85), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .shadow(color: accentColor.opacity(0.4), radius: 10)
            }

            RoundedRectangle(cornerRadius: 10)
                .stroke(accentColor, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            if rect != .zero {
                ArrowView(start: arrowStart, end: arrowEnd, color: accentColor)
            }

            InfoCardView(overlay: overlay, accentColor: accentColor)
                .frame(width: min(canvasSize.width * 0.6, 280))
                .readSize($cardSize)
                .position(cardCenter)
        }
    }

    private func computeCardOrigin(for rect: CGRect, canvas: CGSize, cardSize: CGSize) -> CGPoint {
        let safeWidth = max(cardSize.width, 200)
        let safeHeight = max(cardSize.height, 120)

        var x = rect.minX - 12
        if x < 12 {
            x = 12
        } else if x + safeWidth > canvas.width - 12 {
            x = canvas.width - safeWidth - 12
        }

        var y = rect.minY - safeHeight - 18
        if y < 12 {
            y = min(canvas.height - safeHeight - 12, rect.maxY + 18)
        }

        return CGPoint(x: x, y: max(12, y))
    }
}

private struct InfoCardView: View {
    let overlay: InfoOverlay
    let accentColor: Color

    private var statusText: String {
        switch overlay.status {
        case .idle, .loading:
            return "Analyzing..."
        case .loaded:
            return ""
        case .failed(let message):
            return message
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(overlay.label.capitalized)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.white)
                Text(String(format: "%.0f%%", overlay.confidence * 100))
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
            }

            if let equation = overlay.equation, !equation.isEmpty {
                Text(equation)
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    .foregroundColor(accentColor)
            } else if overlay.status == .loading {
                Text("Generating equation...")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
            }

            Text(overlay.explanation ?? statusText)
                .font(.system(size: 13))
                .foregroundColor(.white.opacity(0.9))
                .multilineTextAlignment(.leading)

            if case .failed(let message) = overlay.status {
                Text(message)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.red)
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.black.opacity(0.75))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(accentColor.opacity(0.8), lineWidth: 1)
                )
        )
        .shadow(color: .black.opacity(0.5), radius: 10, y: 4)
    }
}

private struct ArrowView: View {
    let start: CGPoint
    let end: CGPoint
    let color: Color

    var body: some View {
        Path { path in
            path.move(to: start)
            path.addLine(to: end)

            let angle = atan2(end.y - start.y, end.x - start.x)
            let arrowLength: CGFloat = 12
            let arrowAngle: CGFloat = .pi / 7

            let point1 = CGPoint(x: end.x - arrowLength * cos(angle - arrowAngle),
                                 y: end.y - arrowLength * sin(angle - arrowAngle))
            let point2 = CGPoint(x: end.x - arrowLength * cos(angle + arrowAngle),
                                 y: end.y - arrowLength * sin(angle + arrowAngle))

            path.move(to: point1)
            path.addLine(to: end)
            path.addLine(to: point2)
        }
        .stroke(color, style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        .shadow(color: color.opacity(0.35), radius: 4)
    }
}

private struct ContourShape: Shape {
    let contour: CGPath
    let rect: CGRect

    func path(in rect: CGRect) -> Path {
        let transform = CGAffineTransform(a: self.rect.width,
                                          b: 0,
                                          c: 0,
                                          d: -self.rect.height,
                                          tx: self.rect.minX,
                                          ty: self.rect.minY + self.rect.height)
        var path = Path(contour)
        path = path.applying(transform)
        return path
    }
}
