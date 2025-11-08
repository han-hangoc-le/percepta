import SwiftUI

struct ShutterButton: View {
    let disabled: Bool
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            ZStack {
                Circle()
                    .stroke(Color(hex: "#f8fafc"), lineWidth: 4)
                    .frame(width: 72, height: 72)
                    .background(
                        Circle()
                            .fill(Color(hex: "rgba(15,23,42,0.48)"))
                            .frame(width: 72, height: 72)
                    )
                    .opacity(disabled ? 0.6 : 1.0)
                Circle()
                    .fill(Color(hex: "#f8fafc"))
                    .frame(width: 48, height: 48)
            }
        }
        .disabled(disabled)
    }
}
