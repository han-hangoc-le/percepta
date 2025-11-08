import SwiftUI

struct PermissionDeniedView: View {
    let onRequestPermission: () -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Text("Camera access needed")
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(Color(hex: "#f8fafc"))
            
            Text("Allow LensWorld to use your camera so we can analyze the scene through your selected lens.")
                .font(.system(size: 14))
                .foregroundColor(Color(hex: "#cbd5f5"))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            
            Button(action: onRequestPermission) {
                Text("Enable Camera")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(Color(hex: "#0f172a"))
                    .padding(.horizontal, 28)
                    .padding(.vertical, 12)
                    .background(Color(hex: "#38bdf8"))
                    .cornerRadius(999)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(hex: "#0f172a"))
    }
}
