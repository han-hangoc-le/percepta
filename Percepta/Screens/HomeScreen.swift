import SwiftUI

struct HomeScreen: View {
    @State private var lensMode: String = "mathematician"
    @Environment(\.dismiss) var dismiss
    @State private var selectedLens: LensMode? = LENS_MODES.first(where: { $0.id == "mathematician" })

    var body: some View {
        NavigationStack {
            ZStack {
                Color(hex: "#0f172a").ignoresSafeArea()
                
                VStack(alignment: .leading, spacing: 16) {
                    Text("WorldLens")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(Color(hex: "#e0f2fe"))
                        .padding(.top, 12)
                    
                    Text("Choose a lens mode to transform the world through knowledge and humor.")
                        .font(.system(size: 16))
                        .foregroundColor(Color(hex: "#cbd5f5"))
                    
                    LensSelectorView(
                        lenses: LENS_MODES,
                        selectedLens: $lensMode
                    )
                    .onChange(of: lensMode) { newValue in
                        selectedLens = LENS_MODES.first(where: { $0.id == newValue })
                    }

                    VStack(alignment: .center, spacing: 8) {
                        Text("Current Lens")
                            .font(.system(size: 12))
                            .foregroundColor(Color(hex: "#94a3b8"))
                            .textCase(.uppercase)
                            .kerning(1.2)
                        
                        Text(selectedLens?.icon ?? "🔍")
                            .font(.system(size: 48))
                            .padding(.vertical, 4)
                        
                        Text(selectedLens?.name ?? "Unknown")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(Color(hex: "#f1f5f9"))
                        
                        Text(selectedLens?.description ?? "Choose a lens to reveal a playful interpretation of what you see.")
                            .font(.system(size: 14))
                            .foregroundColor(Color(hex: "#cbd5f5"))
                            .multilineTextAlignment(.center)
                            .padding(.top, 10)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(20)
                    .background(Color(hex: "#1e293b"))
                    .cornerRadius(16)

                    Spacer()

                    NavigationLink(destination: CameraScreen(lensMode: lensMode)) {
                        Text("Open Camera")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundColor(Color(hex: "#0f172a"))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(Color(hex: "#38bdf8"))
                            .cornerRadius(14)
                    }
                }
                .padding(.horizontal, 24)
            }
        }
        .navigationBarBackButtonHidden(true)
    }
}
