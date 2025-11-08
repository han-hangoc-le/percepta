import SwiftUI

struct LensSelectorView: View {
    let lenses: [LensMode]
    @Binding var selectedLens: String      // lens id
    
    var onSelect: ((String) -> Void)? = nil

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(lenses) { lens in
                    let isSelected = lens.id == selectedLens
                    
                    Button(action: {
                        selectedLens = lens.id
                        onSelect?(lens.id)
                    }) {
                        VStack(spacing: 6) {
                            Text(lens.icon)
                                .font(.system(size: 20))
                                .foregroundColor(isSelected ? Color(hex: "#111827") : .white)
                            
                            Text(lens.name)
                                .font(.system(size: 14))
                                .fontWeight(isSelected ? .semibold : .regular)
                                .foregroundColor(isSelected ? Color(hex: "#111827") : Color(hex: "#f9fafb"))
                        }
                        .padding(.vertical, 12)
                        .padding(.horizontal, 16)
                        .background(isSelected ? Color(hex: lens.color) : Color(hex: "#1f2937"))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color(hex: lens.color), lineWidth: 2)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 8)
        }
    }
}
