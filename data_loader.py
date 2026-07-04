import matplotlib.pyplot as plt

# -----------------------------
# DATA
# -----------------------------
pathways = [
    "Biosynthesis", "Stress_Signaling", "General_Stress",
    "Osmoprotection", "Lipid_Metabolism", "ROS_Detox",
    "Carbohydrate_Metabolism", "Amino_Acid_Metabolism",
    "TCA_Cycle", "Photosynthesis", "Glycolysis",
    "Water_Transport", "Phenylpropanoid"
]

gene_counts = [2155, 2019, 1301, 572, 538, 362, 204, 204, 149, 92, 85, 44, 34]

# Colors: highlight important pathways
colors = []
for p in pathways:
    if p == "ROS_Detox":
        colors.append("red")
    elif p == "Osmoprotection":
        colors.append("green")
    elif p == "Photosynthesis":
        colors.append("blue")
    else:
        colors.append("grey")

# -----------------------------
# PLOT 1: ALL PATHWAYS
# -----------------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(pathways, gene_counts, color=colors)

plt.title("All Pathways — Gene Count\n(Red=ROS | Green=Osmo | Blue=Photo)")
plt.xlabel("Pathways")
plt.ylabel("Gene Count")
plt.xticks(rotation=45)

# Add values on bars
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 20, str(y),
             ha='center', fontsize=9)

plt.tight_layout()
plt.show()   # <-- opens FIRST window


# -----------------------------
# PLOT 2: KEY DROUGHT PATHWAYS
# -----------------------------
key_pathways = ["ROS_Detox", "Osmoprotection", "Photosynthesis"]
key_values = [362, 572, 92]
key_colors = ["red", "green", "blue"]

plt.figure(figsize=(8, 6))
bars = plt.bar(key_pathways, key_values, color=key_colors)

plt.title("Key Drought Pathways — Gene Count")
plt.ylabel("Gene Count")

# Add labels and annotations
labels = [
    "H2O2 Scavenging\nUpregulated ↑",
    "Proline / Betaine\nUpregulated ↑",
    "Calvin / PSII\nDownregulated ↓"
]

for i, bar in enumerate(bars):
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 10, str(y),
             ha='center', fontsize=12, fontweight='bold')

    plt.text(bar.get_x() + bar.get_width()/2, y/2, labels[i],
             ha='center', color='white', fontsize=10)

plt.tight_layout()
plt.show()   # <-- opens SECOND window