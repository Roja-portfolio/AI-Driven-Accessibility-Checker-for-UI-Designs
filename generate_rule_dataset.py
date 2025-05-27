import pandas as pd

# Define example rule-based data
data = {
    "image_name": ["sample_ui.png", "google_light.png", "dark_theme_ui.png"],
    "contrast_score": [34.92, 8.69, 90.45],
    "alt_text_score": [0.0, 0.0, 80.0],
    "text_resize_score": [100.0, 100.0, 90.0],
}

# Compute final accessibility score (weighted average)
def compute_final_score(row):
    return round(
        0.4 * row["contrast_score"] +
        0.3 * row["alt_text_score"] +
        0.3 * row["text_resize_score"], 2
    )

# Create DataFrame
df = pd.DataFrame(data)
df["final_score"] = df.apply(compute_final_score, axis=1)

# Save to CSV
df.to_csv("datasets/rule_labeled_dataset.csv", index=False)
print("✅ rule_labeled_dataset.csv generated successfully.")
