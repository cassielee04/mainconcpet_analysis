import os
import re
import csv
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from test_utt2 import segment_utterances
from tools.normalize_utterances import normalize_utterances

output_csv = "data/test_utterances.csv"
true_file = "data/test_utterances_outcome.xlsx"

# Load Excel file
df = pd.read_excel("data/Utterances_index.xlsx")  # Adjusted path for test folder

# Extract only utterance column
utterances_df = df[["Utterance"]]
utterances_list = utterances_df["Utterance"].dropna().astype(str)
cleaned_utterances = " ".join( 
    re.sub(r'^\d+,|"{2,}|",?|,', '', utt).strip().lower() 
    for utt in utterances_list
)

segmented_utterances = segment_utterances(cleaned_utterances)

# Then normalize each segment individually
normalized_segmented_utterances = normalize_utterances(segmented_utterances)


#Save to CSV with index
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "utterance"])  # Header row
    for i, utt in enumerate(segmented_utterances, start=1):
        utt = utt.strip()
        if utt:  # Skip empty
            writer.writerow([i, utt])

print(f"Saved {len(segmented_utterances)} utterances to {output_csv}")

# from mainconcept import MainConceptAnalyzer
# from mainconcept_normalize import MainConceptAnalyzerNormalize

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# CONFIG_PATH = PROJECT_ROOT / "config/story_config.yml"
# EMBEDDINGS_PATH = PROJECT_ROOT / "config/cinderella_mainconcept_embeddings.pkl"

# analyzer = MainConceptAnalyzerNormalize(
#     config_path=str(CONFIG_PATH),
#     embeddings_file=str(EMBEDDINGS_PATH),
# )
# mainconcept_df = analyzer.get_mainconcept_match(segmented_utterances, normalized_segmented_utterances,  return_score=True)
# mainconcept_df.to_csv(PROJECT_ROOT / "test/data/test_mainconcept_normalize_predicted_output.csv", index=False, encoding="utf-8")

# print("✅ DataFrame saved to test_mainconcept_normalize_predicted_output.csv")