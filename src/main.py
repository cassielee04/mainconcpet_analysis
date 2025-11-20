# main.py
import os
import torch
import pandas as pd
import json
from lexicalrichness import LexicalRichness
from glob import glob

from preprocessing import load_text_from_file, clean_text, load_text_from_aws_json
from features import extract_84_features, analyze_transcript_qpa, get_stanza_pipeline, get_total_speech_duration
from pauses_fillers import count_fillers_pauses_json
from fillers import count_fillers
from ciu import calculate_cinderella_ciu
from segment_utterance import segment_utterances
from mainconcept_normalize import MainConceptAnalyzerNormalize
import sys
from pathlib import Path
from normalize_utterances import normalize_utterances
from coherence import count_errors, calculate_coherence_error_percentages
import os

def process_file(json_path):
    # Extract participant ID
    base = os.path.splitext(os.path.basename(json_path))[0]
    participant_id = base.split('_')[0]

    # Load and clean text
    text = load_text_from_aws_json(json_path)
    cleaned_text = clean_text(text)
    total_words = len(cleaned_text.split())

    # Feature extraction
    df_features = pd.DataFrame([{}]) 


    # Fillers and pause analysis
    json_data = {}
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    df_features['participant'] = participant_id

    # # Utterance Level Features
    # # 1. Noun Verb Ratio
    
    segmented_utterances_cleaned = segment_utterances(cleaned_text)
    analyzer = MainConceptAnalyzerNormalize()
    # # Main Concept Matching
    df_features['total_utterances'] = len(segmented_utterances_cleaned)
    normalized_utterances = normalize_utterances(segmented_utterances_cleaned)
    main_concepts_df = analyzer.get_mainconcept_match(segmented_utterances_cleaned, normalized_utterances,  return_score=True)
    df_features['distance_to_centroid'] = main_concepts_df['distance_to_centroid']
    df_features['num_unique_main_concepts'] = analyzer.get_total_unique_mainconcepts()
    df_features['num_total_main_concepts'] = analyzer.get_total_mainconcepts()
    df_features['unique_main_concept_match_ratio'] = (df_features['num_unique_main_concepts'] / 34)
    df_features['total_main_concept_match_ratio'] = (df_features['num_total_main_concepts'] / df_features['total_utterances'])
    matched_seq, sequence_score = analyzer.score_story_sequence(segmented_utterances_cleaned, normalized_utterances, return_score=True)
    df_features['sequence_score_mainconcept'] = sequence_score


    # After calling count_errors
    participant_codes = [participant_id] * len(segmented_utterances_cleaned)
    counts, utterance_errors = count_errors(segmented_utterances_cleaned, participant_codes)
    total_utterances = int(df_features['total_utterances'].iloc[0])
    # Extract local/global percentages (as you have)
    local_pct, global_pct = calculate_coherence_error_percentages(counts, total_utterances)
    df_features['local_coherence_mainconcept'] = local_pct
    df_features['global_coherence_mainconcept'] = global_pct

    # # Define error types (from classify_error logic)
    error_types = [
        'Topic Switching', 'Missing Referent', 'Tangential Utterance',
        'Propositional Repetition', 'Filler', 'Conceptual Incongruence', 'Coherent', 
    ]

    # Extract counts and ratios for each (0 if not present)
    for error_type in error_types:
        num_col = f'num_{error_type.lower().replace(" ", "_")}'
        ratio_col = f'ratio_{error_type.lower().replace(" ", "_")}_percent'
        
        num = counts.get(error_type, 0)
        ratio = (num / total_utterances * 100) if total_utterances > 0 else 0.0
        
        df_features[num_col] = num
        df_features[ratio_col] = round(ratio, 2)  # Round to 2 decimals for cleanliness

    # Reorder columns
    cols = ['participant'] + [col for col in df_features.columns if col != 'participant']
    df_features = df_features[cols]

    return df_features


# Main routine
def main():
    folder_path = os.path.abspath("../../Data/transcripts/bank/aphasia/aws/cinderella")
    all_dfs = []

    for idx, text_file in enumerate[str](glob(os.path.join(folder_path, "*_transcribed.json"))):
        filename = os.path.basename(text_file)
        # only process wright03a file
        # if filename == "wright03a_cinderella_transcribed.json":
        base = os.path.splitext(os.path.basename(text_file))[0]
        participant_id = base.split('_')[0]
        json_path = f"../../Data/transcripts/bank/aphasia/aws/cinderella/{base}.json"
        print(f"🔍 Processing files for participant: {participant_id}")
        print(f"\n🚀 Processing Participant: {participant_id}")

        df_features = process_file(json_path)
        all_dfs.append(df_features)

    # Merge all participant data into one CSV
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv("../data/classification/aphasia_mainconcept_features_qsub.csv", index=False)
        print("\n Merged features saved to: aphasia_mainconcept_features_qsub.csv")

if __name__ == '__main__':
    main()