import pandas as pd
import yaml
import pathlib
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import umap  # Requires: pip install umap-learn
import torch
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns  # For better plots, optional: pip install seaborn

# Define paths and model name
config_path = "../config/story_config.yml"  # Replace with your actual config path
model_name = 'all-mpnet-base-v2'
patients_df = pd.read_csv('../data/utterances_aphasia_output.csv')  # Replace with your patients file path
controls_df = pd.read_csv('../data/utterances_controls_output.csv')  # Replace with your controls file path

# Extract utterances
# Add group column to distinguish
patients_df['group'] = 'patient'
controls_df['group'] = 'control'

# Combine dataframes for easier grouping (or process separately)
df = pd.concat([patients_df, controls_df], ignore_index=True)

# Extract utterances per participant
# Group by participant_code and group, list utterances
grouped = df.groupby(['participant_code', 'group'])['utterance'].apply(list).reset_index()

# Load the model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(model_name)

# Compute per-utterance embeddings, then average per participant
participant_embeddings = []
participant_labels = []
participant_codes = []

for _, row in grouped.iterrows():
    utterances = row['utterance']
    participant_code = row['participant_code']
    group = row['group']
    
    if len(utterances) == 0:
        continue
    
    # Compute embeddings for this participant's utterances
    embeds = embedder.encode(utterances, device=device, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    
    # Average embedding for the participant
    avg_embed = np.mean(embeds, axis=0)
    
    participant_embeddings.append(avg_embed)
    participant_labels.append(group)
    participant_codes.append(participant_code)

participant_embeddings = np.array(participant_embeddings)

# Load centroid from YAML config
cfg = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
concepts = cfg["stories"][0]["concepts"]
assert len(concepts) == 34, f"Expected 34 concepts, got {len(concepts)}"

# Compute centroid embedding
concept_embeds = embedder.encode(concepts, convert_to_tensor=True, normalize_embeddings=True, device=device)
centroid = torch.mean(concept_embeds, dim=0, keepdim=True).cpu().numpy()  # Convert to numpy for consistency

# Compute cosine distances to centroid (per participant)
participant_distances = cosine_distances(participant_embeddings, centroid.reshape(1, -1)).flatten()

# Separate distances by group for stats
patient_indices = [i for i, label in enumerate(participant_labels) if label == 'patient']
control_indices = [i for i, label in enumerate(participant_labels) if label == 'control']

patient_distances_group = [participant_distances[i] for i in patient_indices]
control_distances_group = [participant_distances[i] for i in control_indices]

# Print average distances per group
print(f"Average cosine distance for patients: {np.mean(patient_distances_group):.4f} (std: {np.std(patient_distances_group):.4f})")
print(f"Average cosine distance for controls: {np.mean(control_distances_group):.4f} (std: {np.std(control_distances_group):.4f})")
print(f"Total participants: {len(participant_labels)} (Patients: {len(patient_indices)}, Controls: {len(control_indices)})")

# Plot histograms of distances (per participant)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(patient_distances_group, bins=20, alpha=0.7, label='Patients', color='red')
plt.hist(control_distances_group, bins=20, alpha=0.7, label='Controls', color='blue')
plt.xlabel('Cosine Distance to Centroid')
plt.ylabel('Frequency (Participants)')
plt.title('Distribution of Participant Distances to Centroid')
plt.legend()

# UMAP plot with centroid (on participant average embeddings)
# --------------------------------------------
# UMAP plot with centroid (on participant average embeddings)
# --------------------------------------------

# Stack participant embeddings and centroid
all_data = np.vstack([participant_embeddings, centroid])

# UMAP tuned a bit more for global-ish structure
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=40,         # larger -> more global structure
    min_dist=0.4,           # larger -> less crumpling
    metric='cosine',        # good for normalized sentence embeddings
    local_connectivity=3,
    random_state=42
)

all_2d = reducer.fit_transform(all_data)

embeddings_2d = all_2d[:-1]              # Participant embeddings in 2D
centroid_2d = all_2d[-1].reshape(1, -1)  # Centroid in 2D

plt.figure(figsize=(14, 6))

# --- LEFT: colored by group ---
plt.subplot(1, 2, 1)

for i, (x, y) in enumerate(embeddings_2d):
    if participant_labels[i] == 'patient':
        plt.scatter(x, y, marker='x', color='red', s=80, linewidths=2)
    else:
        plt.scatter(x, y, marker='o', color='blue', s=60, edgecolor='k', linewidth=0.7)

# centroid
plt.scatter(centroid_2d[:, 0], centroid_2d[:, 1],
            marker='*', color='green', s=180, edgecolor='black', linewidth=0.9)

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP of Participant Embeddings (X = Patient, O = Control)')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='x', color='red', markersize=10, linewidth=0,
           label='Patient'),
    Line2D([0], [0], marker='o', color='blue', markeredgecolor='k', markersize=8,
           label='Control'),
    Line2D([0], [0], marker='*', color='green', markeredgecolor='black',
           markersize=14, label='Centroid')
]
plt.legend(handles=legend_elements, loc='best')


# --- RIGHT: colored by TRUE cosine distance to centroid ---
plt.subplot(1, 2, 2)

for i, (x, y) in enumerate(embeddings_2d):
    if participant_labels[i] == 'patient':
        plt.scatter(x, y, c=participant_distances[i], cmap='viridis',
                    marker='x', s=90, linewidths=2, vmin=min(participant_distances),
                    vmax=max(participant_distances))
    else:
        plt.scatter(x, y, c=participant_distances[i], cmap='viridis',
                    marker='o', s=70, edgecolor='k', linewidth=0.7,
                    vmin=min(participant_distances),
                    vmax=max(participant_distances))

# centroid
plt.scatter(centroid_2d[:, 0], centroid_2d[:, 1],
            marker='*', color='white', s=180, edgecolor='black', linewidth=1.2)

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP Colored by Distance to Centroid')

cbar = plt.colorbar()
cbar.set_label('Cosine Distance to Centroid')

plt.tight_layout()
plt.savefig('participant_embeddings_umap_shapes_aphasia_center.png',
            dpi=300, bbox_inches='tight')
plt.close()

# --------------------------------------------
# OPTIONAL: MDS plot that tries to preserve distances more faithfully
# --------------------------------------------

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances as cosdist

# Compute cosine distance matrix
all_data = np.vstack([participant_embeddings, centroid])
dist_matrix = cosdist(all_data, all_data)

# Fit MDS preserving distances
mds = MDS(
    n_components=2,
    dissimilarity='precomputed',
    random_state=42,
    n_init=4,
    max_iter=300
)

all_mds_2d = mds.fit_transform(dist_matrix)
mds_2d = all_mds_2d[:-1]                # Participant positions
centroid_mds_2d = all_mds_2d[-1].reshape(1, -1)  # Centroid position

plt.figure(figsize=(14, 6))

# -------------------------------------------------
# LEFT: MDS by group with shape markers
# -------------------------------------------------
plt.subplot(1, 2, 1)

for i, (x, y) in enumerate(mds_2d):
    if participant_labels[i] == 'patient':
        plt.scatter(x, y, marker='x', color='red', s=80, linewidths=2)
    else:
        plt.scatter(x, y, marker='o', color='blue', s=60, edgecolor='k', linewidth=0.7)

# centroid
plt.scatter(centroid_mds_2d[:, 0], centroid_mds_2d[:, 1],
            marker='*', color='green', s=180, edgecolor='black', linewidth=1)

plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Embedding (Distance-Preserving)')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='x', color='red', markersize=10, linewidth=0, label='Patient'),
    Line2D([0], [0], marker='o', color='blue', markeredgecolor='k', markersize=8, label='Control'),
    Line2D([0], [0], marker='*', color='green', markeredgecolor='black', markersize=14, label='Centroid')
]
plt.legend(handles=legend_elements, loc='best')

# -------------------------------------------------
# RIGHT: MDS colored by cosine distance
# -------------------------------------------------
plt.subplot(1, 2, 2)

dist_min, dist_max = min(participant_distances), max(participant_distances)

for i, (x, y) in enumerate(mds_2d):
    if participant_labels[i] == 'patient':
        plt.scatter(x, y, c=participant_distances[i], cmap='viridis',
                    marker='x', s=90, linewidths=2, vmin=dist_min, vmax=dist_max)
    else:
        plt.scatter(x, y, c=participant_distances[i], cmap='viridis',
                    marker='o', s=70, edgecolor='k', linewidth=0.7,
                    vmin=dist_min, vmax=dist_max)

# centroid
plt.scatter(centroid_mds_2d[:, 0], centroid_mds_2d[:, 1],
            marker='*', color='white', s=180, edgecolor='black', linewidth=1)

plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Colored by Cosine Distance to Centroid')

cbar = plt.colorbar()
cbar.set_label('Cosine Distance to Centroid')

plt.tight_layout()
plt.savefig('participant_embeddings_mds_shapes_aphasia_center.png',
            dpi=300, bbox_inches='tight')
plt.close()