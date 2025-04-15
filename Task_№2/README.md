# Task 3 — Audio-Based Playlist Clustering

_Pavlo Kukurik_


This repository contains my solution to Task 3 (optional) from It-Jim's 2025 internship challenge. The goal was to automatically cluster a set of music tracks into playlists based purely on audio similarity — without using metadata, tags or genre labels.

---

[Git](https://github.com/PavloKukurik/IT-Jim_TT/tree/master)
## Problem Description

Given a folder with music files, build a tool that clusters the songs into a set of playlists based on how they sound. The program must support two different approaches:

- **Version 1**: Uses algorithmically computed features like MFCCs and tempo.
- **Version 2**: Uses deep learning-based embeddings from an open-source model.

Each version outputs a `.json` file representing clustered playlists.

---

## Project Structure

```
Task_№2/
├── data/                        # Input .mp3 files (15 tracks used)
│
├── scr/
│   ├── main_v1.py               # MFCC + tempo clustering (KMeans)
│   ├── main_v2.py               # YAMNet embeddings clustering (KMeans)
│
├── playlists_v1.json           # Output of Version 1 (MFCC)
├── playlists_v2.json           # Output of Version 2 (YAMNet)
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── report.pdf                  # Summary of the approaches and results
```

---

## Approach Overview

### Version 1 — Classical Audio Features
- Extracted **13 MFCC coefficients** and **tempo** from each track using `librosa`
- Concatenated them into a single feature vector per track
- Scaled features and clustered using `KMeans`
- Saved results as `playlists_v1.json`

### Version 2 — Deep Audio Embeddings
- Used **YAMNet** (pretrained audio classification model by Google)
- Converted each song into a 1024-dim audio embedding by averaging frame-level outputs
- Scaled embeddings and applied `KMeans` clustering
- Output written to `playlists_v2.json`

---

## Output Format (JSON)
Both scripts output a JSON file structured like this:

```json
{
  "playlists": [
    {
      "id": 0,
      "songs": ["song_id_1", "song_id_7", ...]
    },
    {
      "id": 1,
      "songs": ["song_id_3", "song_id_10"]
    }
  ]
}
```

- `song_id` = filename without extension (e.g. `114f5f3d`)
- Each playlist is a cluster of sonically similar songs

---

## How to Run

### Install dependencies:
```bash
  pip install -r requirements.txt
```

### Run Version 1:
```bash
  python3 scr/main_v1.py --path data/ --n 3
```

### Run Version 2:
```bash
  python3 scr/main_v2.py --path data/ --n 3
```

The scripts will print logs and generate:
- `playlists_v1.json`
- `playlists_v2.json`

---
