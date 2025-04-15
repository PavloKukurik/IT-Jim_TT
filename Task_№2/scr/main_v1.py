import os
import json
import argparse
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm


def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.hstack((mfcc_mean, tempo))

def cluster_songs(folder_path, n_clusters):
    print("[INFO] Feature extraction started...")
    song_ids = []
    features = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".mp3"):
            path = os.path.join(folder_path, filename)
            try:
                feat = extract_features(path)
                features.append(feat)
                song_ids.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"[WARNING] Failed on {filename}: {e}")

    print("[INFO] Feature extraction completed.")

    X = np.array(features)
    X = StandardScaler().fit_transform(X)

    print("[INFO] Clustering started...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    playlists = []
    for cluster_id in range(n_clusters):
        songs = [song_ids[i] for i in range(len(song_ids)) if labels[i] == cluster_id]
        playlists.append({"id": cluster_id, "songs": songs})

    output = {"playlists": playlists}
    with open("playlists_v1.json", "w") as f:
        json.dump(output, f, indent=2)

    print("[INFO] playlists_v1.json saved.")
    print("[INFO] Process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to folder with audio files")
    parser.add_argument("--n", type=int, default=3, help="Number of playlists (clusters)")
    args = parser.parse_args()

    cluster_songs(args.path, args.n)
