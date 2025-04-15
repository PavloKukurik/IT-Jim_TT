import os
import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')


def extract_yamnet_embedding(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    scores, embeddings, spectrogram = yamnet_model(y)
    return tf.reduce_mean(embeddings, axis=0).numpy()


def cluster_songs_dl(folder_path, n_clusters):
    print("[INFO] YAMNet embedding extraction started...")
    song_ids = []
    features = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".mp3"):
            path = os.path.join(folder_path, filename)
            try:
                feat = extract_yamnet_embedding(path)
                features.append(feat)
                song_ids.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"[WARNING] Failed on {filename}: {e}")

    print("[INFO] Embedding extraction completed.")

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
    with open("playlists_v2.json", "w") as f:
        json.dump(output, f, indent=2)

    print("[INFO] playlists_v2.json saved.")
    print("[INFO] Process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to folder with audio files")
    parser.add_argument("--n", type=int, default=3, help="Number of playlists (clusters)")
    args = parser.parse_args()

    cluster_songs_dl(args.path, args.n)