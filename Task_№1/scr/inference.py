import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "../outputs/models/baseline_finetuned_mobilenetv2.h5"
TEST_DIR = "../dataset/test"
OUTPUT_CSV = "../outputs/results/test_predictions.csv"
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)
print("✅ Model loaded")

image_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".png")]
image_files.sort()

preds = []
for fname in image_files:
    path = os.path.join(TEST_DIR, fname)
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    preds.append((fname, int(pred > 0.5), float(pred)))

output_df = pd.DataFrame(preds, columns=["filename", "predicted_class", "confidence"])
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to {OUTPUT_CSV}")
