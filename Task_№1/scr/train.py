import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, confusion_matrix


def load_data(train_dir):
    files = [f for f in os.listdir(train_dir) if f.endswith(".png")]
    df = pd.DataFrame({
        "filename": files,
        "class": [int(f.split('_')[-1].split('.')[0]) for f in files]
    })
    return df


def get_generators(df, train_dir, target_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        validation_split=0.2
    )

    train_gen = datagen.flow_from_dataframe(
        df,
        directory=train_dir,
        x_col='filename',
        y_col='class',
        subset='training',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = datagen.flow_from_dataframe(
        df,
        directory=train_dir,
        x_col='filename',
        y_col='class',
        subset='validation',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen


def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main(args):
    os.makedirs(args.model_dir, exist_ok=True)

    df = load_data(args.data_dir)
    train_gen, val_gen = get_generators(df, args.data_dir, batch_size=args.batch_size)

    class_weights_arr = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['class']),
        y=df['class']
    )
    class_weights = {i: class_weights_arr[i] for i in range(len(class_weights_arr))}

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model_path = os.path.join(args.model_dir, "baseline_mobilenetv2.h5")
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset/train")
    parser.add_argument("--model_dir", type=str, default="../outputs/models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args)