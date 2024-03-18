import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import optimizers
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PARENT_DIR = "C:/Users/teodo/PycharmProjects/pythonProject1"
IMG_DIR = f"{PARENT_DIR}/bee_imgs/bee_imgs/"
CSV_DIR = f"{PARENT_DIR}/bee_data.csv"
epochs = 50
df = pd.read_csv(CSV_DIR)

X_pics = [Image.open(f"{IMG_DIR}{img_name}").convert('RGB').resize((64, 64)) for img_name in df["file"]]
X = [np.array(img) / 255.0 for img in X_pics]

y_keys = {
    "healthy": np.array([1, 0, 0, 0, 0, 0]),
    "few varrao, hive beetles": np.array([0, 1, 0, 0, 0, 0]),
    "Varroa, Small Hive Beetles": np.array([0, 0, 1, 0, 0, 0]),
    "ant problems": np.array([0, 0, 0, 1, 0, 0]),
    "hive being robbed": np.array([0, 0, 0, 0, 1, 0]),
    "missing queen": np.array([0, 0, 0, 0, 0, 1])
}
y = [y_keys[label] for label in df["health"]]


def random_imgs(data_frame, num_images, x_pics):
    index_lst = data_frame["file"].sample(n=num_images, random_state=1).index
    return [x_pics[i] for i in index_lst]


def plot_bees(img_lst, title):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 8))
    for i, img in enumerate(img_lst):
        ax[i].imshow(img)
    plt.title(title)
    plt.show()


healthy = random_imgs(df[df["health"] == "healthy"], 4, X_pics)
hive_beetles = random_imgs(df[df["health"] == "few varrao, hive beetles"], 4, X_pics)
ant_probs = random_imgs(df[df["health"] == "ant problems"], 4, X_pics)
hive_robbed = random_imgs(df[df["health"] == "hive being robbed"], 4, X_pics)
varroa = random_imgs(df[df["health"] == "Varroa, Small Hive Beetles"], 4, X_pics)

plot_bees(healthy, "healthy")
plot_bees(hive_beetles, "few varrao, hive beetles")
plot_bees(ant_probs, "ant problems")
plot_bees(hive_robbed, "hive being robbed")
plot_bees(varroa, "Varroa, Small Hive Beetles")

history = tf.keras.callbacks.History()


def train_cnn():
    model = Sequential()
    model.add(Convolution2D(11, 3, 3, input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
    model.add(Convolution2D(21, 3, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation="softmax"))

    model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=1)


def model_plot(history, title):
    y_range = [0.5, 1.0]

    plt.plot(history.history["loss"])
    plt.legend(["Train Loss"])
    plt.title(title + " - Loss")
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.legend(["Train Accuracy"])
    plt.title(title + " - Accuracy")
    plt.ylim(y_range)
    plt.show()


def data_aug(X_train):
    datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.3,
        height_shift_range=0.3
    )

    datagen.fit(X_train)
    return datagen


model1 = train_cnn()
history1 = model1.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_val), np.array(y_val)),
                      verbose=True, shuffle=True, epochs=epochs)

datagen = data_aug(X_train)

model2 = train_cnn()
history2 = model2.fit_generator(datagen.flow(np.array(X_train), np.array(y_train), batch_size=50),
                                validation_data=(np.array(X_val), np.array(y_val)),
                                steps_per_epoch=len(X_train) / 50, epochs=epochs)

model_plot(history1, title="Model 1")
model_plot(history2, title="Model 2 with Augmentation")

def multi_pred(models, X_test, y_test):
    preds = [model.predict(np.array(X_test)) for model in models]
    cols = [f"model{i}_prediction" for i in range(1, len(models) + 1)]
    preds_df = pd.DataFrame(data=np.argmax(np.array(preds).mean(axis=0), axis=1),
                            columns=["ensemble_prediction"])

    for i, col in enumerate(cols):
        preds_df[col] = np.argmax(np.array(preds[i]), axis=1)

    preds_df["target"] = np.argmax(np.array(y_test), axis=1)
    return preds_df


preds = multi_pred([model1, model2], X_test, y_test)
print(preds)
accuracy_1 = accuracy_score(preds["model1_prediction"], preds["target"])
print("Accuracy for Model 1:", accuracy_1)

accuracy_2 = accuracy_score(preds["model2_prediction"], preds["target"])
print("Accuracy for Model 2:", accuracy_2)


def accuracy_table(df):
    models_lst = df.columns.tolist()  # Putting model names into a list
    models_lst.remove("target")
    health_cols = ["healthy", "few varrao, hive beetles", "Varroa, Small Hive Beetles", "ant problems",
                   "hive being robbed", "missing queen"]
    df_acc = []

    for model in models_lst:
        acc_lst = []
        for i in range(6):
            size = (df["target"] == i).sum()
            true = ((df[model] == i) & (df["target"] == i)).sum()
            acc_lst.append(true / size)
        df_acc.append(acc_lst)

    df_acc = pd.DataFrame(df_acc, columns=health_cols)
    df_acc.index = models_lst
    return df_acc


df_acc = accuracy_table(preds)
print("Accuracy Table:")
print(df_acc)

accuracy_file_path = "accuracy_table.csv"
df_acc.to_csv(accuracy_file_path)

print(f"Accuracy table saved to {accuracy_file_path}")
