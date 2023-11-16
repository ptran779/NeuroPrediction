import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
import numpy as np
import pandas as pd
from DataManager import *
import matplotlib.pyplot as plt

from util import basic_func
import seaborn as sns

SEED = 12
BATCH_SIZE = 32
EPOCHS = 200
BUFFER_SIZE = 10000
TRN = 0.6
VAL = 0.2
TST = 0.2
METRICS = [
    keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
    keras.metrics.MeanSquaredError(name='Brier score'),
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1, patience=10, mode='max',
                                                  restore_best_weights=True)
keras.utils.set_random_seed(SEED)

# activate GPU accelerator
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # tf.config.experimental.set_memory_growth(gpus[0], True)  # Set memory growth to minimize GPU memory usage
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Use the first GPU

# loss fun:
# need constrative learning loss func

# this is wrong
def contrastive_loss(y_true, y_pred, margin=0.5):
    # y_true: true labels (1 for similar, 0 for dissimilar)
    # y_pred: predicted distances between pairs
    # Squared Euclidean distance
    squared_pred = tf.square(y_pred)
    # Contrastive loss formula
    loss = y_true * squared_pred + (1 - y_true) * tf.maximum(margin - y_pred, 0)
    # Compute mean loss
    loss = 0.5 * tf.reduce_mean(loss)
    return loss


def plot_metrics(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


# Input Data
datamanager = DataManager()
datamanager.config(BATCH_SIZE, EPOCHS, BUFFER_SIZE)
datamanager.load_dat_feat("./resource/feature_weights.csv")
datamanager.load_dat_morp("./resource/morph_embeddings.csv")
datamanager.load_dat_main("./resource/train_data2.csv")
datamanager.prepTrainValTest(TRN, VAL, TST, SEED)
# Create a model
in0 = keras.layers.Input(shape=(5,), dtype='float32', name='Feature 0')
in1 = keras.layers.Input(shape=(3,), dtype='float32', name='Feature 1')
in2 = keras.layers.Input(shape=(3,), dtype='float32', name='Feature 2')
in3 = keras.layers.Input(shape=(7,), dtype='float32', name='Feature 3')
in4 = keras.layers.Input(shape=(32,), dtype='float32', name='Feature 4')
in5 = keras.layers.Input(shape=(32,), dtype='float32', name='Feature 5')
merged = keras.layers.Concatenate(axis=1)([in0, in1, in2, in3, in4, in5])
dense1 = keras.layers.Dense(64, activation=keras.activations.gelu)(merged)  # try gelu
dense2 = keras.layers.Dense(32, activation=keras.activations.gelu)(dense1)
drop1 = keras.layers.Dropout(0.2)(dense2)
dense3 = keras.layers.Dense(16, activation=keras.activations.gelu)(drop1)
output = keras.layers.Dense(1, activation=keras.activations.sigmoid)(dense3)
model = keras.models.Model([in0, in1, in2, in3, in4, in5], output)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)  # Using binary cross-entropy loss
# model.compile(optimizer='adam', loss=contrastive_loss, metrics=METRICS)  # Using contrastive loss
# Train the model
history = model.fit(
    datamanager.getTrainDs(),
    epochs=EPOCHS,
    steps_per_epoch=datamanager.step_per_epoch,
    callbacks=[early_stopping],
    validation_data=(datamanager.getValDs()))

hist_test = model.evaluate(datamanager.getTestDs())
#
plot_metrics(history)
print("[{} {}]\n[{} {}]".format(hist_test[3], hist_test[4], hist_test[5], hist_test[6]))
print((hist_test[3] / (hist_test[3] + hist_test[6]) + hist_test[5] / (hist_test[5] + hist_test[4])) / 2)


############################ Adding the cosine similarity column##################### BEN
#cosine similarity function
def row_feature_similarity(row):
    pre = row["pre_feature_weights"]
    post = row["post_feature_weights"]
    return (pre * post).sum() / (np.linalg.norm(pre) * np.linalg.norm(post))

# compute the cosine similarity between the pre- and post- feature weights
data["fw_similarity"] = data.apply(row_feature_similarity, axis=1)

#####################################################################################

#################################################### BENEDICT: You probably should modify this to fit the data
# # Apply PCA to data
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# # Exclude non-numeric and non-categorical columns from the PCA - I guess you don't have any here. So you can remove this part
# exclude_cols = ['ID',
#                 'pre_feature_weights',
#                 'post_feature_weights',
#                 'pre_morph_embeddings',
#                 'post_morph_embeddings',
#                 'projection_group',
#                 'connected',
#                 'is_left',
#                 'is_right',
#                 'is_above',
#                 'is_below'
#                ]


# # Select only numeric columns for PCA
# numeric_columns = [col for col in data.columns if col not in exclude_cols]
# pca_num_of_components = 45 # chosen from explained variance plot. It had more feature though

# # Standardize the data (if needed)
# scaler = StandardScaler()
# data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# # Create a PCA object
# pca = PCA(n_components=pca_num_of_components)

# # Fit and transform the data
# pca_result = pca.fit_transform(data[numeric_columns])


######################################################