import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from util import basic_func
import seaborn as sns

BATCH_SIZE = 32
EPOCHS = 1000
BUFFER_SIZE = 1000
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
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
TRN_TST_RATIO = 0.8

# activate GPU accelerator
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # tf.config.experimental.set_memory_growth(gpus[0], True)  # Set memory growth to minimize GPU memory usage
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Use the first GPU

# input preparation -- should make a class to manage this
# read in data
data = pd.read_csv("./resource/train_data2.csv")

# load in additional features for each neuron
feature_weights = pd.read_csv("./resource/feature_weights.csv")
morph_embeddings = pd.read_csv("./resource/morph_embeddings.csv")

# join all feature_weights
feature_weights["feature_weights"] = (
    feature_weights.filter(regex="feature_weight_")
    .sort_index(axis=1)
    .apply(lambda x: np.array(x), axis=1)
)
# delete the feature_weight_i columns
feature_weights.drop(
    feature_weights.filter(regex="feature_weight_").columns, axis=1, inplace=True
)

# join all morph_embed_i columns into a single np.array column
morph_embeddings["morph_embeddings"] = (
    morph_embeddings.filter(regex="morph_emb_")
    .sort_index(axis=1)
    .apply(lambda x: np.array(x), axis=1)
)
# delete the morph_embed_i columns
morph_embeddings.drop(
    morph_embeddings.filter(regex="morph_emb_").columns, axis=1, inplace=True
)
data = (
    data.merge(
        feature_weights.rename(columns=lambda x: "pre_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
    .merge(
        feature_weights.rename(columns=lambda x: "post_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
    .merge(
        morph_embeddings.rename(columns=lambda x: "pre_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
    .merge(
        morph_embeddings.rename(columns=lambda x: "post_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
)

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


# split data to train and test set
train_df, test_df = train_test_split(data, train_size=TRN_TST_RATIO, random_state=0)



def process_df(raw_df):
    # data processing  -- will need a handler for this
    indat1 = np.copy(raw_df[['adp_dist', 'pre_oracle', 'post_oracle', 'pre_skeletal_distance_to_soma',
                           'post_skeletal_distance_to_soma']].values)
    raw = raw_df.pre_brain_area.values
    indat2 = basic_func.one_hot_convert(raw)

    raw = raw_df.post_brain_area.values
    indat3 = basic_func.one_hot_convert(raw)

    raw = raw_df.compartment.values
    indat4 = basic_func.one_hot_convert(raw)

    raw = raw_df.connected.values
    labels = basic_func.one_hot_convert(raw)[:, 1]

    # Normalization
    # transform bulk data
    arr = np.transpose(indat1)
    arr[0] = arr[0] / 500  # scale down large value
    arr[1] = arr[1] * 10  # upscale
    arr[2] = arr[2] * 10  # upscale
    arr[3] = arr[3] / 250000  # scale down large value
    arr[4] = arr[4] / 180000  # scale down large value`
    # sns.boxplot(indat1)
    # return (np.reshape(indat1, (len(indat1), 5, 1)),
    #         np.reshape(indat2, (len(indat1), 3, 1),),
    #         np.reshape(indat3, (len(indat1), 3, 1)),
    #         np.reshape(indat4, (len(indat1), 7, 1))), labels
    return  (indat1, indat2, indat3, indat4), labels


# process and obtain data for train and test
f_trn, l_trn = process_df(train_df)
f_tst, l_tst = process_df(test_df)

pos_sample = np.where(l_trn == 1)
neg_sample = np.where(l_trn == 0)


# compile data and ready for tf
def make_ds(features_list, labels, index):
    """
    :param features_list: [feature1, feature2, ...]
    :param labels: output label
    :param index: position for this dataset
    :return: tf.dataset
    """
    used_features = {}
    for c, feature in enumerate(features_list):
        used_features["Feature {}".format(c)] = feature[index]
    ds = tf.data.Dataset.from_tensor_slices((used_features, labels[index]))
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds


def make_ds_test(features_list, labels):
    used_features = {}
    for c, feature in enumerate(features_list):
        used_features["Feature {}".format(c)] = feature
    ds = tf.data.Dataset.from_tensor_slices((used_features, labels))
    ds = ds.batch(BATCH_SIZE).prefetch(2)
    return ds


pos_ds = make_ds(f_trn, l_trn, np.where(l_trn == 1))
neg_ds = make_ds(f_trn, l_trn, np.where(l_trn == 0))

resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

resampled_steps_per_epoch = np.ceil(2.0*len(neg_sample)/BATCH_SIZE)

# loss fun:
# need constrative learning loss func
# WIP

# Create a model
in1 = keras.layers.Input(shape=(5,), dtype='float32', name='Feature 0')
in2 = keras.layers.Input(shape=(3,), dtype='float32', name='Feature 1')
in3 = keras.layers.Input(shape=(3,), dtype='float32', name='Feature 2')
in4 = keras.layers.Input(shape=(7,), dtype='float32', name='Feature 3')
merged = keras.layers.Concatenate(axis=1)([in1, in2, in3, in4])
dense1 = keras.layers.Dense(12, activation=keras.activations.relu)(merged)      # try gelu
dense2 = keras.layers.Dense(12, activation=keras.activations.relu)(dense1)
dense3 = keras.layers.Dense(6, activation=keras.activations.relu)(dense2)
output = keras.layers.Dense(1, activation=keras.activations.sigmoid)(dense3)
model = keras.models.Model([in1, in2, in3, in4], output)

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)  # Using binary cross-entropy loss

# Train the model
history = model.fit(
    resampled_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    steps_per_epoch=resampled_steps_per_epoch)

# validation
val_ds = make_ds_test(f_tst, l_tst)

history_val = model.evaluate(val_ds)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    # plt.plot(history.epoch, history.history['val_'+metric],
    #          color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()


plot_metrics(history)
print(history_val.history)


