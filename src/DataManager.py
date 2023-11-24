import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataManager:
    """
    Handle DNN dataset and apply modification needed
    """
    def __init__(self):
        self._dat_main = None
        self._dat_lboard = None
        self._dat_feat = None
        self._dat_morp = None
        self.BATCH_SIZE = 0
        self.EPOCHS = 0
        self.BUFFER_SIZE = 0
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self.step_per_epoch = None
        self.lea_ds = None  # wip. just a bunch of feature for now

    def config(self, batch_size: int, epochs: int, buffer_size: int):
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.BUFFER_SIZE = buffer_size

    def load_dat_feat(self, fname: str):
        self._dat_feat = pd.read_csv(fname)
        # join all morph_embed_i columns into a single np.array column
        self._dat_feat["feature_weights"] = (
            self._dat_feat.filter(regex="feature_weight_")
            .sort_index(axis=1)
            .apply(lambda x: np.array(x), axis=1)
        )
        self._dat_feat.drop(self._dat_feat.filter(regex="feature_weight_").columns, axis=1, inplace=True)

    def load_dat_morp(self, fname: str):
        self._dat_morp = pd.read_csv(fname)
        self._dat_morp["morph_embeddings"] = (
            self._dat_morp.filter(regex="morph_emb_")
            .sort_index(axis=1)
            .apply(lambda x: np.array(x), axis=1)
        )
        self._dat_morp.drop(self._dat_morp.filter(regex="morph_emb_").columns, axis=1, inplace=True)

    def load_dat_main(self, fname: str):
        if not self.dataIsLoadedDat():
            raise Exception("please load morp and feature data first")
        self._dat_main = pd.read_csv(fname)

        self._dat_main = (
            self._dat_main.merge(
                self._dat_feat.rename(columns=lambda x: "pre_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
            .merge(
                self._dat_feat.rename(columns=lambda x: "post_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
            .merge(
                self._dat_morp.rename(columns=lambda x: "pre_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
            .merge(
                self._dat_morp.rename(columns=lambda x: "post_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
        )

        dm1, dm2 = len(self._dat_main['pre_morph_embeddings'][0]), len(self._dat_main['post_morph_embeddings'][0])
        for i, r in self._dat_main.iterrows():
            if isinstance(r['pre_morph_embeddings'], float):
                self._dat_main.at[i, 'pre_morph_embeddings'] = np.full(dm1, 0.001)
            if isinstance(r['post_morph_embeddings'], float):
                self._dat_main.at[i, 'post_morph_embeddings'] = np.full(dm2, 0.001)

    def load_dat_lboard(self, fname: str):
        if not self.dataIsLoadedDat():
            raise Exception("please load morp and feature data first")

        self._dat_lboard = pd.read_csv(fname)

        self._dat_lboard = (
            self._dat_lboard.merge(
                self._dat_feat.rename(columns=lambda x: "pre_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
            .merge(
                self._dat_feat.rename(columns=lambda x: "post_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
            .merge(
                self._dat_morp.rename(columns=lambda x: "pre_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
            .merge(
                self._dat_morp.rename(columns=lambda x: "post_" + x),
                how="left",
                validate="m:1",
                copy=False,
            )
        )

        dm1, dm2 = len(self._dat_lboard['pre_morph_embeddings'][0]), len(self._dat_lboard['post_morph_embeddings'][0])
        for i, r in self._dat_lboard.iterrows():
            if isinstance(r['pre_morph_embeddings'], float):
                self._dat_lboard.at[i, 'pre_morph_embeddings'] = np.full(dm1, 0.001)
            if isinstance(r['post_morph_embeddings'], float):
                self._dat_lboard.at[i, 'post_morph_embeddings'] = np.full(dm2, 0.001)

    def dataIsLoadedDat(self):
        return not (self._dat_morp is None or self._dat_feat is None)

    def prepTrainValTest(self, w_trn, w_val, w_tst, random_state=0):
        # split data
        tmp = w_tst/(w_trn+w_val+w_tst)
        dm_df, test_df = train_test_split(self._dat_main, test_size=tmp, random_state=random_state)
        tmp = w_val/(w_val+w_trn)
        train_df, val_df = train_test_split(dm_df, test_size=tmp, random_state=random_state)

        # process
        f_trn, l_trn = DataManager.processFromDf(train_df)
        f_val, l_val = DataManager.processFromDf(val_df)
        f_tst, l_tst = DataManager.processFromDf(test_df)

        # split positive and negative sample and make them
        pos_ds = self.makeDs(f_trn, l_trn, np.where(l_trn == 1))
        neg_ds = self.makeDs(f_trn, l_trn, np.where(l_trn == 0))
        resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[.5, .5])

        self._train_ds = resampled_ds.batch(self.BATCH_SIZE).prefetch(2)
        # self._train_ds = self.makeDs(f_trn, l_trn)  # test without rebalancing
        # self._train_ds = pos_ds.batch(self.BATCH_SIZE).prefetch(2)  # train with just positive to offset the balance
        self.step_per_epoch = np.ceil(2.0 * (len(l_trn) - np.count_nonzero(l_trn)) / self.BATCH_SIZE)

        self._val_ds = self.makeDs(f_val, l_val)
        self._test_ds = self.makeDs(f_tst, l_tst)

    def loadLeader(self):
        self.lea_ds, _ = self.processFromDf(self._dat_lboard, skip_y=1)

    def getTrainDs(self):
        return self._train_ds
    
    def getValDs(self):
        return self._val_ds
    
    def getTestDs(self):
        return self._test_ds

    def getLeadDS(self):
        return self.lea_ds

    @staticmethod
    def oneHotConverter(raw):
        """
        convert a 1D array to one-hot-coding
        :param raw: an array
        :return: 2D matrix
        """
        cls = np.unique(raw)
        out = np.zeros([len(raw), len(cls)])
        cls_look_up = {item: index for index, item in enumerate(cls)}  # convert array to dict
        for c, i in enumerate(raw):
            out[c, cls_look_up[i]] = 1  # mark target code to 1
        return out

    @staticmethod
    def cosin_feature_similarity(df, feat1, feat2):
        # perform cosin similarity on 2 feature and return array
        pre = np.vstack(df[feat1].values)
        post = np.vstack(df[feat2].values)
        out = np.zeros(len(pre))
        for c, _ in enumerate(out):
            out[c] = np.dot(pre[c], post[c])/(np.linalg.norm(pre[c]) * np.linalg.norm(post[c]))
        return out

    @staticmethod
    def processFromDf(raw_df, skip_y=0):
        raw_df['fweight_cosin'] = DataManager.cosin_feature_similarity(
            raw_df, 'pre_feature_weights', 'post_feature_weights')
        raw_df['morp_cosin'] = DataManager.cosin_feature_similarity(
            raw_df, 'pre_morph_embeddings', 'post_morph_embeddings')

        indat1 = np.copy(raw_df[['adp_dist', 'pre_oracle', 'post_oracle', 'pre_skeletal_distance_to_soma',
                                 'post_skeletal_distance_to_soma', 'pre_test_score', 'post_test_score', 'fweight_cosin',
                                 'morp_cosin']].values)
        # brain area
        raw = raw_df.pre_brain_area.values
        indat2 = DataManager.oneHotConverter(raw)
        raw = raw_df.post_brain_area.values
        indat3 = DataManager.oneHotConverter(raw)
        # compartment
        raw = raw_df.compartment.values
        indat4 = DataManager.oneHotConverter(raw)

        # morph embedding
        indat5 = np.vstack(raw_df.pre_morph_embeddings.values)
        indat6 = np.vstack(raw_df.post_morph_embeddings.values)

        # feature weight
        indat7 = np.stack(raw_df.pre_feature_weights)
        indat8 = np.stack(raw_df.post_feature_weights)

        labels = None
        if not skip_y:
            raw = raw_df.connected.values
            labels = DataManager.oneHotConverter(raw)[:, 1]

        # Normalization
        # transform bulk data  -- try forcing to magnitude of 10
        arr = np.transpose(indat1)
        arr[0] = arr[0] / 5000  # scale down large value
        # arr[1] = arr[1] * 10  # upscale
        # arr[2] = arr[2] * 10  # upscale
        arr[3] = arr[3] / 2000000  # scale down large value
        arr[4] = arr[4] / 2000000  # scale down large value`
        return (indat1, indat2, indat3, indat4, indat5, indat6), labels

    def makeDs(self, features_list, labels, index=None):
        """
        :param features_list: [feature1, feature2, ...]
        :param labels: output label
        :param index: position for this dataset -- training need index to split pos and neg sample for rabalancing, 
        val and test does not. I know it's wrong in every shape and form, but I'm lazy
        :return: tf.dataset
        """
        used_features = {}
        ds = None
        if index is None:
            for c, feature in enumerate(features_list):
                used_features["Feature {}".format(c)] = feature
            ds = tf.data.Dataset.from_tensor_slices((used_features, labels)).cache()
            ds = ds.batch(self.BATCH_SIZE).prefetch(2)
        else:
            for c, feature in enumerate(features_list):
                used_features["Feature {}".format(c)] = feature[index]
            ds = tf.data.Dataset.from_tensor_slices((used_features, labels[index]))
            ds = ds.shuffle(self.BUFFER_SIZE).repeat()
        return ds
