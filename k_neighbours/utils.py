import collections
import copy
import warnings

import pandas
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

from models import UserInput

warnings.simplefilter(action='ignore', category=FutureWarning)



class RecSys:
    """
        fit - load_data, prepare df
        predict(input)
            - does not change target_df
            - calculate similarity based on df and get k-neighbour
    """
    __n_neighbour: int
    __ranked_list: pandas.DataFrame
    __target_df: pandas.DataFrame

    __target_columns: list[str] = [
        "gender",
        "age",
        "heatlfy",
        "smoke",
        "stress",
        "sleep_well",
        "chronus",
        "wake_up",
        "sleep_time",
        "coffe_near",
        "gourmet",
        "office",
        "home_seater",
        "ill",
    ]

    def __init__(
            self,
            df,
            n_neighbour: int
    ):
        self.__n_neighbour = n_neighbour
        self.df: pd.DataFrame = df

    @staticmethod
    def to_df(user_model: UserInput):
        return pandas.DataFrame(
            dict(
                gender=[user_model.gender],
                age=[user_model.age],
                heatlfy=[user_model.heatlfy],
                smoke=[user_model.smoke],
                stress=[user_model.stress],
                sleep_well=[user_model.sleep_well],
                chronus=[user_model.chronus],
                wake_up=[user_model.wake_up],
                sleep_time=[user_model.sleep_time],
                coffe_near=[user_model.coffe_near],
                gourmet=[user_model.gourmet],
                office=[user_model.office],
                home_seater=[user_model.home_seater],
                ill=[user_model.ill],
            )
        )

    @classmethod
    def prepare_df(cls, df: pd.DataFrame):
        # Prepare enums to number
        df.loc[:, 'gender'] = df['gender'].replace({'Женщина': 0, 'Мужчина': 1})
        df.loc[:, 'chronus'] = df['chronus'].replace({'Сова': 0, 'Жаворонок': 1})
        df.loc[:, 'wake_up'] = df['wake_up'].apply(lambda x: int(x.replace(":", "")))

        binary_columns = [
            "smoke", "sleep_well", "coffe_near", "gourmet", "office", "home_seater", "ill"
        ]
        for c in binary_columns:
            df.loc[:, c] = df[c].replace({'Да': 1, 'Нет': 0})
        df.loc[:, 'chronus'] = df['chronus'].replace({'Сова': 0, 'Жаворонок': 1})

        # Cast values to float
        df = df.astype(float)
        # min-max scale data
        df = cls.normalize_data(df)
        return df

    def fit(self):
        target_df = self.df[self.__target_columns]
        target_df = self.prepare_df(target_df)
        self.__target_df = target_df

    @staticmethod
    def normalize_data(df) -> pandas.DataFrame:
        """Scale columns to 0..1 range"""
        scaler = preprocessing.Normalizer()
        df.iloc[:, :] = scaler.fit_transform(df.iloc[:, :].to_numpy())
        return df

    def predict(self, user_input: UserInput):
        user_df = self.to_df(user_input)
        user_df = self.prepare_df(user_df)
        normalized_user_df = self.normalize_data(user_df)

        user_vector = normalized_user_df.values
        target_vectors = self.__target_df.values

        # find similarity values
        diffs = rbf_kernel(target_vectors, user_vector)
        result_df = pandas.DataFrame(diffs, columns=['similarity_rate'])

        # join asserted results to similarity rates
        joined_result_df = copy.copy(self.df[["drink"]])
        joined_result_df[['similarity_rate']] = result_df

        # sort df by similarity rates
        sorted_list = joined_result_df.sort_values(by="similarity_rate", ascending=False)

        # get top 5 by similarity
        head_neighbours = sorted_list.head(self.__n_neighbour)
        target_values = head_neighbours['drink'].values
        # find the most frequent value
        m = dict(collections.Counter(target_values))
        predicted_answer = max(m, key=m.get)
        return predicted_answer
