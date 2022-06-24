import argparse
import pandas as pd
import numpy as np


class MF:
    def __init__(self, rating_matrix):
        self.rating_array = np.array(rating_matrix.values)
        self.dim_of_latent = 200
        self.epochs = 150
        self.learning_rate = 0.0052
        self.reg_param = 0.052
        self.num_of_users, self.num_of_movies = rating_matrix.shape

        self.user_latent_matrix = np.random.normal(size=(self.num_of_users, self.dim_of_latent))
        self.movie_latent_matrix = np.random.normal(size=(self.num_of_movies, self.dim_of_latent))

        self.user_bias = np.zeros(self.num_of_users, np.double)
        self.movie_bias = np.zeros(self.num_of_movies, np.double)
        self.bias = np.mean(self.rating_array[np.where(self.rating_array != 0)])

        self.user_id_idx_dict = {}
        self.movie_id_idx_dict = {}
        for user_idx, user_id in enumerate(rating_matrix.index):
            self.user_id_idx_dict[user_id] = user_idx
        for movie_idx, movie_id in enumerate(rating_matrix.columns):
            self.movie_id_idx_dict[movie_id] = movie_idx

    # squared error(실제 평점과 예측 평점의 차이)와 함께
    # 과적합을 방지하는 정규화 사용하여
    # 학습을 진행한다.
    # 모델이 학습을 진행함에 따라 파라미터 값이 점점 커지는데 이는 overfitting을 발생시킬 수 있다.
    # 따라서 학습 파라미터들이 너무 커지지 않도록 규제한다.
    # 여기에 bias를 적용하는데
    # 이는 사용자 별로 영화에 평점을 매길 때 편향성이 있을 수 있기 때문이다.
    # 따라서 사용자가 가진 편향성과 영화 자체의 편향성, 모든 평점의 평균을 더해 학습을 진행한다.

    def optimize(self, i, j):
        real_rate = self.rating_array[i][j]
        calculated_rate = np.dot(self.user_latent_matrix[i, :], self.movie_latent_matrix[j, :].T)
        bias = self.bias + self.user_bias[i] + self.movie_bias[j]

        # 이 error가 0에 가까워져야 한다
        error = real_rate - calculated_rate - bias

        # bias 업데이트
        self.user_bias[i] += self.learning_rate * (error - self.reg_param * self.user_bias[i])
        self.movie_bias[j] += self.learning_rate * (error - self.reg_param * self.movie_bias[j])

        d_user = (error * self.movie_latent_matrix[j, :]) - (self.reg_param * self.user_latent_matrix[i, :])
        d_movie = (error * self.user_latent_matrix[i, :]) - (self.reg_param * self.movie_latent_matrix[j, :])
        self.user_latent_matrix[i, :] += self.learning_rate * d_user
        self.movie_latent_matrix[j, :] += self.learning_rate * d_movie

    def train(self):
        for epoch in range(self.epochs):
            print(f'{epoch + 1}/{self.epochs}')
            for user_idx in range(self.num_of_users):
                for movie_idx in range(self.num_of_movies):
                    if self.rating_array[user_idx][movie_idx] > 0:
                        self.optimize(user_idx, movie_idx)

    def test(self, test_data, max_val, min_val):
        test_user_id_list = test_data[:, 0]
        test_movie_id_list = test_data[:, 1]
        calculated_rates = self.bias + np.expand_dims(self.user_bias, -1) + np.expand_dims(self.movie_bias, 0) + np.dot(
            self.user_latent_matrix, self.movie_latent_matrix.T)

        result_rate_list = []
        for user_id, movie_id in zip(test_user_id_list, test_movie_id_list):
            if movie_id not in self.movie_id_idx_dict:
                if user_id not in self.user_id_idx_dict:
                    result_rate_list.append(self.bias)
                else:
                    result_rate_list.append(self.bias + self.user_bias[self.user_id_idx_dict[user_id]])
            elif user_id not in self.user_id_idx_dict:
                result_rate_list.append(self.bias + self.movie_bias[self.movie_id_idx_dict[movie_id]])
            else:
                user_idx = self.user_id_idx_dict[user_id]
                movie_idx = self.movie_id_idx_dict[movie_id]
                rate = round(max(min_val, calculated_rates[user_idx][movie_idx]), 4)
                rate = round(min(rate, max_val), 4)
                result_rate_list.append(rate)

        return np.array(result_rate_list)


def read_file(train_file_name, test_file_name):
    header = ['user_id', 'movie_id', 'rating', 'time_stamp']
    train_data_frame = pd.read_csv(train_file_name, sep='\t', names=header)
    test_data_frame = pd.read_csv(test_file_name, sep='\t', names=header)

    train_data_frame.drop('time_stamp', axis=1, inplace=True)
    user_movie_rating = train_data_frame.pivot_table('rating', index='user_id', columns='movie_id').fillna(0)

    test_data_frame.drop('time_stamp', axis=1, inplace=True)
    test_data = np.array(test_data_frame)

    max_value = max(train_data_frame['rating'].unique())
    min_value = min(train_data_frame['rating'].unique())

    return user_movie_rating, test_data, max_value, min_value


def main(train_file_name, test_file_name):
    rating_matrix, test_data, max_val, min_val = read_file(train_file_name, test_file_name)
    matrix_factorization = MF(rating_matrix)
    matrix_factorization.train()
    prediction = matrix_factorization.test(test_data, max_val, min_val)
    test_user = test_data[:, 0]
    test_movie = test_data[:, 1]

    with open(train_file_name + '_prediction.txt', 'w') as file:
        for u, i, r in zip(test_user, test_movie, prediction):
            file.write(str(u) + '\t' + str(i) + '\t' + str(r) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_filename",
                        help="train data file path, if it is in same directory only file name",
                        type=str)
    parser.add_argument("test_filename",
                        help="test data file path, if it is in same directory only file name",
                        type=str)
    args = parser.parse_args()
    main(args.train_filename, args.test_filename)
