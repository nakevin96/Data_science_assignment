import argparse
import os.path

import pandas as pd
import math
import numpy as np
from typing import Final

NOISE: Final = -1


# - DBSCAN 알고리즘
# - 무작위로 point를 하나 선택한다
# - Eps와 MinPts값을 이용하여 density reachable한 모든 포인트를 탐색한다
# - 선택한 point가 core라면 cluster가 형성된다.
# - 선택한 point가 border이고 density reachable한 point가 없다면 다음 point탐색으로 넘어간다
# - 모든 포인트에 대해 위 과정을 수행한다

class DBSCAN:
    def __init__(self, input_data_frame, n_cluster, Eps, MinPts):
        self.input_data = np.array(input_data_frame.values)
        self.num_of_cluster = n_cluster
        self.Eps = Eps
        self.MinPts = MinPts
        self.point_list = []
        self.cluster_label = 0

        for data in self.input_data:
            tmp = Point(int(data[0]), float(data[1]), float(data[2]))
            self.point_list.append(tmp)

    def make_neighbor_list(self, core):
        return [point for point in self.point_list if point != core and core.calculate_distance(point) <= self.Eps]

    def retrieve_cluster(self, neighbor_list, label):
        for neighbor in neighbor_list:
            if neighbor.label is None:
                neighbor.label = label
                other_neighbor_list = self.make_neighbor_list(neighbor)
                if self.MinPts <= len(other_neighbor_list):
                    neighbor_list.extend(other_neighbor_list)
            elif neighbor.label == NOISE:
                neighbor.label = label

    def make_cluster_list(self):
        cluster_list = [[] for _ in range(0, self.cluster_label)]
        for point in self.point_list:
            if point.label == NOISE:
                continue
            cluster_list[point.label].append(point.object_id)

        cluster_list.sort(key=len)
        cluster_list.reverse()
        return cluster_list[:self.num_of_cluster]

    def clustering(self):
        for point in self.point_list:
            # label이 없는 point에 대해 labeling
            if point.label is None:
                neighbors = self.make_neighbor_list(point)
                if self.MinPts > len(neighbors):
                    point.label = NOISE
                    continue

                point.label = self.cluster_label
                self.retrieve_cluster(neighbors, self.cluster_label)
                self.cluster_label += 1
            else:
                continue


class Point:
    def __init__(self, object_id, x_pos, y_pos):
        self.label = None
        self.object_id = object_id
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __ne__(self, other):
        return self.object_id != other.object_id

    def calculate_distance(self, point):
        return math.sqrt(math.pow(self.x_pos - point.x_pos, 2) + math.pow(self.y_pos - point.y_pos, 2))


def read_file(input_filename):
    header = ['object_id', 'x_coord', 'y_coord']
    input_data_frame = pd.read_csv(input_filename, sep='\t', names=header)
    return input_data_frame


def create_directory(directory_name):
    try:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    except OSError:
        print("Error: Failed to create the directory.")


def main(input_filename, n, eps, min_pts):
    input_data_frame = read_file(input_filename)
    dbscan = DBSCAN(input_data_frame, n, eps, min_pts)
    dbscan.clustering()
    clusters = dbscan.make_cluster_list()

    directory_name = input_filename.replace('.txt', '') + '_result'
    create_directory(directory_name)
    for i, cluster in enumerate(clusters):
        output_file_name = input_filename.replace('.txt', '') + f"_cluster_{i}.txt"

        with open("./" + directory_name + "/" + output_file_name, "w") as output_file:
            for object_id in cluster:
                output_file.write(f"{object_id}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_filename",
                        help="input data file's name (not clustered)",
                        type=str)
    parser.add_argument("num_of_cluster",
                        help="number of clusters for the corresponding input data",
                        type=int)
    parser.add_argument("max_radius",
                        help="maximum radius of the neighborhood",
                        type=float)
    parser.add_argument("min_num_of_points",
                        help="minimum number of points in an Eps-neighborhood of a given data",
                        type=float)

    args = parser.parse_args()
    main(args.input_data_filename, args.num_of_cluster, args.max_radius, args.min_num_of_points)
