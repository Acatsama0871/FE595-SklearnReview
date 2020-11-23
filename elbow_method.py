# elbow_method.py
# use the elbow method to verify the 3 is the correct choice of populations to
# use for Wine and Iris dataset

# module
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris
from sklearn.cluster import KMeans


def main():
    # load dataset
    # wine dataset
    data_wine, _ = load_wine(return_X_y=True)
    # iris dataset
    data_iris, _ = load_iris(return_X_y=True)

    # clustering for different num of populations
    seed = 777  # set random seed
    wine_ssd = []  # sum of squared distance
    iris_ssd = []
    # for wine dataset
    for i in range(1, 11):
        cur_cluster = KMeans(
            n_clusters=i, init="random", random_state=seed
        ).fit(data_wine)
        wine_ssd.append(cur_cluster.inertia_)
    # for iris dataset
    for i in range(1, 11):
        cur_cluster = KMeans(
            n_clusters=i, init="random", random_state=seed
        ).fit(data_iris)
        iris_ssd.append(cur_cluster.inertia_)

    # plot result
    # wine dataset
    plt.plot(range(1, 11), wine_ssd, marker="o")
    plt.title("Elbow Heuristic: wine dataset")
    plt.xlabel("Num of populations")
    plt.ylabel("sum of squared distance")
    plt.show()
    # iris dataset
    plt.plot(range(1, 11), iris_ssd, marker="o")
    plt.title("Elbow Heuristic: Iris dataset")
    plt.xlabel("Num of populations")
    plt.ylabel("sum of squared distance")
    plt.show()

    # print result
    print(
        '''
        From the plots, we can see the decline rates of sum of squared
        distances begin to decrease after the 3 populations. So, the 3
        populations would be appropriate.
        ''')


if __name__ == "__main__":
    main()
