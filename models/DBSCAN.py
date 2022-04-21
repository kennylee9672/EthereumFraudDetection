from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class DBSCAN_Model():
    def __init__(self, df):
        self.df = df
        self.preprocess_data()
        self.model_name = 'DBSCAN'
        self.n_clusters = 4

    def update_dataframe(self, df):
        self.df = df
        self.preprocess_data()

    def get_model_name(self):
        return self.model_name

    def preprocess_data(self):
        # QuickFix, Need to update it once data cleaning and preparation step completes
        self.df.fillna(self.df.median(), inplace=True)

    def set_n_clusters(self, n):
        self.n_clusters = n

    def run(self):
        nearest_neighbors = NearestNeighbors(n_neighbors=self.n_clusters)
        nearest_neighbors.fit(self.df)
        distances, indices = nearest_neighbors.kneighbors(self.df)
        distances = np.sort(distances, axis=0)[:, 1]

        plt.plot(distances[9811:9841])
        plt.title('Knee Method', fontsize=20)
        plt.xlabel('# of Points')
        plt.ylabel('average distance to neighbors')
        plt.show()

        eps_range = distances[9821:9841]

        for eps in eps_range:
            db = DBSCAN(eps=eps, min_samples=4).fit(self.df)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            print('EPS %s' % eps)
            print('Estimated number of clusters: %d' % n_clusters)
            print('Estimated number of Noise Points: %d' % n_noise)
