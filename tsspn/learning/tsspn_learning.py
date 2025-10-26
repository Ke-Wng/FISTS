import logging
import math

from matplotlib import pyplot as plt
import numpy as np
import ruptures as rpt
from sklearn.cluster import KMeans
from spn.algorithms.splitting.Base import preproc, split_data_by_clusters
from spn.algorithms.splitting.RDC import getIndependentRDCGroups_py
from spn.structure.StatisticalTypes import MetaType

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import euclidean

from tsspn.structure.leaves import IdentityNumericLeaf, Categorical, TimeSeriesLeaf

logger = logging.getLogger(__name__)
MAX_UNIQUE_LEAF_VALUES = 10000


def learn_mspn(
        data,
        ds_context,
        cols="rdc",
        rows="kmeans",
        min_instances_slice=200,
        min_series_slice=200,
        threshold=0.3,
        max_sampling_threshold_cols=10000,
        max_sampling_threshold_rows=100000,
        ohe=False,
        leaves=None,
        memory=None,
        rand_gen=None,
        cpus=-1,
        build_mh=False,
        build_hist=False,
        max_clustering_variance=0.05,
):
    """
    Adapts normal learn_mspn to use custom identity leafs and use sampling for structure learning.
    :param max_sampling_threshold_rows:
    :param max_sampling_threshold_cols:
    :param data:
    :param ds_context:
    :param cols:
    :param rows:
    :param min_instances_slice:
    :param threshold:
    :param ohe:
    :param leaves:
    :param memory:
    :param rand_gen:
    :param cpus:
    :return:
    """
    if leaves is None:
        leaves = create_custom_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    from tsspn.learning.structure_learning import get_next_operation, learn_structure

    def l_mspn(data, ds_context, cols, rows, min_instances_slice, min_series_slice, threshold, ohe, max_clustering_variance=0.05):
        split_cols, split_rows, split_series, split_time_range = get_splitting_functions(max_sampling_threshold_rows, max_sampling_threshold_cols, cols,
                                                         rows, ohe, threshold, rand_gen, cpus, max_clustering_variance)

        node = learn_structure(data, ds_context, split_rows, split_cols, split_series, split_time_range, leaves, min_instances_slice=min_instances_slice, min_series_slice=min_series_slice, build_mh=build_mh, build_hist=build_hist)
        return node

    if memory:
        l_mspn = memory.cache(l_mspn)

    spn = l_mspn(data, ds_context, cols, rows, min_instances_slice, min_series_slice, threshold, ohe, max_clustering_variance)
    return spn


def create_custom_leaf(data, ds_context, scope, cluster_center=None):
    if len(scope) == 2:
        def detect_change_points(time_series, max_cp_num, threshold=0.01):
            time_series = time_series.reshape(-1, 1)
            pelt = rpt.Pelt(model="rbf").fit(time_series)
            change_points = pelt.predict(pen=1)
            change_points[-1] -= 1
            return change_points

            # normalize
            max_metric_value = np.max(time_series)
            min_metric_value = np.min(time_series)
            if max_metric_value == min_metric_value:
                time_series = np.zeros_like(time_series)
            else:
                time_series = (time_series - min_metric_value) / (max_metric_value - min_metric_value)

            change_points = []
            for i in range(1, len(time_series)):
                change = np.abs(time_series[i] - time_series[i-1])
                if change > threshold:
                    change_points.append(i)
            return change_points
            
        # timestamps = []
        # last_ts = data[0, 1]
        # for i in range(data.shape[0]):
        #     if data[i, 1] < last_ts:
        #         break
        #     timestamps.append(data[i, 1])
        #     last_ts = data[i, 1]

        # if cluster_center is None:
        #     time_series = transform_data_to_time_series(data[:, 0], data.shape[0] // len(timestamps), len(timestamps))
        #     cluster_center = np.mean(time_series, axis=0)

        # # plt.figure(figsize=(10, 6))
        # # for i in range(time_series.shape[0]):
        # #     plt.plot(range(time_series.shape[1]), time_series[i, :, 0])
        # # plt.plot(range(mean_time_series.shape[0]), mean_time_series.ravel(), 'k--')
        # # plt.xlabel('Timestamp')
        # # plt.ylabel('Metric Value')
        # # plt.title('Time Series Distribution')
        # # plt.savefig('time_series_distribution.png')
        # # plt.close()

        # metric_bins_num = 10
        # change_points = detect_change_points(cluster_center, len(cluster_center) // metric_bins_num)
        # time_bins, metric_bins_dict, probs_dict = [], {}, {}
        # start_idx = 0
        # segment_idx = 0
        # total_points = data.shape[0]
        # for cp in change_points:
        #     segment = time_series[:, start_idx:cp, :]
        #     time_bins.append(timestamps[cp])

        #     segment = segment.reshape(-1, 1)
        #     sorted_metric_values = np.sort(segment)
        #     metric_bins = np.percentile(sorted_metric_values, np.linspace(0, 100, metric_bins_num+1))
        #     metric_bins = np.unique(metric_bins)
        #     metric_bins_dict[segment_idx] = metric_bins
        #     hist, _ = np.histogram(segment, bins=metric_bins, density=False)
        #     probs_dict[segment_idx] = hist / total_points

        #     start_idx = cp
        #     segment_idx += 1

        # segment = time_series[:, start_idx:, :]
        # time_bins.append(timestamps[-1])

        # segment = segment.reshape(-1, 1)
        # sorted_metric_values = np.sort(segment)
        # metric_bins = np.percentile(sorted_metric_values, np.linspace(0, 100, metric_bins_num+1))
        # metric_bins = np.unique(metric_bins)
        # metric_bins_dict[segment_idx] = metric_bins
        # hist, _ = np.histogram(segment, bins=metric_bins, density=False)
        # probs_dict[segment_idx] = hist / total_points

        # leaf = TimeSeriesLeaf(time_bins, metric_bins_dict, probs_dict, scope, cardinality=data.shape[0])
        # return leaf

        # leaf = TimeSeriesLeaf(np.array(timestamps), mean_time_series.ravel(), scope, cardinality=data.shape[0])
        # return leaf

        leaf = TimeSeriesLeaf(data, scope, cardinality=data.shape[0])
        return leaf

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]

    if meta_type == MetaType.REAL:
        assert len(scope) == 1, "scope for more than one variable?"

        unique_vals, counts = np.unique(data[:, 0], return_counts=True)

        if hasattr(ds_context, 'no_compression_scopes') and idx not in ds_context.no_compression_scopes and \
                len(unique_vals) > MAX_UNIQUE_LEAF_VALUES:
            # if there are too many unique values build identity leaf with histogram representatives
            hist, bin_edges = np.histogram(data[:, 0], bins=MAX_UNIQUE_LEAF_VALUES, density=False)
            logger.debug(f"\t\tDue to histograms leaf size was reduced "
                         f"by {(1 - float(MAX_UNIQUE_LEAF_VALUES) / len(unique_vals)) * 100:.2f}%")
            unique_vals = bin_edges[:-1]
            probs = hist / data.shape[0]
            lidx = len(probs) - 1

            assert len(probs) == len(unique_vals)

        else:
            probs = np.array(counts, np.float64) / len(data[:, 0])
            lidx = len(probs) - 1

        null_value = ds_context.null_values[idx]
        leaf = IdentityNumericLeaf(unique_vals, probs, null_value, scope, cardinality=data.shape[0])

        return leaf

    elif meta_type == MetaType.DISCRETE:
        unique, counts = np.unique(data[:, 0], return_counts=True)
        if unique.shape[0] < len(ds_context.domains[idx]) // 2:
            p = counts / data.shape[0]
            null_value = ds_context.null_values[idx]
            node = Categorical(p, null_value, scope, unique=unique, cardinality=data.shape[0])
        else:
            # +1 because of potential 0 value that might not occur
            sorted_counts = np.zeros(len(ds_context.domains[idx]) + 1, dtype=np.float64)
            for i, x in enumerate(unique):
                sorted_counts[int(x)] = counts[i]
            p = sorted_counts / data.shape[0]
            null_value = ds_context.null_values[idx]
            node = Categorical(p, null_value, scope, cardinality=data.shape[0])

        return node


def get_splitting_functions(max_sampling_threshold_rows, max_sampling_threshold_cols, cols, rows, ohe, threshold,
                            rand_gen, n_jobs, max_clustering_variance=0.05):
    from spn.algorithms.splitting.Clustering import get_split_rows_TSNE, get_split_rows_GMM
    from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
    from spn.algorithms.splitting.RDC import get_split_rows_RDC_py

    if isinstance(cols, str):

        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(max_sampling_threshold_cols=max_sampling_threshold_cols,
                                               threshold=threshold,
                                               rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif cols == "poisson":
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):

        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans(max_sampling_threshold_rows=max_sampling_threshold_rows)
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows, get_split_rows_TimeSeries(max_variance=max_clustering_variance), get_split_rows_TimeRange()

# noinspection PyPep8Naming
def get_split_rows_KMeans(max_sampling_threshold_rows, n_clusters=2, pre_proc=None, ohe=False, seed=17):
    # noinspection PyPep8Naming
    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        if data.shape[0] > max_sampling_threshold_rows:
            data_sample = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]

            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit(data_sample).predict(data)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit_predict(data)

        cluster_centers = kmeans.cluster_centers_
        result = split_data_by_clusters(local_data, clusters, scope, rows=True)

        return result, cluster_centers.tolist()

    return split_rows_KMeans


def pre_process_series(time_series, window_size=10):
    # smooth
    smoothed_series = np.zeros_like(time_series)
    from scipy.ndimage import uniform_filter1d
    for i in range(time_series.shape[0]):
        smoothed_series[i] = uniform_filter1d(time_series[i].flatten(), size=window_size).reshape(-1, 1)
    
    # normalize
    max_metric_value = np.max(smoothed_series)
    min_metric_value = np.min(smoothed_series)
    if max_metric_value == min_metric_value:
        normalized_series = np.zeros_like(smoothed_series)
    else:
        normalized_series = (smoothed_series - min_metric_value) / (max_metric_value - min_metric_value)
    
    return normalized_series


def transform_data_to_time_series(field_data, n_series, n_timestamps):
    time_series = np.zeros((n_series, n_timestamps, 1))
    for i in range(n_series):
        field_data_per_series = field_data[i * n_timestamps:(i + 1) * n_timestamps]
        time_series[i, :, 0] = field_data_per_series
    return time_series


def downsample_time_series(time_series, window_size):
    if window_size == 1:
        return time_series
    downsampled = []
    for series in time_series:
        series = series.flatten()
        new_length = len(series) // window_size
        downsampled_series = []
        for i in range(new_length):
            window = series[i * window_size : (i + 1) * window_size]
            downsampled_series.append(np.mean(window))
        downsampled.append(downsampled_series)
    return np.array(downsampled)[:, :, np.newaxis]
        

def find_optimal_n_segments(field_data, n_series, n_timestamps, sample_ratio=0.3, downsample_granularity=10, min_points_per_segment=50):
    time_series = transform_data_to_time_series(field_data, n_series, n_timestamps)
    n_series = time_series.shape[0]
    n_ponints = time_series.shape[1]
    sampled_series = time_series
    # sampled_indices = np.random.choice(n_series, int(n_series * sample_ratio), replace=False)
    # sampled_series = time_series[sampled_indices]
    # sampled_series = downsample_time_series(sampled_series, downsample_granularity)

    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(range(sampled_series.shape[1]), sampled_series[i, :, 0])
    plt.xlabel('Timestamp')
    plt.ylabel('Metric Value')
    plt.title('Time Series Distribution')
    plt.savefig('time_series_distribution.png')
    plt.close()

    possible_segments = np.linspace(1, sampled_series.shape[1], 10, dtype=int)
    avg_n_clusters = []
    for k in possible_segments:
        total_n_clusters = 0
        n_ponints_per_segment = sampled_series.shape[1] // k
        for i in range(k):
            time_series_segment = sampled_series[:, i * n_ponints_per_segment : (i + 1) * n_ponints_per_segment, :]
            clusters = time_series_clustering(time_series_segment)
            unique_clusters = np.unique(clusters)
            total_n_clusters += len(unique_clusters)
        avg_n_clusters.append(total_n_clusters / k / sample_ratio)

    def sigmoid(k, init, final, k0, a):
        return init + (final - init) * (1 / (1 + np.exp(-a * (k - k0))))

    params = [avg_n_clusters[0], avg_n_clusters[-1], 3, 0.8]
    from scipy.optimize import curve_fit
    try:
        params, _ = curve_fit(sigmoid, possible_segments, np.array(avg_n_clusters), p0=params)
    except RuntimeError as e:
        print(f"Warning: {e}")
    init, final, k0, a = params

    # plt.figure(figsize=(10, 6))
    # plt.plot(possible_segments, sigmoid(possible_segments, init, final, k0, a))
    # plt.scatter(possible_segments, avg_n_clusters)
    # plt.xlabel('k')
    # plt.ylabel('average clusters per segment')
    # plt.savefig('sigmoid.png')
    # plt.close()

    def NClusters(k):
        C_k = sigmoid(k, init, final, k0, a)
        return k * C_k
    max_segments = math.ceil(n_ponints / min_points_per_segment)
    k_values = np.arange(1, max_segments + 1)
    nc_values = np.array([NClusters(k) for k in k_values])
    min_k = k_values[np.argmin(nc_values)]
    min_clusters = np.min(nc_values)

    # plt.figure(figsize=(10, 6))
    # plt.plot(k_values, nc_values)
    # plt.xlabel('k')
    # plt.ylabel('total number of clusters')
    # plt.savefig('nclusters.png')
    # plt.close()

    return int(min_k)

    # def Storage(k):
    #     C_k = sigmoid(k, init, final, k0, a)
    #     return k * C_k * n_tag + 2 * n_ponints * 24 * C_k
    # max_segments = math.ceil(n_ponints / min_points_per_segment)
    # k_values = np.arange(1, max_segments + 1)
    # S_values = np.array([Storage(k) for k in k_values])
    # min_k = k_values[np.argmin(S_values)]
    # min_Storage = np.min(S_values)

    # plt.figure(figsize=(10, 6))
    # plt.plot(k_values, S_values)
    # plt.xlabel('k')
    # plt.ylabel('storage overhead')
    # plt.savefig('storage.png')
    # plt.close()

    # return int(min_k)


def time_series_clustering(time_series, min_cluster_size=1, max_variance=0.10, seed=17):
    
    class Counter:
        def __init__(self):
            self.value = 0
        def get_counter(self):
            val = self.value
            self.value += 1
            return val

    counter = Counter()
    def recursive_clustering(ts_data, current_indices, current_clusters, distance_threshold):
        n_series = ts_data.shape[0]
        n_points = ts_data.shape[1]
        
        if n_series <= min_cluster_size:
            current_clusters[current_indices] = counter.get_counter()
            return
        
        # if n_series * n_timestamps > max_sampling_threshold_rows:
        #     window_size = n_timestamps / (max_sampling_threshold_rows / n_series)
        #     window_size = max(1, int(np.ceil(window_size)))
        #     ts_data_sample = downsample_time_series(ts_data, window_size)
        # else:
        #     ts_data_sample = ts_data
        ts_data_sample = ts_data
        
        tskmeans = TimeSeriesKMeans(n_clusters=2, metric="euclidean", random_state=seed)
        cluster_labels = tskmeans.fit_predict(ts_data_sample)

        # center_distance = euclidean(
        #     kmeans.cluster_centers_[0].ravel(), 
        #     kmeans.cluster_centers_[1].ravel()
        # )

        # if center_distance <= max_distance:
        #     current_clusters[current_indices] = counter.value
        #     counter.value += 1
        #     return

        cluster_0_indices = current_indices[cluster_labels == 0]
        cluster_0 = ts_data[cluster_labels == 0]
        distances_0 = np.linalg.norm(cluster_0 - tskmeans.cluster_centers_[0], axis=1)
        avg_distance_0 = np.mean(distances_0)
        if avg_distance_0 <= distance_threshold:
            current_clusters[cluster_0_indices] = counter.get_counter()

            # if len(cluster_0_indices) > 1:
            #     plt.figure(figsize=(10, 6))
            #     for i in range(cluster_0.shape[0]):
            #         plt.plot(range(cluster_0.shape[1]), cluster_0[i, :, 0])
            #     plt.plot(np.arange(cluster_0.shape[1]), tskmeans.cluster_centers_[0, :, 0], 'k--')
            #     plt.xlabel('Timestamp')
            #     plt.ylabel('Metric Value')
            #     plt.title('Time Series Distribution')
            #     plt.savefig('time_series_distribution.png')
            #     plt.close()
        else:
            recursive_clustering(
                cluster_0,
                cluster_0_indices,
                current_clusters,
                distance_threshold
            )

        cluster_1_indices = current_indices[cluster_labels == 1]
        cluster_1 = ts_data[cluster_labels == 1]
        distances_1 = np.linalg.norm(cluster_1 - tskmeans.cluster_centers_[1], axis=1)
        avg_distance_1 = np.mean(distances_1)
        if avg_distance_1 <= distance_threshold:
            current_clusters[cluster_1_indices] = counter.get_counter()
        else:
            recursive_clustering(
                cluster_1,
                cluster_1_indices,
                current_clusters,
                distance_threshold
            )

    n_total_series = time_series.shape[0]
    current_clusters = np.zeros(n_total_series, dtype=int)
    current_indices = np.arange(n_total_series)

    time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
    max_variance_value = (np.max(time_series) - np.min(time_series)) * max_variance
    distance_threshold = euclidean(
        np.zeros(time_series.shape[1]),
        np.ones(time_series.shape[1]) * max_variance_value
    )
    recursive_clustering(time_series, current_indices, current_clusters, distance_threshold)
    return current_clusters


def get_split_rows_TimeSeries(max_variance=0.05, n_clusters=2, pre_proc=None, ohe=False, seed=17):

    def split_rows_TimeSeries(local_data, ds_context, scope, n_timestamps):
        # return [(local_data, n_timestamps, 1)], None
        data = preproc(local_data, ds_context, pre_proc, ohe)

        assert ds_context.field_idx in scope, "field column not found in scope"
        assert ds_context.timestamp_idx in scope, "timestamp column not found in scope"

        n_series = local_data.shape[0] // n_timestamps
        field_data = data[:, scope.index(ds_context.field_idx)]
        time_series = transform_data_to_time_series(field_data, n_series, n_timestamps)
        assert time_series.shape == (n_series, n_timestamps, 1)
        time_series = downsample_time_series(time_series, 6)

        # clusters = time_series_clustering(time_series, min_cluster_size=ds_context.n_series * 0.1, max_variance=0.05)

        # time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
        cluster_center = np.mean(time_series, axis=0) #TODO: pass cluster center as parameter
        distance_threshold = np.linalg.norm(cluster_center * max_variance)
        avg_distance = np.mean(np.linalg.norm(time_series - cluster_center, axis=1))
        if avg_distance <= distance_threshold:
            return [(local_data, n_timestamps, 1)], None

        tskmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=seed)
        cluster_labels = tskmeans.fit_predict(time_series)

        clusters = np.repeat(cluster_labels, n_timestamps)
        data_slices = split_data_by_clusters(data, clusters, scope, rows=True)
        for i in range(len(data_slices)):
            data_slices[i] = (data_slices[i][0], n_timestamps, data_slices[i][2])
        return data_slices, tskmeans.cluster_centers_.tolist()

    return split_rows_TimeSeries


def get_split_rows_by_one_column(min_instances_slices):
    # group by the column with minimum # of unique values
    def split_rows_by_one_column(data, scope):
        num_rows, num_cols = data.shape
        target_col_idx = None
        breaks = None

        for col_idx in range(num_cols):
            unique_values = np.unique(data[:, col_idx])
            if breaks is None or len(unique_values) < len(breaks):
                breaks = unique_values
                target_col_idx = col_idx
        
        logging.debug(f"\t\tsplit rows by scope [{scope[target_col_idx]}]")

        # if len(breaks) > num_rows / min_instances_slices:
        #     if len(scope) > 2:
        #         return [(data, scope, 1)]
        #     # _, breaks = np.histogram(data[:, target_col_idx], bins=int(num_rows / min_instances_slices))
        #     quantiles = np.linspace(0, 1, int(num_rows / min_instances_slices) + 1 + 1)  # 计算分位数
        #     breaks = np.quantile(data[:, target_col_idx], quantiles)  # 计算边界

        breaks[0] -= 1e-6
        breaks[-1] += 1e-6

        clusters = np.digitize(data[:, target_col_idx], breaks)
        result = split_data_by_clusters(data, clusters, scope, rows=True)

        return result
    return split_rows_by_one_column


def get_split_rows_TimeRange(n_segments=2):
    def split_rows_TimeRange(local_data, ds_context, n_timestamps):
        if n_segments == 1:
            return [local_data]
        all_segments = []
        n_series = local_data.shape[0] // n_timestamps
        n_timestamps_per_segment = n_timestamps // n_segments
        for i in range(n_segments):
            segments = []
            segment_start_idx = i * n_timestamps_per_segment
            segment_end_idx = (i + 1) * n_timestamps_per_segment
            for j in range(n_series):
                series_start_idx = j * n_timestamps
                segment = local_data[series_start_idx + segment_start_idx:series_start_idx + segment_end_idx, :]
                segments.append(segment)
            all_segments.append((np.concatenate(segments, axis=0), n_timestamps_per_segment, n_timestamps_per_segment / n_timestamps))
        return all_segments

    return split_rows_TimeRange


# noinspection PyPep8Naming
def get_split_cols_RDC_py(max_sampling_threshold_cols=10000, threshold=0.3, ohe=True, k=10, s=1 / 6,
                          non_linearity=np.sin,
                          n_jobs=-2, rand_gen=None):
    from spn.algorithms.splitting.RDC import split_data_by_clusters

    def split_cols_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        if ds_context.field_idx in scope and ds_context.timestamp_idx in scope:
            clusters = np.zeros(local_data.shape[1])
            clusters[scope.index(ds_context.field_idx)] = 1
            clusters[scope.index(ds_context.timestamp_idx)] = 1
            return split_data_by_clusters(local_data, clusters, scope, rows=False)

        if local_data.shape[0] > max_sampling_threshold_cols:
            local_data_sample = local_data[np.random.randint(local_data.shape[0], size=max_sampling_threshold_cols), :]
            clusters = getIndependentRDCGroups_py(
                local_data_sample,
                threshold,
                meta_types,
                domains,
                k=k,
                s=s,
                # ohe=True,
                non_linearity=non_linearity,
                n_jobs=n_jobs,
                rand_gen=rand_gen,
            )
            return split_data_by_clusters(local_data, clusters, scope, rows=False)
        else:
            clusters = getIndependentRDCGroups_py(
                local_data,
                threshold,
                meta_types,
                domains,
                k=k,
                s=s,
                # ohe=True,
                non_linearity=non_linearity,
                n_jobs=n_jobs,
                rand_gen=rand_gen,
            )
            return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py
