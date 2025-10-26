import argparse
import ast
import csv
import logging
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tsspn')))

import numpy as np
import pandas as pd

from config import QUERIES_DIR_PATH, RESULTS_DIR_PATH
from schema import ALL_TABLE_SCHEMAS
from tsspn.tsspn_estimator import TSSPNEstimator

parser = argparse.ArgumentParser()
parser.add_argument("--no-hot", action="store_true", help="Disable hotness tracking")
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def calculate_qerror(pred_card, real_card):
    if real_card == 0 and pred_card == 0:
        return 1
    elif real_card == 0 or pred_card == 0:
        return max(real_card, pred_card)
    else:
        return max(pred_card, real_card) / min(pred_card, real_card)

class QueryLoader:
    def __init__(self, query_file):
        self.query_df = pd.read_csv(query_file)
        self.query_df['tagk_set'].apply(ast.literal_eval)
        self.query_df['qtime'] = pd.to_datetime(self.query_df['qtime'], unit='s')
        self.index = 0
        self.start_time = time.time()
        self.base_qtime = self.query_df.iloc[0]['qtime']
        self.time_offset = 0

    def next_query(self, skip_wait=False, max_wait_time=10.0):
        if self.index >= len(self.query_df):
            return None

        current_query = self.query_df.iloc[self.index]
        current_qtime = current_query['qtime']
        delay = (current_qtime - self.base_qtime).total_seconds()

        if not skip_wait:
            now = time.time() + self.time_offset
            elapsed = now - self.start_time
            wait_time = delay - elapsed
            if wait_time > 0:
                sleep_time = min(wait_time, max_wait_time)
                time.sleep(sleep_time)
                self.time_offset += wait_time - sleep_time
        else:
            # 跳过等待：将内部时钟直接拨到当前查询对应的时间
            self.start_time = time.time() - delay

        self.index += 1
        return current_query

    def n_queries(self):
        return len(self.query_df)


class HotnessTracker:
    def __init__(self, tag_sets, update_rate=10, min_overlap_rate=0.8, inherit_factor=0.3):
        self.current_batch_counts = {tag: 0 for tag in tag_sets}
        self.accumulated_hotness = {tag: 0.0 for tag in tag_sets}
        self.prev_hot_tags = set()
        
        self.update_rate = update_rate
        self.min_overlap_rate = min_overlap_rate
        self.inherit_factor = inherit_factor
        
        self.track_count = 0

    def track(self, query_tagk_set):
        for tag in query_tagk_set:
            if tag in self.current_batch_counts:
                self.current_batch_counts[tag] += 1
        
        self.track_count += 1
        
        if self.track_count >= self.update_rate:
            return self._update_hotness()
        return None
            
    def _update_hotness(self):
        self.accumulated_hotness = {
            tag: (1 - self.inherit_factor) * self.current_batch_counts[tag] + self.inherit_factor * self.accumulated_hotness[tag]
            for tag in self.accumulated_hotness
        }
        
        self.current_batch_counts = {tag: 0 for tag in self.current_batch_counts}
        self.track_count = 0

        cur_hot_tags = self.get_hot_tags()
        overlap_rate = len(cur_hot_tags.intersection(self.prev_hot_tags)) / len(cur_hot_tags)
        prev_hot_tags = self.prev_hot_tags
        self.prev_hot_tags = cur_hot_tags
        if overlap_rate >= 0.9:
            self.inherit_factor = min(0.5, self.inherit_factor + 0.1)
        elif overlap_rate <= 0.6:
            self.inherit_factor = max(0.1, self.inherit_factor - 0.1)

        if overlap_rate < self.min_overlap_rate:
            return cur_hot_tags
        return None
        
    def get_hot_tags(self):
        return {
            tag 
            for tag, hotness in self.accumulated_hotness.items() 
            if hotness >= 1
        }


def report_bottleneck(results):
    def print_section(title, index, value_name):
        print("=" * 100)
        print(f"Top 5 queries by {value_name}:")
        print("-" * 100)
        print(f"{'Query No.':<10} | {title:<12} | SQL")
        print("-" * 100)
        top_items = sorted(results, key=lambda x: x[index], reverse=True)[:5]
        for r in top_items:
            print(f"{r[0]:<10} | {r[index]:<12.4f} | {r[1][r[1].find('WHERE'):]}")
        print()

    print_section("infer_time", 5, "infer_time")
    print_section("points_qerror", 4, "points_qerror")
    print_section("series_qerror", 8, "series_qerror")

def evaluate_tsspn(query_file, model_path, schema, result_path, sync=True):
    results = []
    points_qerrors = []
    points_total_infer_time = 0
    series_qerrors = []
    series_total_infer_time = 0
    query_loader = QueryLoader(query_file)
    total_queries_num = query_loader.n_queries()

    estimator = TSSPNEstimator(schema, model_path, overwrite=True)
    hotness_tracker = HotnessTracker(estimator.get_tags())

    # Used for evaluating tsspn's max runtime storage
    # tags={"site", "ip", "scrape_endpoint", "app_name", "namespace", "instance", "node_sn", "pod", "node_app_group", "cluster"}
    # estimator.update_hot_tags(tags, sync=True, min_hot_tags_num=1, max_hot_tags_num=100)
    # size = estimator.save("./tsspn.pkl")
    # print(f"Model size: {size / 1024:.2f}KB")
    # return

    for i in range(total_queries_num):
        query = query_loader.next_query(skip_wait=not estimator.is_updating())
        assert query is not None

        query_no, sql, real_points, real_series, tagk_set = query.query_no, query.query, query.real_points_cardinality, query.real_series_cardinality, ast.literal_eval(query.tagk_set)
        pred_points, pred_series, infer_time = estimator.estimate_cardinality(sql)
        points_qerror = calculate_qerror(pred_points, real_points)
        series_qerror = calculate_qerror(pred_series, real_series)
        points_qerrors.append(points_qerror)
        series_qerrors.append(series_qerror)
        points_total_infer_time += infer_time
        series_total_infer_time += infer_time
        results.append((query_no, sql, real_points, pred_points, points_qerror, infer_time, real_series, pred_series, series_qerror, infer_time, estimator.last_query_miss))

        if len(estimator.hot_tags_scope) == 0 or not args.no_hot:
            # Track hotness if there are no hot tags yet or hotness tracking is enabled
            hot_tags = hotness_tracker.track(tagk_set)
            if hot_tags is not None:
                estimator.update_hot_tags(hot_tags, sync=sync)

        if query_no % 100 == 0:
            print(f"Evaluting query {query_no}, remaining {total_queries_num - query_no} queries ...")

    save_tsspn_eval_results(result_path, results)

    return results, points_qerrors, points_total_infer_time, series_qerrors, series_total_infer_time, estimator.updating_cnt, estimator.updating_time, total_queries_num, estimator.miss_cnt

def save_tsspn_eval_results(result_path, results):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["query_no", "query", "real_points", "pred_points", "points_qerror", "points_infer_time", "real_series", "pred_series", "series_qerror", "series_infer_time", "miss"])
        for result in results:
            csv_writer.writerow(result)
        
def main():
    for schema in ALL_TABLE_SCHEMAS:
        points_results_dict = {
            "estimator": [],
            "median": [],
            "90th": [],
            "95th": [],
            "99th": [],
            "max": [],
            "mean": [],
            "infer_time": [],
        }
        series_results_dict = {
            "estimator": [],
            "median": [],
            "90th": [],
            "95th": [],
            "99th": [],
            "max": [],
            "mean": [],
            "infer_time": [],
        }

        query_file = os.path.join(QUERIES_DIR_PATH, f"{schema.table_name}_realworld.csv")
        model_path = f"./models/tsspn/{schema.table_name}_0.80data_updated.pkl"
        result_file_name = f"{schema.table_name}_realworld_0.80data_updated"
        if args.no_hot:
            result_file_name += "_no_hot"
        result_file = os.path.join(RESULTS_DIR_PATH, "tsspn", f"{result_file_name}.csv")
        results, points_qerrors, points_total_infer_time, series_qerrors, series_total_infer_time, updating_cnt, updating_time, total_queries_num, miss_cnt = \
            evaluate_tsspn(query_file, model_path, schema, result_file, sync=False)

        points_results_dict["estimator"].append("tsspn")
        points_results_dict["median"].append(np.median(points_qerrors))
        points_results_dict["90th"].append(np.round(np.percentile(points_qerrors, 90), 2))
        points_results_dict["95th"].append(np.round(np.percentile(points_qerrors, 95), 2))
        points_results_dict["99th"].append(np.round(np.percentile(points_qerrors, 99), 2))
        points_results_dict["max"].append(np.round(np.max(points_qerrors), 2))
        points_results_dict["mean"].append(np.round(np.mean(points_qerrors), 2))
        points_results_dict["infer_time"].append(np.round(points_total_infer_time * 1000 / total_queries_num, 2))

        series_results_dict["estimator"].append("tsspn")
        series_results_dict["median"].append(np.round(np.median(series_qerrors), 2))
        series_results_dict["90th"].append(np.round(np.percentile(series_qerrors, 90), 2))
        series_results_dict["95th"].append(np.round(np.percentile(series_qerrors, 95), 2))
        series_results_dict["99th"].append(np.round(np.percentile(series_qerrors, 99), 2))
        series_results_dict["max"].append(np.round(np.max(series_qerrors), 2))
        series_results_dict["mean"].append(np.round(np.mean(series_qerrors), 2))
        series_results_dict["infer_time"].append(np.round(series_total_infer_time * 1000 / total_queries_num, 2))
        
        print()
        print(f"Results for {schema.table_name} ...")
        print(f"Points results:")
        print(pd.DataFrame(points_results_dict))
        print(f"Series results:")
        print(pd.DataFrame(series_results_dict))

        print()
        print(f"executed {updating_cnt} times updates, {updating_time / updating_cnt} secs/update")
        print(f"miss rate: {miss_cnt / total_queries_num}")
        print()

        report_bottleneck(results)


if __name__ == "__main__":
    main()