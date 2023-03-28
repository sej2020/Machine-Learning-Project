import csv
import os.path

from config import settings
from services.s3Service import S3Service

def download_results_file(results_filename):
    s3Service = S3Service(settings.S3_RESULTS_BUCKET)
    temp_download_path = f"{settings.TEMP_DOWNLOAD_DIR}/{results_filename}"
    if os.path.exists(temp_download_path):
        os.remove(temp_download_path)
    s3Service.download_file(results_filename, temp_download_path)
    return temp_download_path

def get_boxplot_data(results_filename):
    result_filepath = download_results_file(results_filename)
    headers = ['regressorName', 'metricName', 'metricValue']
    visualization_data = [headers]
    metrics_list = set()
    result_file = open(result_filepath, 'r')
    csv_file = csv.DictReader(result_file)
    for row in csv_file:
        for k, v in row.items():
            if len(k) == 0:
                continue
            k_split = k.split("-")
            visualization_data.append([k_split[0], k_split[1], float(v)])
            metrics_list.add(k_split[1])
    return visualization_data, list(metrics_list)

def get_lineplot_data_cv(results_filename):
    result_filepath = download_results_file(results_filename)
    result_file = open(result_filepath, 'r')
    csv_file = csv.DictReader(result_file)

    visualization_data = {}
    num_rows = 0
    for row in csv_file:
        num_rows += 1
        for k, v in row.items():
            if len(k) == 0:
                continue
            k_split = k.split("-")
            regressorName, metricName = k_split[0], k_split[1]
            if metricName not in visualization_data:
                visualization_data[metricName] = {}
            if regressorName not in visualization_data[metricName]:
                visualization_data[metricName][regressorName] = []
            visualization_data[metricName][regressorName].append(float(v))
    visualization_data['num_cv_folds'] = num_rows
    return visualization_data
