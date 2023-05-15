import logging

import pika
import json
import os

from configuration.config import settings
from db.models.auto_ml_request import AutoMLRequest, AutoMLRequestRepository
from services import data_services
from services.data_services import visualize_data
from services.s3Service import S3Service
from utils import comparison_wrapper

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Publisher:
    def __init__(self) -> None:
        self.credentials = pika.PlainCredentials(os.getenv('rmq_user') or 'guest', os.getenv('rmq_password') or 'guest')
        log.info("connecting to local rabbitmq")
        self.rmq_host, self.rmq_port = os.getenv('rmq_host') or 'localhost', int(os.getenv('rmq_port') or '5672')
        log.info('rmq_host: {} and rmq_port: {}'.format(self.rmq_host, self.rmq_port))
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.rmq_host, port=self.rmq_port, credentials=self.credentials))
        self.connection.channel().exchange_declare(exchange='autoMLExchange', exchange_type='direct', durable=True)

    def publish(self, queue, body):
        if self.connection.is_closed:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.rmq_host, port=self.rmq_port, credentials=self.credentials))
        self.connection.channel().basic_publish(exchange='', routing_key=queue, body=json.dumps(body),
                                                properties=pika.BasicProperties(delivery_mode=1))
        log.info('message sent to queue: {}'.format(queue))

def convert_str_to_list(input_str, separator=','):
    return input_str.split(separator)

def send_create_request_message(automl_create_request: AutoMLRequest):
    regressor_map = {}
    req_regressor_list = convert_str_to_list(automl_create_request.regressor_list)
    for regressor in settings.REGRESSOR_LIST:
        regressor_map[regressor] = 1 if regressor in req_regressor_list else 0
    data = {
        'id':               automl_create_request.id,
        'which_regressors': regressor_map,
        'metric_list':      convert_str_to_list(automl_create_request.metrics),
        'n_vizualized_tb':  0,
        'test_set_size':    automl_create_request.test_set_size,
        'n_cv_folds':       automl_create_request.num_cv_folds,
        'score_method':     automl_create_request.metric_score_method,
        'datapath':         automl_create_request.datafile,
        'n_workers':        1,
        'figure_lst':       settings.VISUALIZATION_LIST,
        'setting':          automl_create_request.default_setting
        }
    rmq_producer = Publisher()
    rmq_producer.publish(settings.RMQ_AUTOML_REQ_IN, data)

def process_automlrequest(body):
    request_id = None
    try:
        payload = json.loads(body.decode('utf8'))
        request_id = payload['id']
        s3_service = S3Service(settings.S3_DATA_BUCKET)
        s3_service.download_file(payload['datapath'], f"{settings.TEMP_DOWNLOAD_DIR}{settings.PATH_SEPARATOR}{payload['datapath']}")
        log.info(f'successfully downloaded file from s3 for {request_id}')
        payload['datapath'] = f"{settings.TEMP_DOWNLOAD_DIR}{settings.PATH_SEPARATOR}{payload['datapath']}"
        setting = payload['setting']
        del payload['setting']
        comparison_result = comparison_wrapper(setting, payload)
        log.info(f'completed running automl pipeline for {request_id}')
        result_file_list = [comparison_result['output_path'],
                            f'{settings.TEMP_UPLOAD_DIR}{settings.PATH_SEPARATOR}perf_stats_Accuracy_over_Various_Proportions_of_Training_Set_{request_id}.csv',
                            f'{settings.TEMP_UPLOAD_DIR}{settings.PATH_SEPARATOR}perf_stats_Error_by_Datapoint_{request_id}.csv']
        s3_service = S3Service(settings.S3_RESULTS_BUCKET)
        result_s3_key_list = []
        for i, result_file in enumerate(result_file_list):
            result_s3_key = f"{payload['id']}_result_{i}.csv"
            result_s3_key_list.append(result_s3_key)
            s3_service.upload_file(result_file, result_s3_key)
        log.info(f"creating data visualization for request {request_id}")
        data_visualization_response = generate_data_visualization_response(payload['datapath'], result_file_list)
        log.info(f"completed processing request {request_id}, updating database")
        AutoMLRequestRepository.update_request(request_id, 1, data_visualization_response, result_file=','.join(result_s3_key_list))
        log.info(f"successfully updated {request_id} in database")
    except Exception as e:
        if request_id is not None:
            AutoMLRequestRepository.update_request(request_id, status=-1)
        log.error(f'error while processing message: {e}')

def generate_data_visualization_response(data_path, result_file_list):
    response = {'coloring_data': data_services.get_coloring(result_file_list[2])}
    for dimensionality in [2, 3]:
        response[dimensionality] = {}
        for dim_red_algo in ['tsne', 'pca']:
            dimensions = data_services.visualize_data(data_path, n_components=int(dimensionality), algorithm=dim_red_algo)
            response[dimensionality][dim_red_algo] = {}
            for i in range(len(dimensions)):
                response[dimensionality][dim_red_algo][f'dimension{i}'] = dimensions[i]
    jsonResponse = json.dumps(response)
    return jsonResponse
