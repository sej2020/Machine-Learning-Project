import logging

import pika
import json
import os

from config import settings
from db.models.auto_ml_request import AutoMLRequest

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
        'id': automl_create_request.id,
        'which_regressors': regressor_map,
        'metric_list': convert_str_to_list(automl_create_request.metrics),
        'n_vizualized_tb': 0,
        'test_set_size': automl_create_request.test_set_size,
        'n_cv_folds': automl_create_request.num_cv_folds,
        'score_method': automl_create_request.metric_score_method,
        'datapath': automl_create_request.datafile,
        'n_workers': 1,
        'figure_lst': settings.VISUALIZATION_LIST
        }
    rmq_producer = Publisher()
    rmq_producer.publish(settings.RMQ_AUTOML_REQ_IN, data)
