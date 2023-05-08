import json
import logging
import os
import threading

import pika


from configuration.config import settings
from db.models.auto_ml_request import AutoMLRequestRepository
from services.s3Service import S3Service
from utils import comparison_wrapper

# logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s  %(name)s  %(levelname)s {%(pathname)s:%(lineno)d}: %(message)s')
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Consumer:
    def __init__(self, queue) -> None:
        threading.Thread.__init__(self)
        rmq_username = os.getenv('rmq_username') or 'guest'
        rmq_password = os.getenv('rmq_password') or 'guest'
        log.info('rmq_username: {}, rmq_password: {}'.format(rmq_username, rmq_password))
        self.credentials = pika.PlainCredentials(rmq_username, rmq_password)
        if os.getenv('rmq_service_name'):
            log.info("pointing to kubernetes cluster")
            rmqServiceName = os.getenv('rmq_service_name')
            rmq_host, rmq_port = os.getenv('{}_SERVICE_HOST'.format(rmqServiceName)), os.getenv('{}_SERVICE_PORT'.format(rmqServiceName))
        else:
            log.info("is this from log? pointing to local")
            rmq_host, rmq_port = os.getenv('rmq_host') or 'localhost', int(os.getenv('rmq_port') or '5672')
        log.info('rmq_url: {}:{}'.format(rmq_host, rmq_port))
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rmq_host, port=rmq_port, credentials=self.credentials))
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=10)
        self.queue = queue

    def getMessage(self, ch, method, properties, body):
        request_id = None
        try:
            payload = json.loads(body.decode('utf8'))
            request_id = payload['id']
            s3_service = S3Service(settings.S3_DATA_BUCKET)
            s3_service.download_file(payload['datapath'], f"{settings.TEMP_DOWNLOAD_DIR}/{payload['datapath']}")
            log.info(f'successfully downloaded file from s3 for {request_id}')
            payload['datapath'] = f"{settings.TEMP_DOWNLOAD_DIR}/{payload['datapath']}"
            setting = payload['setting']
            del payload['setting']
            comparison_result = comparison_wrapper(setting, payload)
            log.info(f'completed running automl pipeline for {request_id}')
            result_file_list = [comparison_result['output_path'],
                                f'{settings.TEMP_UPLOAD_DIR}/perf_stats_Accuracy_over_Various_Proportions_of_Training_Set_{request_id}.csv',
                                f'{settings.TEMP_UPLOAD_DIR}/perf_stats_Error_by_Datapoint_{request_id}.csv']
            s3_service = S3Service(settings.S3_RESULTS_BUCKET)
            result_s3_key_list = []
            for i, result_file in enumerate(result_file_list):
                print(i, result_file)
                result_s3_key = f"{payload['id']}_result_{i}.csv"
                result_s3_key_list.append(result_s3_key)
                s3_service.upload_file(result_file, result_s3_key)
            AutoMLRequestRepository.update_request(request_id, status=1, result_file=','.join(result_s3_key_list))
        except Exception as e:
            if request_id is not None:
                AutoMLRequestRepository.update_request(request_id, status=-1)
            log.error(f'error while processing message: {e}')

    def consume(self):
        log.info('starting consumer')
        self.channel.queue_declare(queue=self.queue, durable=True)
        self.channel.basic_consume(queue=self.queue, auto_ack=True, on_message_callback=self.getMessage)
        self.channel.start_consuming()
