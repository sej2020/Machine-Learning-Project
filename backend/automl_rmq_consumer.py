import logging
import os
import threading

import pika

from services.rmq_services import process_automlrequest

# logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s  %(name)s  %(levelname)s {%(pathname)s:%(lineno)d}: %(message)s')
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Consumer:
    def __init__(self, queue) -> None:
        threading.Thread.__init__(self)
        rmq_username = os.getenv('rmq_username') or 'guest'
        rmq_password = os.getenv('rmq_password') or 'guest'
        self.credentials = pika.PlainCredentials(rmq_username, rmq_password)
        if os.getenv('rmq_service_name'):
            rmqServiceName = os.getenv('rmq_service_name')
            rmq_host, rmq_port = os.getenv('{}_SERVICE_HOST'.format(rmqServiceName)), os.getenv('{}_SERVICE_PORT'.format(rmqServiceName))
        else:
            rmq_host, rmq_port = os.getenv('rmq_host') or 'localhost', int(os.getenv('rmq_port') or '5672')
        log.info('rmq_url: {}:{}'.format(rmq_host, rmq_port))
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rmq_host, port=rmq_port, credentials=self.credentials))
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.queue = queue

    def getMessage(self, ch, method, properties, body):
        t = threading.Thread(target=process_automlrequest, args=(body,))
        t.start()
        log.info("handed over the execution of request to a thread, acknowledging the message.")


    def consume(self):
        log.info('starting consumer')
        self.channel.queue_declare(queue=self.queue, durable=True)
        self.channel.basic_consume(queue=self.queue, auto_ack=True, on_message_callback=self.getMessage)
        self.channel.start_consuming()
