from automl_rmq_consumer import Consumer
from utils import *

if __name__ == "__main__":
    consumer = Consumer(settings.RMQ_AUTOML_REQ_IN)
    consumer.consume()
