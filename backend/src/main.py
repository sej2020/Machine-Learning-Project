from config import settings
from src.automl_rmq_consumer import Consumer
from utils import *

if __name__ == "__main__":
    # main(id)
    consumer = Consumer(settings.RMQ_AUTOML_REQ_IN)
    consumer.consume()
