from src.automl_rmq_consumer import Consumer
from utils import *

def main(id):
    s3_in_buck = S3Service('incoming_data')
    s3_out_buck = S3Service('outgoing_data')
    paramdict = retrieve_params(id, s3_in_buck)
    out_file_path = comparison(**paramdict)
    s3_out_buck.upload_file(out_file_path)
    update_db_w_results(os.path.basename(out_file_path), id)
    path = pathlib.Path(out_file_path)
    path.unlink()
    return

if __name__ == "__main__":
    # main(id)
    consumer = Consumer(settings.RMQ_AUTOML_REQ_IN)
    consumer.consume()
