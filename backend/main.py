import logging
import os.path
import uuid

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from fastapi import FastAPI, Depends

from config import settings
from data_models.automl import AutoMLCreateRequest, AutoMLCreateResponse, AutoMLCreateResponseContents
from db.base import Base
from db.models.auto_ml_request import AutoMLRequestRepository
from db.session import engine
from fastapi import File, UploadFile

from services.rmq_services import Publisher, send_create_request_message
from services.s3Service import S3Service
from fastapi.middleware.cors import CORSMiddleware
# from db.base import create_tables

app = FastAPI()
origins = [
        "http://localhost:4200"
        ]
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )
def create_tables():
    logging.info('creating all database tables')
    print('am i coming here?')
    Base.metadata.create_all(bind=engine)

@app.get("/")
async def root():
    return {"message": "Hello World"}

def generate_request_id():
    return str(uuid.uuid4())

def save_file(file_contents, filepath):
    file = open(filepath, 'wb')
    file.write(file_contents)
    file.close()

def delete_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

@app.post('/createAutoMLRequest', response_model_exclude_none=True)
async def create_automl_request(form_data: AutoMLCreateRequest = Depends(), file_data: UploadFile = File(...)):
    request_id = generate_request_id()
    s3Service = S3Service(settings.S3_DATA_BUCKET)
    print(settings.S3_DATA_BUCKET)
    s3_key = f'{request_id}_{file_data.filename}'
    temp_path = f'{settings.TEMP_UPLOAD_DIR}/{file_data.filename}'
    save_file(file_data.file.read(), temp_path)
    try:
        s3Service.upload_file(temp_path, s3_key)
        automl_request = AutoMLRequestRepository.add_request(request_id, s3_key, form_data.regressor_list, form_data.email, form_data.metrics,
                                                             form_data.metric_score_method, form_data.test_set_size, form_data.num_cv_folds)
        send_create_request_message(automl_request)
        data = AutoMLCreateResponseContents(request_id=request_id, request_status=0, estimated_time_completion=300)
        response = jsonable_encoder(AutoMLCreateResponse(error=None, data=data))
        return JSONResponse(content=response, status_code=201)
    except Exception as e:
        logging.error(e)
        data = AutoMLCreateResponseContents(request_id=request_id, request_status=-1, estimated_time_completion=0)
        response = jsonable_encoder(AutoMLCreateResponse(error=str(e), data=data))
        return JSONResponse(content=response, status_code=500)
    finally:
        delete_file(temp_path)

@app.get('/getAutoMLRequest', response_model_exclude_none=True)
async def get_automl_request(request_id):
    try:
        automl_request = AutoMLRequestRepository.get_request_by_id(request_id)
        data = AutoMLCreateResponseContents(request_id=request_id, request_status=automl_request.status, estimated_time_completion=0)
        if automl_request.status == 1:
            data.result_link = f'http://192.168.1.216:8081/results?key={automl_request.resultfile}'
        response = jsonable_encoder(AutoMLCreateResponse(error=None, data=data))
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        data = AutoMLCreateResponseContents(request_id=request_id, request_status=-1, estimated_time_completion=0)
        response = jsonable_encoder(AutoMLCreateResponse(error=str(e), data=data))
        return JSONResponse(content=response, status_code=404)

@app.get('/results')
async def download_results(key):
    s3_service = S3Service(settings.S3_RESULTS_BUCKET)
    temp_download_path = f"{settings.TEMP_DOWNLOAD_DIR}/{key}"
    s3_service.download_file(key, temp_download_path)
    return FileResponse(path=temp_download_path, filename=key)


if __name__ == "__main__":
    create_tables()
    uvicorn.run(app, host="0.0.0.0", port=8081)
