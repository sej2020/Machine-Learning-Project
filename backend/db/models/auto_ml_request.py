import uuid

import sqlalchemy
from sqlalchemy import Column, String, Float, Integer

from db.models.base_model import Base
from db.session import get_db_actual

class AutoMLRequest(Base):
    id = Column(String, primary_key=True, index=True)
    datafile = Column(String)
    regressor_list = Column(String)
    email = Column(String)
    status = Column(Integer)
    metrics = Column(String, default='RMSE, MSE, MAE, R2')
    metric_score_method = Column(String, default='RMSE')
    test_set_size = Column(Float, default=0.2)
    num_cv_folds = Column(Integer, default=10)
    resultfile = Column(String)
    data_visualization = Column(String)
    default_setting = Column(Integer, default=1)

class AutoMLRequestRepository:
    db_session = get_db_actual()

    @staticmethod
    def convert_list_to_str(input_list, separator=','):
        trimmed_list = []
        for ip in input_list:
            trimmed_list.append(str(ip).strip())
        return separator.join(trimmed_list)

    @staticmethod
    def add_request(request_id, datafile, regressor_list, email, metrics, metric_score_method, test_set_size, num_cv_folds, default_setting):
        regressor_list_str = AutoMLRequestRepository.convert_list_to_str(regressor_list)
        metrics_str = AutoMLRequestRepository.convert_list_to_str(metrics)
        automl_request = AutoMLRequest(id=request_id, datafile=datafile, regressor_list=regressor_list_str, email=email, status=0,
                                       metrics=metrics_str, metric_score_method=metric_score_method, test_set_size=test_set_size,
                                       num_cv_folds=num_cv_folds, default_setting=default_setting, data_visualization="")
        try:
            AutoMLRequestRepository.db_session.add(automl_request)
            AutoMLRequestRepository.db_session.commit()
            return automl_request
        except Exception as e:
            error_message = f'error while inserting to database: {str(e)}'
            raise Exception(error_message)

    @staticmethod
    def get_request_by_id(request_id) -> AutoMLRequest:
        query_result = AutoMLRequestRepository.db_session.query(AutoMLRequest).filter(AutoMLRequest.id == request_id).all()
        if len(query_result) == 0:
            error_message = f'request details not found for {request_id}'
            raise Exception(error_message)
        return query_result[0]

    @staticmethod
    def update_request(request_id, status, data_visualization_response=None, result_file=None):
        update_params = {'status': status}
        if status == 1:
            update_params['resultfile'] = result_file
            update_params['data_visualization'] = data_visualization_response
        AutoMLRequestRepository.db_session.query(AutoMLRequest).filter(AutoMLRequest.id == request_id).update(update_params)
        AutoMLRequestRepository.db_session.commit()

