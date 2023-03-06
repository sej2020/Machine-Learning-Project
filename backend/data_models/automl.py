from typing import List, Union

from pydantic import BaseModel

from db.models.auto_ml_request import AutoMLRequest

class AutoMLCreateRequest(BaseModel):
    # id: str
    email: str
    regressor_list: List[str]
    metrics: List[str]
    metric_score_method: Union[str, None] = 'Root Mean Squared Error'
    test_set_size: Union[float, None] = 0.2
    num_cv_folds: Union[int, None] = 10

class AutoMLCreateResponseContents(BaseModel):
    request_id: str
    request_status: str
    estimated_time_completion: int
    result_link: Union[str, None]
class AutoMLCreateResponse(BaseModel):
    error: Union[str, None]
    data: AutoMLCreateResponseContents