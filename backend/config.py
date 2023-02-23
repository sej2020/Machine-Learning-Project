import os
from dotenv import load_dotenv

from pathlib import Path

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    PROJECT_NAME: str = "AutoML"
    PROJECT_VERSION: str = "1.0.0"

    POSTGRES_USER: str = os.getenv("POSTGRES_USER", 'postgres')
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", 'postgres')
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", 5001)  # default postgres port is 5432
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "auto-ml-db")
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

    TEMP_UPLOAD_DIR = '/home/madhavan/python-workspace/Machine-Learning-Project/backend/uploads'
    TEMP_DOWNLOAD_DIR = '/home/madhavan/python-workspace/Machine-Learning-Project/backend/downloads'
    S3_DATA_BUCKET = 'data-bucket'
    S3_RESULTS_BUCKET = 'results-bucket'

    RMQ_AUTOML_REQ_IN = 'autml_req_in'

    REGRESSOR_LIST = ['ARDRegression', 'AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge', 'CCA', 'DecisionTreeRegressor', 'DummyRegressor',
                      'ElasticNet', 'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GammaRegressor', 'GaussianProcessRegressor',
                      'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'HuberRegressor', 'IsotonicRegression', 'KNeighborsRegressor',
                      'KernelRidge', 'Lars', 'Lasso', 'LassoLars', 'LassoLarsIC', 'LinearRegression', 'LinearSVR', 'MLPRegressor',
                      'MultiTaskElasticNet', 'MultiTaskLasso', 'NuSVR', 'OrthogonalMatchingPursuit', 'PLSCanonical', 'PLSRegression',
                      'PassiveAggressiveRegressor', 'PoissonRegressor', 'QuantileRegressor', 'RANSACRegressor', 'RadiusNeighborsRegressor',
                      'RandomForestRegressor', 'Ridge', 'SGDRegressor', 'SVR', 'TheilSenRegressor', 'TransformedTargetRegressor', 'TweedieRegressor']

settings = Settings()
