# Code base

Please checkout the code in _feature/web-backend_ branch
# Backend server

## Create docker containers
Refer [here](https://github.com/sej2020/Machine-Learning-Project/wiki/Web-Service) to create docker containers for RabbitMQ, Postgres, and S3 Server.

## Starting required docker containers
```commandline
docker start some-rabbit
docker start s3server
docker start auto-ml-postgres
```

## Installing python dependencies

Install required python modules by running the following command from root of the project.
```commandline
pip install -r backend/requirements_updated.txt
```

## Running web server

```commandline
cd backend
python main.py
```

You will be able to access the API doc by accessing [http://localhost:8081/docs](http://localhost:8081/docs).

## Running consumer application

The consumer application subscribes to RMQ and runs ML pipeline

```commandline
cd backend
python consumer_main.py
```

# Front End application

## Installing dependencies

```commandline
cd web.AutoML
npm ci
```

if above command gives error, please try ```npm ci --legacy-peer-deps```

## Running front end application

```commandline
cd web.AutoML
ng serve
```

if ng command is not found, please refer this [link](https://medium.com/@angela.amarapala/ways-to-fix-bash-ng-command-not-found-7f329745795)

You should be able to access the webpage on [http://localhost:4200](http://localhost:4200)