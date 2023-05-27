import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { catchError, Observable, retry, throwError } from 'rxjs';
export class UploadRequest{
  
  // email?: string
  // metric_score_method?: string
  // test_set_size?: string
  // num_cv_folds?: number
  regressors?: string[]
  metrics?: string[]
  file_data?: File
}

export interface IDataVisualizationResponse {
  tsne: Record<string, number[]>,
  pca: Record<string, number[]>,
  coloring_data: Record<string, Record<string, Record<string, number>>>
}

@Injectable({
  providedIn: 'root'
})


export class UploadRequestService {

  constructor(private httpClient: HttpClient) { }
  httpHeader = {
    headers: new HttpHeaders({
      // 'Content-Type': 'multipart/form-data',
      'Accept': 'application/json',
      // 'Content-Type': 'multipart/form-data'
    }),
  };

  // api_route = 'http://192.168.1.216:8081';
  // api_route = 'http://localhost:8081';
  api_route = 'https://dalkilic.luddy.indiana.edu/api';


  createMLRequest(form_data: FormData, regressor: string[], metrics: string[], file_data: File, email: string, metric_score_method: string, test_set_size: string, num_cv_folds: string, default_setting: string): Observable<any> {
    if (default_setting === "1") {
      return this.httpClient.post(`${this.api_route}/createAutoMLRequest?email=${email}`, form_data, this.httpHeader).pipe(retry(1), catchError(this.processError));
    } else {
      return this.httpClient.post(
        `${this.api_route}/createAutoMLRequest?email=${email}&metric_score_method=${metric_score_method}&num_cv_folds=${num_cv_folds}&test_set_size=${test_set_size}&default_setting=${default_setting}`,
        form_data, this.httpHeader)
        .pipe(retry(1), catchError(this.processError));
    }
  }

  getRequestStatus(request_id: any){
    return this.httpClient.get(`${this.api_route}/getAutoMLRequest?request_id=${request_id}`)
    .pipe(retry(1), catchError(this.processError));
  } 

  validateDataRequest(dataFile: any) {
    const formData = new FormData();
    formData.append("file_data", dataFile);
    return this.httpClient.post(`${this.api_route}/validateData`, formData).pipe(retry(1), catchError(this.processError));
  }

  getVisualizationData(requestId: string, dimensionality: number) {
    return this.httpClient.get(`${this.api_route}/dataVisualization?requestId=${requestId}&dimensionality=${dimensionality}`).pipe(retry(1), catchError(this.processError));
  }
  
  processError(err: any) {
    let message = '';
    if (err.error instanceof ErrorEvent) {
      message = err.error.message;
    } else {
      message = `Error Code: ${err.status}\nMessage: ${err.message}`;
    }
    console.log(message);
    return throwError(() => {
      message;
    });
  }

}
