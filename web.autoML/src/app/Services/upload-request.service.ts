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

  api_route = 'http://192.168.1.216:8081';

  createMLRequest(form_data: FormData, regressor: string[], metrics: string[], file_data: File, email:string, metric_score_method:string, test_set_size:string, num_cv_folds: string): Observable<any>{
    return this.httpClient.post(
      `${this.api_route}/createAutoMLRequest?email=${email}&metric_score_method=${metric_score_method}&num_cv_folds=${num_cv_folds}&test_set_size=${test_set_size}`,
      form_data, this.httpHeader)
      .pipe(retry(1), catchError(this.processError));
   }

  getRequestStatus(request_id: any){
    return this.httpClient.get(`${this.api_route}/getAutoMLRequest?request_id=${request_id}`)
    .pipe(retry(1), catchError(this.processError));
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
