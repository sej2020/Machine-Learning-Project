import { FormContainerComponent } from './../form-container/form-container.component';
import { UploadRequest, UploadRequestService } from './../../Services/upload-request.service';
import { Component } from '@angular/core';
export interface IResultsData{
  request_id : string,
  request_status: string,
  estimated_time_completion: number,
  result_link: string 

}
export interface IResults{
  data : IResultsData,
  error_message : string
}
@Component({
  selector: 'app-result-page',
  templateUrl: './result-page.component.html',
  styleUrls: ['./result-page.component.scss']
})



export class ResultPageComponent {

  constructor(private uploadRequestService : UploadRequestService
   ){}
    req_id : string ='';
    resultsData! : IResults;
    ngOnInit(){
     this.req_id = history.state.id;
     console.log("aaaaaa", history.state.id);
     this.getResults();
     
    }
    getResults(){
       this.uploadRequestService.getRequestStatus(this.req_id)
       .subscribe((data: any)=> {
        this.resultsData = data;
        console.log("results data", this.resultsData);
      });
    }
}
