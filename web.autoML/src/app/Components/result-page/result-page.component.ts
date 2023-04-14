import { FormContainerComponent } from './../form-container/form-container.component';
import { UploadRequest, UploadRequestService } from './../../Services/upload-request.service';
import { Component } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';

export interface IResultsData {
  request_id: string,
  request_status: string,
  estimated_time_completion: number,
  result_link: string
  visualization_data: Record<string, [string[]]>,
  metrics_list: string[],
  regressor_list: string[]
}

export interface IResults {
  data: IResultsData,
  error_message: string
}
@Component({
  selector: 'app-result-page',
  templateUrl: './result-page.component.html',
  styleUrls: ['./result-page.component.scss']
})



export class ResultPageComponent {

  constructor(private uploadRequestService: UploadRequestService, public route : ActivatedRoute, private router: Router) { }

  req_id: string = '';
  resultsData!: IResults;
  resultsFetched = false;

  ngOnInit() {
    this.req_id = history.state.id? history.state.id: '';
    if(this.req_id.length > 0) {
      this.getResults();
    }
  }

  changeInRequestId(event: any) {
    this.req_id = event.target.value;
  }

  getResults() {
    this.uploadRequestService.getRequestStatus(this.req_id)
      .subscribe((data: any) => {
        this.resultsData = data;
        this.resultsFetched = true;
        console.log("results data", this.resultsData);
      });
  }

  redirectToVisualization() {
    this.router.navigate(["", "result-charts"],{relativeTo: this.route, skipLocationChange :false, state: {id: this.req_id}});
  }

}
