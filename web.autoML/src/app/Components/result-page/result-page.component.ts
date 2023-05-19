import { NONE_TYPE } from '@angular/compiler';
import { UploadRequestService } from './../../Services/upload-request.service';
import { Component } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Observable, Subject, Subscription, interval, takeUntil } from 'rxjs';

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

  constructor(private uploadRequestService: UploadRequestService, public route: ActivatedRoute, private router: Router) { }

  req_id: string = '';
  resultsData!: IResults;
  resultInterval = interval(1000);
  stopPlay$: Subject<any> = new Subject();
  keepFetching: boolean = true;
  ngOnInit() {
    this.req_id = history.state.id ? history.state.id : '';
    if (this.req_id.length > 0) {
      this.getResultsWithSubscription();
    }
  }

  changeInRequestId(event: any) {
    this.stopPlay$.next(10);
    this.req_id = event.target.value;
  }

  getResultsWithSubscription() {
    
    this.resultInterval.pipe(takeUntil(this.stopPlay$)).subscribe(() => {
      this.getResults();
    });
   
  }

  getResults() {
    this.uploadRequestService.getRequestStatus(this.req_id)
      .subscribe((data: any) => {
        this.resultsData = data;
        if (this.resultsData.data.request_status !== '0') {
          this.keepFetching = false;
          this.stopPlay$.next(10);
        }
      });
  }

  redirectToVisualization() {
    this.router.navigate(["", "result-charts"], { relativeTo: this.route, skipLocationChange: false, state: { id: this.req_id } });
  }

}
