import { Component } from '@angular/core';
import { UploadRequestService } from 'src/app/Services/upload-request.service';


export interface IDataValidationResponse {
  issues: string[]
}

@Component({
  selector: 'app-data-validation',
  templateUrl: './data-validation.component.html',
  styleUrls: ['./data-validation.component.scss']
})
export class DataValidationComponent {

  constructor(public uploadRequestService : UploadRequestService){}

  uploadFile!: any;
  error: string[] = [];
  dataValidationResponse!:IDataValidationResponse;
  validationDone: boolean = false;

  onFileChange(event: any) {
    this.uploadFile = event.target.files[0];
    console.log("uploadFile: " + this.uploadFile);
  }

  validateData() {
    if(this.uploadFile) {
      this.uploadRequestService.validateDataRequest(this.uploadFile).subscribe((dvResponse:any) => {
        this.dataValidationResponse = dvResponse;
        this.error = this.dataValidationResponse.issues;
        this.validationDone = true;
      });
    } else {
      this.error = ['No data file supplied']
    }
  }
}
