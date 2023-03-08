import { DisclaimerDialogComponent } from './../disclaimer-dialog/disclaimer-dialog.component';
import { UploadRequestService } from './../../Services/upload-request.service';
import { Component, EventEmitter, Output  } from '@angular/core';
import { FormControl, FormGroup  } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { MatDialog } from '@angular/material/dialog';
export interface regressors{
  groupName: string;
  regressorName: string[];
}
@Component({
  selector: 'app-form-container',
  templateUrl: './form-container.component.html',
  styleUrls: ['./form-container.component.scss'],
})

export class FormContainerComponent {
 
  constructor(
    public uploadRequestService : UploadRequestService,
    public route : ActivatedRoute,
    private router: Router,
    public dialog: MatDialog
  ){}
  request_id!: string ;
  mlGeneratorForm!: FormGroup;  
  regressorList : regressors[]=[ {
                                    groupName: 'Linear Models',
                                    regressorName : ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'Lars', 'LassoLars', 'LassoLarsIC', 'OrthogonalMatchingPursuit', 
                                      'BayesianRidge', 'ARDRegression', 'TransformedTargetRegressor']
                                  },
                                  {
                                    groupName:'Generalized Linear Models',
                                    regressorName:['TweedieRegressor', 'GammaRegressor', 'PoissonRegressor']
                                  },
                                  {
                                    groupName:'Linear Models Robust to Outliers',
                                    regressorName:['HuberRegressor', 'TheilSenRegressor', 'RANSACRegressor']
                                  },
                                  {
                                    groupName:'Support Vector Machines',
                                    regressorName:['NuSVR', 'SVR', 'LinearSVR']
                                  },
                                  {
                                    groupName:'Nearest Neighbors',
                                    regressorName:['KNeighborsRegressor', 'RadiusNeighborsRegressor']
                                  },
                                  {
                                    groupName: 'Tree-Based Regressors',
                                    regressorName: ['DecisionTreeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 'ExtraTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor',
                                      'HistGradientBoostingRegressor', 'GradientBoostingRegressor']
                                  },
                                  {
                                    groupName:'Others',
                                    regressorName: ['MLPRegressor', 'PassiveAggressiveRegressor', 'PLSRegression']
                                  }];
  visualizationMetricsList: string[] = ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R-Squared'];
  
  regressors!: FormControl;
  visualizationMetrics!: FormControl;
  rankingMetrics!: FormControl;
  testSetSize!: FormControl;
  cvFold!: FormControl;
  fileUpload!: FormControl;
  email!: FormControl;

  // title = 'Upload the data file here'; // title / heading
  // allowFileTypes = ".xls,.csv,.txt"; // file types
  // fileLimit = 1; // file limit
  // allowMultiple =  false; // allow multiple files

  buildForm() {
    this.mlGeneratorForm = new FormGroup({
      regressors : new FormControl(''),
      visualizationMetrics: new FormControl(''),
      rankingMetrics: new FormControl(''),
      testSetSize: new FormControl(''),
      cvFold: new FormControl(''),
      fileUpload: new FormControl(''),
      email: new FormControl('')
    });
  }
  getDroppedFiles(e: any){
    console.log("Dropped recieved File/s >>>> ", e); // here you can proceed with the captured files.
   }
ngOnInit(){
  this.buildForm();
  console.log("ML generator form:", this.mlGeneratorForm.value);
}
onFileChange(event: any){
  if(event.target.files.length >0){
    const file = event.target.files[0];
    this.mlGeneratorForm.patchValue({
      fileUpload: file
    });
  }
}

public openDisclaimer() {
  const dialogRef = this.dialog.open(DisclaimerDialogComponent);

  dialogRef.afterClosed().subscribe(result => {
    console.log(`Dialog result: ${result}`);
  });

}


public onSubmit(formData: any) {
  console.log(this.mlGeneratorForm);
  console.log(typeof this.mlGeneratorForm.value.fileUpload);
  const form_data = new FormData();
  form_data.append("regressor_list", this.mlGeneratorForm.get('regressors')?.value);
  form_data.append("metrics", this.mlGeneratorForm.get('visualizationMetrics')?.value);
  form_data.append("file_data", this.mlGeneratorForm.get('fileUpload')?.value);
  this.uploadRequestService.createMLRequest(form_data, formData.regressors, formData.visualizationMetrics,formData.fileUpload, formData.email,
     formData.rankingMetrics, "0.1", formData.cvFold).subscribe(response =>{
      console.log(response);
      this.request_id = response.data['request_id'];
      console.log(this.request_id);
    
  this.router.navigate(["", "results"],{relativeTo: this.route, skipLocationChange :false, state: {id: this.request_id}});
  });
}

}
