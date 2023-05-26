import { ResultsEchartsComponent } from './Components/results-echarts/results-echarts.component';
import { FormContainerComponent } from './Components/form-container/form-container.component';
import { AboutPageComponent } from './Components/about-page/about-page.component';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ResultPageComponent } from './Components/result-page/result-page.component';
import { HomepageComponent } from './Components/homepage/homepage.component';
import { DataValidationComponent } from './Components/data-validation/data-validation.component';

const routes: Routes = [{path:'', component: HomepageComponent},
                        {path:'home', component: HomepageComponent},
                        {path:'newRequest', component: FormContainerComponent},
                        {path:'results', component: ResultPageComponent},
                        {path:'result-charts', component: ResultsEchartsComponent},
                        {path:'dataValidation', component: DataValidationComponent},
                        {path:'walkthrough', component: AboutPageComponent}
                      ];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
