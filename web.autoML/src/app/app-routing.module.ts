import { FormContainerComponent } from './Components/form-container/form-container.component';
import { MainComponentComponent } from './Components/main-component/main-component.component';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ResultPageComponent } from './Components/result-page/result-page.component';
import { HomepageComponent } from './Components/homepage/homepage.component';

const routes: Routes = [{path:'', component: HomepageComponent},
                        {path:'home', component: HomepageComponent},
                        {path:'newRequest', component: FormContainerComponent},
                        {path:'results', component: ResultPageComponent}
                      ];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
