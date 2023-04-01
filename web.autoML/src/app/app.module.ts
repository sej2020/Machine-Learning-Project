import { NgModule,CUSTOM_ELEMENTS_SCHEMA, NO_ERRORS_SCHEMA } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MainComponentComponent } from './Components/main-component/main-component.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import {MatToolbarModule} from '@angular/material/toolbar';
import {MatIconModule} from '@angular/material/icon';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatGridListModule} from '@angular/material/grid-list';
import {FormContainerComponent } from './Components/form-container/form-container.component';
import {MatSelectModule} from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { ReactiveFormsModule, FormControl } from '@angular/forms';
import {MatCardModule} from '@angular/material/card';
import {MatListModule} from '@angular/material/list';
import { MatInputModule } from '@angular/material/input';
import { EasyDragAndDropFileUploadModule } from  'easy-drag-and-drop-file-upload'; 
import {MatButtonModule} from '@angular/material/button';
import {MatDividerModule} from '@angular/material/divider';
import { HttpClientModule } from '@angular/common/http';
import { ResultPageComponent } from './Components/result-page/result-page.component';
import {MatDialogModule} from '@angular/material/dialog';
import { HomepageComponent } from './Components/homepage/homepage.component';
import { DisclaimerDialogComponent } from './Components/disclaimer-dialog/disclaimer-dialog.component';
import { NgxEchartsModule } from 'ngx-echarts';
import { ResultsEchartsComponent } from './Components/results-echarts/results-echarts.component';
import * as echarts from 'echarts';
import { UserProfileCardsComponent } from './Components/user-profile-cards/user-profile-cards.component';
import { MatCarouselModule} from 'ng-mat-carousel';
@NgModule({
  
  declarations: [
    AppComponent,
    MainComponentComponent,
    FormContainerComponent,
    ResultPageComponent,
    HomepageComponent,
    DisclaimerDialogComponent,
    ResultsEchartsComponent,
    UserProfileCardsComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    MatToolbarModule,
    MatIconModule,
    MatSidenavModule,
    MatGridListModule,
    MatSelectModule,
    MatFormFieldModule,
    ReactiveFormsModule,
    MatCardModule,
    MatInputModule,
    EasyDragAndDropFileUploadModule,
    MatButtonModule,
    MatDividerModule,
    HttpClientModule,
    MatDividerModule,
    AppRoutingModule,
    MatListModule,
    MatDialogModule,
    NgxEchartsModule.forRoot({
      echarts
    }),
    MatCarouselModule.forRoot(),
  ],
  exports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    MatToolbarModule,
    MatIconModule,
    MatSidenavModule,
    MatGridListModule,
    MatSelectModule,
    MatFormFieldModule,
    ReactiveFormsModule ,
    MatCardModule,
    MatInputModule,
    EasyDragAndDropFileUploadModule,
    MatButtonModule,
    MatDividerModule,
    HttpClientModule,
    MatDividerModule,
    AppRoutingModule,
    MatListModule,
    MatDialogModule,
  ],
  providers: [],
  bootstrap: [AppComponent],
  schemas: [CUSTOM_ELEMENTS_SCHEMA]
 
})
export class AppModule { }
