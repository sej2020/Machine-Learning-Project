import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ResultsEchartsComponent } from './results-echarts.component';

describe('ResultsEchartsComponent', () => {
  let component: ResultsEchartsComponent;
  let fixture: ComponentFixture<ResultsEchartsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ResultsEchartsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ResultsEchartsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
