import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UserProfileCardsComponent } from './user-profile-cards.component';

describe('UserProfileCardsComponent', () => {
  let component: UserProfileCardsComponent;
  let fixture: ComponentFixture<UserProfileCardsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ UserProfileCardsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UserProfileCardsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
