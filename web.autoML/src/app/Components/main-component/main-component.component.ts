import { Component, ViewChild } from '@angular/core';
import {MatToolbarModule} from '@angular/material/toolbar';
import {MatIconModule} from '@angular/material/icon';
import {MatDrawer, MatSidenav, MatSidenavModule} from '@angular/material/sidenav';
import {MatGridListModule} from '@angular/material/grid-list';
import { BreakpointObserver } from '@angular/cdk/layout';

export interface InavList{
  iconName: string
  itemName: string
  itemLink: string
}
@Component({
  selector: 'app-main-component',
  templateUrl: './main-component.component.html',
  styleUrls: ['./main-component.component.scss']
})
export class MainComponentComponent {

  toggleMenu : Boolean = false;
  @ViewChild(MatSidenav) sidenav !: MatSidenav;
  constructor(private observer: BreakpointObserver) {}

  listItems : InavList[] = [
    {
      iconName: "home",
      itemName: "Home",
      itemLink: "/home"
    },
    {
      iconName: "assignment_turned_in",
      itemName: "Data Validation",
      itemLink: "/dataValidation"
    },
    {
      iconName: "draw",
      itemName: "Create New Request",
      itemLink: "/newRequest"
    },
    {
      iconName: "stacked_bar_chart",
      itemName: "View Results",
      itemLink: "/results"
    },
    {
      iconName: "stacked_bar_chart",
      itemName: "Result Visualization",
      itemLink: "/result-charts"
    }
  ]

  ngAfterViewInit() {
    this.observer.observe(['(max-width: 800px)']).subscribe((res) => {
      if (res.matches) {
        this.sidenav.mode = 'over';
        this.sidenav.close();
      } else {
        this.sidenav.mode = 'side';
        this.sidenav.open();
      }
    });
  }

  ngOnDestroy() {}
}
