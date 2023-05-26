import { Component } from '@angular/core';
import { MatList } from '@angular/material/list';
import { MatCarouselSlide, MatCarouselSlideComponent } from 'ng-mat-carousel';
import { Observer } from 'rxjs';
import { Observable } from 'rxjs/internal/Observable';

interface usersInterface {
  UserName: string
  UserTitle: string
  UserDetails: string
  UserProfileLinks: string[]
}
export interface Tabs {
  label: string;
}
@Component({
  selector: 'app-user-profile-cards',
  templateUrl: './user-profile-cards.component.html',
  styleUrls: ['./user-profile-cards.component.scss']
})

export class UserProfileCardsComponent {

  studentsProfile: usersInterface[] = [];
  advisorsProfile: usersInterface[] = [];
  asyncTabs: any;
  constructor(){
    this.asyncTabs = new Observable((observer: Observer<Tabs[]>) => {
      setTimeout(() => {
        observer.next([
          {label: 'Students'},
          {label: 'Advisors'},
        ]);
      }, 1000);
    });
  }
  ngOnInit() {
    this.studentsProfile = [
      {
        "UserName": "Sam Johnson",
        "UserTitle": "Undergraduate student at IUB",
        "UserDetails": "Sam Johnson is a Data Science student at IU whose research interests include Data-Centric AI, ML, and computational logic",
        "UserProfileLinks": ['https://github.com/sej2020', 'sj110.pages.iu.edu']
      },
      {
        "UserName": "Joshua-Elms",
        "UserTitle": "Undergraduate student at IUB",
        "UserDetails": "Josh Elms is a Data Science student at Indiana University with an interest in ML and meteorology",
        "UserProfileLinks": ['https://github.com/Joshua-Elms']
      },
      {
        "UserName": "Madhavan Kalkunte Ramachandra",
        "UserTitle": "Graduate student at IUB",
        "UserDetails": "Madhavan is a graduate in Computer Science from Luddy School of Informatics, Computing, and Engineering, Indiana University Bloomington.",
        "UserProfileLinks": ['https://github.com/MadhavanKR']
      },
      {
        "UserName": "Keerthana Sugasi",
        "UserTitle": "Graduate student at IUB",
        "UserDetails": "Keerthana graduated in Masters in Computer Science from Luddy School of Informatics, Computing, and Engineering, Indiana University Bloomington.",
        "UserProfileLinks": ['https://github.com/keerthana-mk']
      },
      {
        "UserName": "Prof. Hasan Kurban",
        "UserTitle": "Professor at IUB",
        "UserDetails": "Sam Johnson is a Data Science student at IU whose research interests include Data-Centric AI, ML, and computational logic",
        "UserProfileLinks": ['https://github.com/sej2020', 'sj110.pages.iu.edu']
      },
      {
        "UserName": "Prof. Mehmet M Dalkilic",
        "UserTitle": "Professor at IUB",
        "UserDetails": "Mehmet M Dalkilic is a Computer Science researcher with expertise in AI applications across multiple fields. He has authored numerous papers and developed unique courses, while pursuing interests in music, writing, and reading.",
        "UserProfileLinks": ['https://github.com/sej2020', 'sj110.pages.iu.edu']
      }
    ];
    this.advisorsProfile = [
      {
        "UserName": "Prof. Hasan Kurban",
        "UserTitle": "Professor at IUB",
        "UserDetails": "Sam Johnson is a Data Science student at IU whose research interests include Data-Centric AI, ML, and computational logic",
        "UserProfileLinks": ['https://github.com/sej2020', 'sj110.pages.iu.edu']
      },
      {
        "UserName": "Prof. DD",
        "UserTitle": "Professor at IUB",
        "UserDetails": "Sam Johnson is a Data Science student at IU whose research interests include Data-Centric AI, ML, and computational logic",
        "UserProfileLinks": ['https://github.com/sej2020', 'sj110.pages.iu.edu']
      },
    ];
  }
}