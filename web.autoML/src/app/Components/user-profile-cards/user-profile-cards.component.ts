import { Component } from '@angular/core';
import { MatCarouselSlide, MatCarouselSlideComponent } from 'ng-mat-carousel';

interface usersInterface {
  UserName: string
  UserTitle: string
  UserDetails: string
  UserProfileLinks: string[]
}

@Component({
  selector: 'app-user-profile-cards',
  templateUrl: './user-profile-cards.component.html',
  styleUrls: ['./user-profile-cards.component.scss']
})
export class UserProfileCardsComponent {

  usersProfile: usersInterface[] = [];
  ngOnInit() {
    this.usersProfile = [
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
      }
    ];
  }
}