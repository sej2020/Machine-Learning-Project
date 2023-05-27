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
  ProfilePicture: string
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
        "UserProfileLinks": ['https://github.com/sej2020', 'https://sj110.pages.iu.edu'],
        "ProfilePicture": '../../../assets/profile-pics/sam.jpg'
      },
      {
        "UserName": "Joshua-Elms",
        "UserTitle": "Undergraduate student at IUB",
        "UserDetails": "Josh Elms is a Data Science student at Indiana University with an interest in ML and meteorology",
        "UserProfileLinks": ['https://github.com/Joshua-Elms'],
        "ProfilePicture": '../../../assets/profile-pics/josh.jpg'
      },
      {
        "UserName": "Madhavan K R",
        "UserTitle": "Graduate student at IUB",
        "UserDetails": "Madhavan is a graduate in Computer Science from Luddy School of Informatics, Computing, and Engineering, Indiana University Bloomington.",
        "UserProfileLinks": ['https://github.com/MadhavanKR'],
        "ProfilePicture": '../../../assets/profile-pics/madhavan.jpg'
      },
      {
        "UserName": "Keerthana Sugasi",
        "UserTitle": "Graduate student at IUB",
        "UserDetails": "Keerthana graduated in Masters in Computer Science from Luddy School of Informatics, Computing, and Engineering, Indiana University Bloomington.",
        "UserProfileLinks": ['https://github.com/keerthana-mk'],
        "ProfilePicture": '../../../assets/profile-pics/keerthana.jpg'
      }
    ];
    this.advisorsProfile = [
      {
        "UserName": "Prof. Mehmet M Dalkilic",
        "UserTitle": "Professor at IUB",
        "UserDetails": "Mehmet M Dalkilic is a Computer Science researcher with expertise in AI applications across multiple fields. He has authored numerous papers and developed unique courses, while pursuing interests in music, writing, and reading.",
        "UserProfileLinks": ['', 'https://luddy.indiana.edu/contact/profile/?profile_id=187'],
        "ProfilePicture": '../../../assets/profile-pics/dalkilic.jpg'
      },
      {
        "UserName": "Prof. Hasan Kurban",
        "UserTitle": "Visiting Associate Professor at IUB",
        "UserDetails": "Hasan Kurban is a Visiting Associate Professor at Indiana University, specializing in data-centric AI and its applications in materials science. He has received awards for his work on improving expectation-maximization algorithms and his CRAN R package DCEM has garnered over 24K downloads.",
        "UserProfileLinks": ['', 'https://www.hasankurban.com'],
        "ProfilePicture": '../../../assets/profile-pics/kurban.jpg'
      },
      {
        "UserName": "Parichit Sharma",
        "UserTitle": "Ph.D. student at IU, Bloomington",
        "UserDetails": "Parichit Sharma is a Computer Science PhD student at IU, Bloomington, focusing on Machine Learning and Bioinformatics. His primary focus is developing algorithms for mining big data and extracting patterns from genomics data, contributing to high-performance computing solutions.",
        "UserProfileLinks": ['https://github.com/parichit', 'https://parichitsharma.com/'],
        "ProfilePicture": '../../../assets/profile-pics/parichit.jpg'
      },
    ];
  }
}