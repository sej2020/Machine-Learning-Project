import { Component } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-about-page',
  templateUrl: './about-page.component.html',
  styleUrls: ['./about-page.component.scss']
})
export class AboutPageComponent {
  constructor(public route: ActivatedRoute, private router: Router) { }

  redirectToVisualization() {
    this.router.navigate(["", "result-charts"], { relativeTo: this.route, skipLocationChange: false, state: { id: 'bcf9b35d-002a-4a83-898b-5ade2117985b' } });
  }
}
