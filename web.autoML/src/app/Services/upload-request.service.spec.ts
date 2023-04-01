import { TestBed } from '@angular/core/testing';

import { UploadRequestService } from './upload-request.service';

describe('UploadRequestServiceService', () => {
  let service: UploadRequestService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(UploadRequestService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
