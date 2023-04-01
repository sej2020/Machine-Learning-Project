import { Component, OnInit, Renderer2 } from '@angular/core';
import { EChartsOption, registerTransform } from 'echarts';
import { ScriptService } from 'src/app/Services/script.service';
import { TransformComponent } from 'echarts/components';
import * as echarts from 'echarts';
import { aggregate } from '@manufac/echarts-simple-transform';
import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";
import { UploadRequestService } from 'src/app/Services/upload-request.service';
import { IResults } from '../result-page/result-page.component';
import { Router } from '@angular/router';


@Component({
  selector: 'app-results-echarts',
  templateUrl: './results-echarts.component.html',
  styleUrls: ['./results-echarts.component.scss']
})
export class ResultsEchartsComponent implements OnInit {

  boxplotData!: [string[]]
  cvLineplotData!: any
  resultsData!: IResults;

  constructor(private uploadRequestService: UploadRequestService, private router: Router) { }

  requestId: string = "";
  currentMetric: string = "";
  chartType: string = "boxplot";

  chartOptions!: EChartsOption;
  yAxisData!: string[];

  ngOnInit() {
    this.requestId = history.state.id ? history.state.id : "";
    if (this.requestId.length > 0) {
      this.fetchRawData();
    }
    echarts.registerTransform(aggregate as ExternalDataTransform);
  }

  fetchRawData = () => {
    this.uploadRequestService.getRequestStatus(this.requestId)
      .subscribe((data: any) => {
        this.resultsData = data;
        this.boxplotData = this.resultsData.data['visualization_data']['boxplot']
        this.cvLineplotData = this.resultsData.data['visualization_data']['cv_lineplot'];
        this.yAxisData = this.resultsData.data['metrics_list']
        this.currentMetric = this.yAxisData[0]
        this.chartOptions = this.getChartOptions(this.chartType);
        console.log(this.chartOptions);
      })
  }

  changeInRequestId(event: any) {
    this.requestId = event.target.value;
  }

  changeInCurrentMetric(value: any) {
    this.currentMetric = value;
    this.chartOptions = this.getChartOptions(this.chartType);
  }

  changeInChartType(value: any) {
    this.chartType = value;
    this.chartOptions = this.getChartOptions(this.chartType);
  }

  getChartOptions(chartType: string) {
    if (chartType === 'boxplot') {
      return this.getBoxPlotCharOptions(this.currentMetric, this.boxplotData);
    } else {
      return this.getCvLinePlotChartOptions(this.currentMetric, this.cvLineplotData['num_cv_folds'], this.cvLineplotData[this.currentMetric]);
    }
  }

  getDataPercentChartOptions(currentMetric: string, dataPercentages: number[], trainMetric: any, testMetric: any) {
    let dataPercentChartOptions = {
      xAxis: {
        type: 'category',
        data: dataPercentages
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          data: trainMetric,
          type: 'line',
          smooth: true,
          label: {
            show: true,
            position: "top"
          }
        },
        {
          data: testMetric,
          type: 'line',
          smooth: true,
          label: {
            show: true,
            position: "top"
          }
        }
      ]
    }
  }

  getCvLinePlotChartOptions(currentMetric: string, num_cv_folds: number, lineData: any) {
    let regressorNames = Object.keys(lineData);
    let lineYAxisData: object[] = [];
    for (var i = 0; i < regressorNames.length; i++) {
      lineYAxisData.push({
        name: regressorNames[i],
        type: 'line',
        data: lineData[regressorNames[i]]
      })
    }

    let cvLineplotOptions: EChartsOption = {
      title: {
        text: 'Regressors - ' + currentMetric
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: regressorNames,
        top: 30
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      toolbox: {
        feature: {
          saveAsImage: {}
        }
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: [...Array(num_cv_folds).keys()],
        name: "CV Fold"
      },
      yAxis: {
        type: 'value',
        name: currentMetric
      },
      series: lineYAxisData
    };

    return cvLineplotOptions;
  }

  getBoxPlotCharOptions(currentMetric: string, dataSource: [string[]]) {
    let boxplotOptions: EChartsOption = {
      title: {
        text: 'Regressors - ' + currentMetric
      },
      tooltip: {
        trigger: 'axis',
        confine: true
      },
      toolbox: {
        show: true,
        feature: {
          dataView: {
            readOnly: false
          },
          saveAsImage: {}
        }
      },
      dataset: [
        {
          id: 'raw',
          source: dataSource
        }, {
          id: 'raw_select_metric',
          fromDatasetId: 'raw',
          transform: [
            {
              type: 'filter',
              config: {
                dimension: 'metricName',
                value: currentMetric
              }
            }
          ]
        }, {
          id: 'raw_aggregate',
          fromDatasetId: 'raw_select_metric',
          transform: [
            {
              type: 'ecSimpleTransform:aggregate',
              config: {
                resultDimensions: [
                  { name: 'min', from: 'metricValue', method: 'min' },
                  { name: 'Q1', from: 'metricValue', method: 'Q1' },
                  { name: 'median', from: 'metricValue', method: 'median' },
                  { name: 'Q3', from: 'metricValue', method: 'Q3' },
                  { name: 'max', from: 'metricValue', method: 'max' },
                  { name: 'regressorName', from: 'regressorName' }
                ],
                groupBy: 'regressorName'
              }
            }, {
              type: 'sort',
              config: {
                dimension: 'Q3',
                order: 'asc'
              }
            }
          ]
        }
      ],
      xAxis: {
        name: this.currentMetric,
        nameLocation: 'middle',
        nameGap: 30,
        scale: true
      },
      yAxis: {
        type: 'category',
        splitArea: {
          show: true
        }
      },
      grid: {
        bottom: 100
      },
      series: [
        {
          type: 'boxplot',
          name: 'boxplot',
          itemStyle: {
            color: '#b8c5f2',
            borderColor: '#429',
            borderWidth: 3
          },
          datasetId: "raw_aggregate",
          encode: {
            x: ['min', 'Q1', 'median', 'Q3', 'max'],
            y: 'regressorName',
            itemName: ['regressorName'],
            tooltip: ['min', 'Q1', 'median', 'Q3', 'max']
          }
        }],
    };

    return boxplotOptions;
  }

}



