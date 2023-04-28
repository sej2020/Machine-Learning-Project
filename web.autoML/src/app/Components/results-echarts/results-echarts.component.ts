import { Component, OnInit, Renderer2 } from '@angular/core';
import { EChartsOption, registerTransform } from 'echarts';
import { ScriptService } from 'src/app/Services/script.service';
import { TransformComponent } from 'echarts/components';
import * as echarts from 'echarts';
import { aggregate } from '@manufac/echarts-simple-transform';
import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";
import { IDataVisualizationResponse, UploadRequestService } from 'src/app/Services/upload-request.service';
import { IResults } from '../result-page/result-page.component';
import { Router } from '@angular/router';
import { transform } from "echarts-stat";

@Component({
  selector: 'app-results-echarts',
  templateUrl: './results-echarts.component.html',
  styleUrls: ['./results-echarts.component.scss']
})
export class ResultsEchartsComponent implements OnInit {

  boxplotData!: [string[]]
  cvLineplotData!: any
  trainTestErrorData!: any
  resultsData!: IResults;
  visualizationResponse!: IDataVisualizationResponse;

  constructor(private uploadRequestService: UploadRequestService, private router: Router) { }

  defaultChartTypes: string[] = ['boxplot', 'cv_line_chart', 'train_test_error'];
  visualizationChartTypes: string[] = ['tsne_visualization_2d', 'pca_visualization_2d'];
  visualizationTypes: string[] = ['default_visualization', 'data_centric_visualization'];

  requestId: string = "";
  currentMetric: string = "";
  currentRegressor: string = "";
  chartType: string = "boxplot";
  visualizationType: string = this.visualizationTypes[0];

  chartOptions!: EChartsOption;
  yAxisData!: string[];
  regressorList!: string[];

  ngOnInit() {
    this.requestId = history.state.id ? history.state.id : "";
    if (this.requestId.length > 0) {
      this.fetchRawData();
    }
    echarts.registerTransform(aggregate as ExternalDataTransform);
    echarts.registerTransform(transform.clustering);
  }

  fetchRawData = () => {
    this.uploadRequestService.getRequestStatus(this.requestId)
      .subscribe((data: any) => {
        this.resultsData = data;
        this.boxplotData = this.resultsData.data['visualization_data']['boxplot']
        this.cvLineplotData = this.resultsData.data['visualization_data']['cv_lineplot'];
        this.trainTestErrorData = this.resultsData.data['visualization_data']['train_test_error'];
        this.yAxisData = this.resultsData.data['metrics_list'];
        this.yAxisData.push('Raw Mean Absolute Percentage Error');
        this.currentMetric = this.yAxisData[0];
        this.regressorList = this.resultsData.data['regressor_list'];
        this.currentRegressor = this.regressorList[0];
        this.chartOptions = this.getChartOptions();
      });

    this.uploadRequestService.getVisualizationData(this.requestId)
      .subscribe((data: any) => {
        this.visualizationResponse = data;
      });
  }

  changeInRequestId(event: any) {
    this.requestId = event.target.value;
  }

  changeInCurrentMetric(value: any) {
    this.currentMetric = value;
    this.chartOptions = this.getChartOptions();
  }

  changeInRegressor(value: any) {
    this.currentRegressor = value;
    this.chartOptions = this.getChartOptions();
  }

  changeInChartType(value: any) {
    this.chartType = value;
    this.chartOptions = this.getChartOptions();
  }

  changeInVisualizationType(value: any) {
    this.visualizationType = value;
    this.chartType = this.visualizationType == 'default_visualization' ? this.defaultChartTypes[0] : this.visualizationChartTypes[0];
    console.log(this.defaultChartTypes[0]);
    console.log(this.visualizationChartTypes[0]);
    console.log('chartType: ' + this.chartType);
    this.chartOptions = this.getChartOptions();
  }

  getChartOptions() {
    if (this.chartType === 'boxplot') {
      return this.getBoxPlotCharOptions(this.currentMetric, this.boxplotData);
    } else if (this.chartType === 'cv_line_chart') {
      return this.getCvLinePlotChartOptions(this.currentMetric, this.cvLineplotData['num_cv_folds'], this.cvLineplotData[this.currentMetric]);
    } else if (this.chartType === 'tsne_visualization_2d') {
      return this.getScatterPlot(this.currentMetric, 'tsne');
    } else if (this.chartType == 'pca_visualization_2d') {
      return this.getScatterPlot(this.currentMetric, 'pca');
    } else {
      return this.getDataPercentChartOptions();
    }
  }

  getDataPercentChartOptions() {
    let dataPercentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let dataPercentChartOptions: EChartsOption = {
      xAxis: {
        type: 'category',
        data: dataPercentages
      },
      yAxis: {
        type: 'value'
      },
      legend: {
        data: ['Train ' + this.currentMetric, 'Test ' + this.currentMetric],
        top: 30
      },
      series: [
        {
          name: 'Train ' + this.currentMetric,
          data: this.trainTestErrorData[this.currentMetric][this.currentRegressor]['train'],
          type: 'line',
          smooth: true,
          label: {
            show: true,
            position: "top"
          }
        },
        {
          name: 'Test ' + this.currentMetric,
          data: this.trainTestErrorData[this.currentMetric][this.currentRegressor]['test'],
          type: 'line',
          smooth: true,
          label: {
            show: true,
            position: "top"
          }
        }
      ]
    };
    return dataPercentChartOptions;
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


  getScatterPlot(currentMetric: string, algorithm: string) {
    let coloring_information = this.visualizationResponse.coloring_data[currentMetric][this.currentRegressor]
    let dimension1: number[];
    let dimension2: number[];

    if (algorithm == 'tsne') {
      dimension1 = this.visualizationResponse.tsne['dimension1'];
      dimension2 = this.visualizationResponse.tsne['dimension2'];
    } else {
      dimension1 = this.visualizationResponse.pca['dimension1'];
      dimension2 = this.visualizationResponse.pca['dimension2'];
    }

    let data = [];
    for (var i = 0; i < dimension1.length; i++) {
      data.push([dimension1[i], dimension2[i], this.getColor(coloring_information[i])])
    }
    return this.get2DScatterPlotOptions(data);
  }

  get2DScatterPlotOptions(data: any) {
    let scatterPlotOption: EChartsOption = {
      dataset: [
        {
          source: data
        },
      ],
      tooltip: {
        position: 'top'
      },
      visualMap: {
        type: 'piecewise',
        dimension: 2,
        splitNumber: 3,
        pieces: [
          { min: 0, max: 0.250, color: 'green' },
          { min: 0.251, max: 0.5, color: 'black' },
          { min: 0.501, max: 100, color: 'red' }
        ]
      },
      grid: {
        left: 120
      },
      xAxis: {},
      yAxis: {},
      series: [{
        type: 'scatter',
        encode: { tooltip: [0, 1] },
        symbolSize: 15,
        itemStyle: {
          borderColor: '#555'
        },
        datasetIndex: 0
      }]
    };
    return scatterPlotOption
  }

  // get3DScatterPlotOptions(data: any) {
  //   let scatterPlotOption: EChartsOption = {
  //     visualMap: [
  //       {
  //         max: max.color / 2
  //       },
  //       {
  //         max: max.symbolSize / 2
  //       }
  //     ],
  //     xAxis3D: {
  //       name: config.xAxis3D
  //     },
  //     yAxis3D: {
  //       name: config.yAxis3D
  //     },
  //     zAxis3D: {
  //       name: config.zAxis3D
  //     },
  //     series: {
  //       dimensions: [
  //         config.xAxis3D,
  //         config.yAxis3D,
  //         config.yAxis3D,
  //         config.color,
  //         config.symbolSiz
  //       ],
  //       data: data.map(function (item, idx) {
  //         return [
  //           item[fieldIndices[config.xAxis3D]],
  //           item[fieldIndices[config.yAxis3D]],
  //           item[fieldIndices[config.zAxis3D]],
  //           item[fieldIndices[config.color]],
  //           item[fieldIndices[config.symbolSize]],
  //           idx
  //         ];
  //       })
  //     }
  //   }
  // }

  getColor(colorValue: number) {
    return colorValue
  }


}



