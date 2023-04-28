import { Component, OnInit } from '@angular/core';
import { EChartsOption } from 'echarts';
import * as echarts from 'echarts';
import { aggregate } from '@manufac/echarts-simple-transform';
import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";
import { IDataVisualizationResponse, UploadRequestService } from 'src/app/Services/upload-request.service';
import { IResults } from '../result-page/result-page.component';
import { Router } from '@angular/router';
import { transform } from "echarts-stat";
import 'echarts-gl/dist/echarts-gl';
import 'echarts-gl/src/chart/scatter3D';
// import { Scatter3DChart } from 'echarts-gl/dist';
// import { Grid3DComponent } from 'echarts-gl/lib/component/*';

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
  visualizationChartTypes: string[] = ['tsne_visualization_2d', 'pca_visualization_2d', 'tsne_visualization_3d', 'pca_visualization_3d'];
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
    // echarts.use([Scatter3DChart, Grid3DComponent]);
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
      return this.getScatterPlot('tsne',2);
    }else if (this.chartType === 'tsne_visualization_3d') {
      console.log("coming from getChartOptions", this.chartType);
      return this.getScatterPlot('tsne',3); 
    }else if (this.chartType === 'pca_visualization_2d') {
      return this.getScatterPlot('pca', 2);
    }else if (this.chartType === 'pca_visualization_3d') {
      return this.getScatterPlot('pca', 3);
    }else {
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


  getScatterPlot(algorithm: string, dimension:number) {
    let coloring_information = this.visualizationResponse.coloring_data[this.currentMetric][this.currentRegressor]
    let dimension1: number[];
    let dimension2: number[];
    let dimension3: number[];
    let data = [];

    if (algorithm == 'tsne') {
      // console.log("tsne=====",JSON.stringify(this.visualizationResponse.tsne));
      dimension1 = this.visualizationResponse.tsne['dimension0'];
      dimension2 = this.visualizationResponse.tsne['dimension1'];
      dimension3 = this.visualizationResponse.tsne['dimension2'];
    } else {
      dimension1 = this.visualizationResponse.pca['dimension0'];
      dimension2 = this.visualizationResponse.pca['dimension1'];
      dimension3 = this.visualizationResponse.pca['dimension2'];
    }

    if(dimension === 3){
      for (var i = 0; i < dimension1.length; i++) {
          data.push([dimension1[i], dimension2[i], dimension3[i], this.getColor(coloring_information[i])]);
      }
      return this.get3DScatterPlotOptions(data);
    }
    else{
        for (var i = 0; i < dimension1.length; i++) {
        data.push([dimension1[i], dimension2[i], this.getColor(coloring_information[i])]);
        }
        return this.get2DScatterPlotOptions(data);
    }
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
    return scatterPlotOption;
  }

  // get3DScatterPlotOptions(data: any) {
  //   // console.log("data===", data[0], );
  //   var symbolSize = 4.5;
  //   var schema = [
  //     {
  //       name: 'dimension1',
  //       index: 0
  //     },
  //     {
  //       name: 'dimension2',
  //       index: 1
  //     },
  //     {
  //       name: 'dimension3',
  //       index: 2
  //     }
  //   ];
  //   var fieldIndices = schema.reduce(function (obj: any, item) {
  //     obj[item.name] = item.index;
  //     return obj;
  //   }, {});
  //   var fieldNames = schema.map(function (item) {
  //     return item.name;
  //   });
  //   let scatter3DPlotOption: EChartsOption = {
  //     // visualMap: [
  //     //   {
  //     //     max: max.color / 2
  //     //   },
  //     //   {
  //     //     max: max.symbolSize / 2
  //     //   }
  //     // ],
  //     visualMap: {
  //       type: 'piecewise',
  //       dimension: 3,
  //       splitNumber: 3,
  //       pieces: [
  //         { min: 0, max: 0.250, color: 'green' },
  //         { min: 0.251, max: 0.5, color: 'black' },
  //         { min: 0.501, max: 100, color: 'red' }
  //       ]
  //     },
  //     gradientColor: [
  //         '#f6efa6',
  //         '#d88273',
  //         '#bf444c'
  //       ],
  //     xAxis3D: {
  //       name: "X"
  //     },
  //     yAxis3D: {
  //       name: "Y"
  //     },
  //     zAxis3D: {
  //       name: "Z"
  //     },
  //     series: {
  //       dimensions: [
  //         schema[0].name,
  //         schema[1].name,
  //         schema[2].name,
  //       ],
  //       data: data,
  //       // data: data.map(function (item:any, idx:number) {
  //       //   return [
  //       //     item[fieldIndices[data[0][0]]],
  //       //     item[fieldIndices[data[1][0]]],
  //       //     item[fieldIndices[data[2][0]]],
  //       //     // item[fieldIndices[config.color]],
  //       //     // item[fieldIndices[config.symbolSize]],
  //       //     idx
  //       //   ];
  //       // })
  //     }
  //   }
  //   return scatter3DPlotOption;
  // }
  get3DScatterPlotOptions(data: any) {
    var symbolSize = 4.5;
    let scatter3DPlotOption: EChartsOption = {
      grid3D: {},
      xAxis3D: {
        type: 'category'
      },
      yAxis3D: {},
      zAxis3D: {},
      dataset: {
        dimensions: [
          'Income',
          'Life Expectancy',
          'Population',
          'Country',
          { name: 'Year', type: 'ordinal' }
        ],
        source: data
      },
      series: [
        {
          type: 'scatter3D',
          symbolSize: symbolSize,
          encode: {
            x: 'Country',
            y: 'Life Expectancy',
            z: 'Income',
            tooltip: [0, 1, 2, 3, 4]
          }
        }
      ]
    };
    return scatter3DPlotOption;
  }

  getColor(colorValue: number) {
    return colorValue
  }


}



