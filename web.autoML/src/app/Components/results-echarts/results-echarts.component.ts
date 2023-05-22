import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { Component, OnInit } from '@angular/core';
import { IDataVisualizationResponse, UploadRequestService } from 'src/app/Services/upload-request.service';
import { IResults } from '../result-page/result-page.component';
import { Router } from '@angular/router';
import { aggregate } from '@manufac/echarts-simple-transform';
import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";
import { transform } from "echarts-stat";
import * as echarts from 'echarts';
// import 'node_modules/echarts-gl/dist/echarts-gl.js';
declare let Plotly: any;
// declare let echarts-gl any;

@Component({
  selector: 'app-results-echarts',
  templateUrl: './results-echarts.component.html',
  styleUrls: ['./results-echarts.component.scss']
})
export class ResultsEchartsComponent implements OnInit {
  boxplotData!: [string[]]
  cvLineplotData!: any
  trainTestErrorData!: any
  testBestModelsData!: any
  resultsData!: IResults;
  visualizationResponse!: IDataVisualizationResponse;
  visualizationResponse2d!: IDataVisualizationResponse;

  constructor(private uploadRequestService: UploadRequestService, private router: Router) { }

  defaultChartTypes: string[] = ['boxplot', 'cv_line_chart', 'train_test_error'];
  visualizationChartTypes: string[] = ['2d', '3d', 'heatmap'];
  visualizationTypes: string[] = ['default_visualization', 'data_centric_visualization'];

  requestId: string = "";
  currentMetric: string = "";
  currentRegressor: string = "";
  chartType: string = "cv_line_chart";
  actualChartType: string = "cv_line_chart";
  visualizationType: string = this.visualizationTypes[0];
  visualizationAlgorithm: string = "pca";
  chartOptions!: echarts.EChartsOption;
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
        this.testBestModelsData = this.resultsData.data['visualization_data']['test_best_models'];
        this.yAxisData = this.resultsData.data['metrics_list'];
        this.currentMetric = this.yAxisData[0];
        this.regressorList = this.resultsData.data['regressor_list'];
        this.currentRegressor = this.regressorList[0];
        this.visualizationType = this.visualizationTypes[0];
        this.chartType = this.defaultChartTypes[0];
        this.updateChartOptions();
      });

    this.uploadRequestService.getVisualizationData(this.requestId, 3)
      .subscribe((data: any) => {
        this.visualizationResponse = data;
      });

    this.uploadRequestService.getVisualizationData(this.requestId, 2)
      .subscribe((data: any) => {
        this.visualizationResponse2d = data;
      });
  }

  changeInRequestId(event: any) {
    this.requestId = event.target.value;
  }

  changeInCurrentMetric(value: any) {
    this.currentMetric = value;
    this.updateChartOptions();
  }

  changeInRegressor(value: any) {
    this.currentRegressor = value;
    this.updateChartOptions();
  }

  changeInChartType(value: any) {
    this.actualChartType = value;
    if (this.visualizationType == 'default_visualization') {
      this.chartType = value;
    } else {
      this.chartType = value == 'heatmap' ? this.chartType = this.visualizationAlgorithm + "_" + value : this.chartType = this.visualizationAlgorithm + '_visualization_' + value;
    }
    this.updateChartOptions();
  }

  changeInVisualizationAlgorithm(value: any) {
    if (this.visualizationType != 'default_visualization') {
      this.visualizationAlgorithm = value;
      this.chartType = this.actualChartType == 'heatmap' ? this.visualizationAlgorithm + "_" + this.actualChartType : this.chartType = this.visualizationAlgorithm + '_visualization_' + this.actualChartType;
      this.updateChartOptions();
    }
  }

  changeInVisualizationType(value: any) {
    this.visualizationType = value;
    this.currentRegressor = this.regressorList[0];
    this.actualChartType = this.visualizationType == 'default_visualization' ? this.defaultChartTypes[0] : this.visualizationChartTypes[0];
    this.chartType = this.visualizationType == 'default_visualization' ? this.defaultChartTypes[0] : this.visualizationAlgorithm + '_visualization_' + this.visualizationChartTypes[0];
    if (this.visualizationType == 'default_visualization') {
      this.yAxisData = this.resultsData.data['metrics_list'];
    } else {
      this.yAxisData = ['Raw Mean Absolute Percentage Error'];
    }
    this.currentMetric = this.yAxisData[0];
    this.updateChartOptions();
  }

  updateChartOptions() {
    if (this.chartType === 'boxplot') {
      this.getBoxPlotCharOptions(this.currentMetric, this.boxplotData);
    } else if (this.chartType === 'test_best_models') {
      this.getTestBestModelsChartOptions();
    } else if (this.chartType === 'cv_line_chart') {
      this.getCvLinePlotChartOptions(this.currentMetric, this.cvLineplotData['num_cv_folds'], this.cvLineplotData[this.currentMetric]);
    } else if (this.chartType === 'tsne_visualization_2d') {
      this.getScatterPlot('tsne', 2);
    } else if (this.chartType === 'tsne_visualization_3d') {
      this.getScatterPlot('tsne', 3);
    } else if (this.chartType === 'pca_visualization_2d') {
      this.getScatterPlot('pca', 2);
    } else if (this.chartType === 'pca_visualization_3d') {
      this.getScatterPlot('pca', 3);
    } else if (this.chartType == 'pca_heatmap') {
      this.getHeatMap('pca');
    } else if (this.chartType == 'tsne_heatmap') {
      this.getHeatMap('tsne');
    } else {
      this.getDataPercentChartOptions();
    }
  }

  getTestBestModelsChartOptions() {
    console.log(this.testBestModelsData);
    let curData = this.testBestModelsData[this.currentMetric];
    let regressorNames = Object.keys(curData);
    let plotData = [['Regressor', this.currentMetric]]
    for (var i = 0; i < regressorNames.length; i++) {
      plotData.push([regressorNames[i], curData[regressorNames[i]]])
    }
    
    console.log(plotData)

    let sortBarOptions: echarts.EChartsOption = {
      toolbox: {
        feature: {
          dataView: {
            readOnly: false,
            title: 'Data View'
          },
          saveAsImage: {
            type: 'png',
            name: 'best_models_' + this.requestId,
            title: 'Download Plot',
            pixelRatio: 2
          },
        },
      },
      title: {
        text: 'Ranking of regressors based on ' + this.currentMetric,
        padding: [5, 5, 5, 5],
        top: '2%',
        left: '1%'
      },
      dataset: [
        {
          dimensions: ['Regressor', this.currentMetric],
          source: plotData
        }
      ],
      xAxis: {
        type: 'category',
        nameLocation: 'middle',
        nameGap: 30,
        scale: true,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359',
          rotate: 30
        },
        // show: false
      },
      yAxis: {
        type: 'value',
        axisLine: {
          show: true
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
        name: this.currentMetric,
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
      },
      series: {
        type: 'bar',
        encode: { x: 'Regressor', y: this.currentMetric },
        datasetIndex: 0,
        label: {
          show: true,
          position: 'inside'
          // position: [5, -15],
          // formatter: function(params) {
          //     return params.name;
          // }
        }
      }
    };
    this.chartOptions = sortBarOptions;
  }

  getDataPercentChartOptions() {
    let dataPercentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let dataPercentChartOptions: echarts.EChartsOption = {
      responsive: true,
      maintainAspectRatio: true,
      title: {
        text: 'Train Test Error - ' + this.currentMetric,
        padding: [5, 5, 5, 5],
        top: '2%',
        left: '1%'
      },
      xAxis: {
        type: 'category',
        data: dataPercentages,
        boundaryGap: false,
        name: 'Percentage of Data Change',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        type: 'value',
        splitArea: {
          interval: 'auto'
        },
        axisLine: {
          show: true
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359',
        },
        name: this.currentMetric,
        nameLocation: 'middle',
        nameGap: 50,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black',
          padding: [0, 10, 0, 0]
        },
      },
      legend: {
        data: ['Train ' + this.currentMetric, 'Test ' + this.currentMetric],
        top: '10%',
        itemHeight: 16,
        textStyle: {
          fontWeight: 'bold',
          color: '#003300',
          fontSize: 16
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '5%',
        top: '20%',
        containLabel: true
      },
      tooltip: {
        show: true,
        trigger: 'axis',
      },
      toolbox: {
        feature: {
          dataView: {
            readOnly: false,
            title: 'Data View'
          },
          saveAsImage: {
            type: 'png',
            name: this.currentRegressor + '_' + this.currentMetric + '_' + this.chartType,
            title: 'Download Plot',
            pixelRatio: 2
          },
        },
      },
      series: [
        {
          name: 'Train ' + this.currentMetric,
          data: this.trainTestErrorData[this.currentMetric][this.currentRegressor]['train'],
          tooltip: {
            show: true
          },
          type: 'line',
          smooth: true,
          label: {
            show: true,
            position: "top"
          },
          showSymbol: true
        },
        {
          name: 'Test ' + this.currentMetric,
          data: this.trainTestErrorData[this.currentMetric][this.currentRegressor]['test'],
          type: 'line',
          smooth: true,
          tooltip: {
            show: true
          },
          showSymbol: true,
          label: {
            show: true,
            position: "top"
          }
        }
      ]
    };
    this.chartOptions = dataPercentChartOptions;
  }

  getCvLinePlotChartOptions(currentMetric: string, num_cv_folds: number, lineData: any) {
    let regressorNames = Object.keys(lineData);
    let lineYAxisData: object[] = [];
    for (var i = 0; i < regressorNames.length; i++) {
      lineYAxisData.push({
        name: regressorNames[i],
        type: 'line',
        data: lineData[regressorNames[i]],
        lineStyle: {
          width: 3,
          type: 'solid'
        }
      })
    }

    let cvLineplotOptions: echarts.EChartsOption = {
      responsive: true,
      maintainAspectRatio: true,
      title: {
        text: 'Regressors - ' + currentMetric,
        padding: [5, 5, 5, 5],
        top: '2%',
        left: '1%'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: regressorNames,
        top: '10%',
        textStyle: {
          fontWeight: 'bold',
          color: '#003300',
          fontSize: 16
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '5%',
        top: '20%',
        containLabel: true
      },
      toolbox: {
        feature: {
          dataView: {
            readOnly: false,
            title: 'Data View'
          },
          saveAsImage: {
            type: 'png',
            name: this.currentRegressor + '_' + this.currentMetric + '_' + this.chartType,
            title: 'Download Plot',
            pixelRatio: 2
          },
        },
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: [...Array(num_cv_folds).keys()],
        name: "CV Fold",
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        type: 'value',
        axisLine: {
          show: true
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
        name: currentMetric,
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'

        },
      },
      series: lineYAxisData
    };

    this.chartOptions = cvLineplotOptions;
  }

  getBoxPlotCharOptions(currentMetric: string, dataSource: [string[]]) {

    let boxplotOptions: echarts.EChartsOption = {
      responsive: true,
      maintainAspectRatio: true,
      title: {
        text: 'Regressors - ' + currentMetric,
        padding: [5, 5, 5, 5],
        top: '2%',
        left: '1%'
      },
      tooltip: {
        trigger: 'axis',
        confine: true
      },
      toolbox: {
        feature: {
          dataView: {
            readOnly: false,
            title: 'Data View'
          },
          saveAsImage: {
            type: 'png',
            name: this.currentRegressor + '_' + this.currentMetric + '_' + this.chartType,
            title: 'Download Plot',
            pixelRatio: 2
          },
        },
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
        scale: true,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        type: 'category',
        splitArea: {
          show: true
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
        nameGap: 30,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'

        },
      },
      // grid: {
      //   bottom: 100
      // },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '5%',
        top: '10%',
        containLabel: true
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

    this.chartOptions = boxplotOptions;
  }

  getScatterPlot(algorithm: string, dimension: number) {
    let dimension1: number[];
    let dimension2: number[];
    let dimension3: number[] = [];
    let data = [];
    let curVisualizationResponse = this.is3DVisualization() ? this.visualizationResponse : this.visualizationResponse2d;
    let coloring_information = curVisualizationResponse.coloring_data[this.currentMetric][this.currentRegressor];
    if (algorithm == 'tsne') {
      dimension1 = curVisualizationResponse.tsne['dimension0'];
      dimension2 = curVisualizationResponse.tsne['dimension1'];
      if (dimension == 3) {
        dimension3 = curVisualizationResponse.tsne['dimension2'];
      }
    } else {
      dimension1 = curVisualizationResponse.pca['dimension0'];
      dimension2 = curVisualizationResponse.pca['dimension1'];
      if (dimension == 3) {
        dimension3 = curVisualizationResponse.tsne['dimension2'];
      }
    }

    if (dimension === 3) {
      return this.get3DScatterPlotOptions(dimension1, dimension2, dimension3, coloring_information);
    }
    else {
      for (var i = 0; i < dimension1.length; i++) {
        data.push([dimension1[i], dimension2[i], coloring_information[i]]);
      }
      return this.get2DScatterPlotOptions(data);
    }
  }

  get2DScatterPlotOptions(data: any) {
    let scatterPlotOption: echarts.EChartsOption = {
      title: {
        text: this.currentRegressor + " : " + this.currentMetric,
        padding: [5, 5, 5, 5],
        top: '2%',
        left: '1%'
      },
      toolbox: {
        feature: {
          dataView: {
            readOnly: false,
            title: 'Data View'
          },
          saveAsImage: {
            type: 'png',
            name: this.currentRegressor + '_' + this.currentMetric + '_' + this.chartType,
            title: 'Download Plot',
            pixelRatio: 2
          },
        },
      },
      dataset: [
        {
          source: data
        },
      ],
      tooltip: {
        trigger: 'item',
        confine: true,
        position: 'top'
      },
      legend: {
        data: data,
        top: '10%',
        textStyle: {
          fontWeight: 'bold',
          color: '#003300',
          fontSize: 16
        },
      },
      visualMap: {
        type: 'continuous',
        precision: 3,
        min: 0,
        max: 1,
        dimension: 2,
        calculable: true,
        inRange: {
          color: [
            '#313695',
            '#4575b4',
            '#74add1',
            '#abd9e9',
            '#e0f3f8',
            '#ffffbf',
            '#fee090',
            '#fdae61',
            '#f46d43',
            '#d73027',
            '#a50026'
          ]
        },
        orient: 'vertical',
        right: 10,
        top: 'center',
        text: ['Prediction - Bad', 'Prediction - Good']
      },
      grid: {
        left: '3%',
        right: '20%',
        bottom: '5%',
        top: '20%',
        containLabel: true
      },

      xAxis: {
        name: 'Dimension 1',
        nameLocation: 'start',
        nameRotate: 90,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 400,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        name: 'Dimension 2',
        nameLocation: 'start',
        nameGap: 40,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 400,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      animation: true,
      animationEasing: 'cubicInOut',
      series: [{
        type: 'scatter',
        encode: { tooltip: [0, 1] },
        emphasis: {
          scale: true
        },
        markPoint: {
          symbol: 'pin'
        },
        symbolSize: 15,
        itemStyle: {
          borderColor: '#555'
        },
        datasetIndex: 0
      }]
    };
    this.chartOptions = scatterPlotOption;
  }

  get3DScatterPlotOptions(dimension1: number[], dimension2: number[], dimension3: number[], coloring_information: any) {
    var colors = [];
    var originalColors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'];
    // const colorScale = [
    //   ['0', '#292d87'],
    //   ['0.09', '#313695'],
    //   ['0.18', '#4575b4'],
    //   ['0.27', '#74add1'],
    //   ['0.36', '#abd9e9'],
    //   ['0.45', '#e0f3f8'],
    //   ['0.54', '#ffffbf'],
    //   ['0.63', '#fee090'],
    //   ['0.72', '#fdae61'],
    //   ['0.81', '#f46d43'],
    //   ['0.90', '#d73027'],
    //   ['1', '#a50026'],
    // ]
    var colorScale = [
      ['0.0', 'rgb(165,0,38)'],
      ['0.111111111111', 'rgb(215,48,39)'],
      ['0.222222222222', 'rgb(244,109,67)'],
      ['0.333333333333', 'rgb(253,174,97)'],
      ['0.444444444444', 'rgb(254,224,144)'],
      ['0.555555555556', 'rgb(224,243,248)'],
      ['0.666666666667', 'rgb(171,217,233)'],
      ['0.777777777778', 'rgb(116,173,209)'],
      ['0.888888888889', 'rgb(69,117,180)'],
      ['1.0', 'rgb(49,54,149)']
    ]

    for (var i = 0; i < this.visualizationResponse.pca['dimension0'].length; i++) {
      colors.push(this.getColor(coloring_information[i]));
      // colors.push(coloring_information[i])
    }
    // console.log(colors);
    var trace1 = {
      x: dimension1,
      y: dimension2,
      z: dimension3,
      mode: 'markers',
      marker: {
        size: 10,
        opacity: 0.7,
        color: colors,
        // colorscale: colorScale,
        // showscale: true
      },
      type: "scatter3d"
    };
    var plotData = [trace1];
    var layout = {
      title: this.currentRegressor + ' : ' + this.currentMetric,
      font: {
        color: 'black',
        size: 14
      },
      margin: {
        l: 0,
        r: 0,
        b: 50,
        t: 50,
        pad: 4
      },
      scene: {
        xaxis: {
          title: 'Dimension 1',
          backgroundcolor: "#cdd0d1",
          gridcolor: "rgb(255, 255, 255)",
          showbackground: true,
          // zerolinecolor: "rgb(255, 255, 255)",
        },
        yaxis: {
          title: 'Dimension 2',
          backgroundcolor: "#cdd0d1",
          gridcolor: "rgb(255, 255, 255)",
          showbackground: true,
          // zerolinecolor: "rgb(255, 255, 255)"
        },
        zaxis: {
          title: 'Dimension 3',
          backgroundcolor: "#cdd0d1",
          gridcolor: "rgb(255, 255, 255)",
          showbackground: true,
          // zerolinecolor: "rgb(255, 255, 255)"
        }
      }
    };
    Plotly.newPlot('plotly-div', plotData, layout);
  }

  getColor(colorValue: number) {
    let colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    var chunkSize = 1 / colors.length;
    for (var i = 0; i < colors.length; i++) {
      if (colorValue <= (i + 1) * chunkSize) {
        return colors[i];
      }
    }
    return colors[colors.length - 1];
  }

  getHeatMap(algorithm: string) {
    let curVisualizationResponse = this.is3DVisualization() ? this.visualizationResponse : this.visualizationResponse2d;
    let coloring_information = curVisualizationResponse.coloring_data[this.currentMetric][this.currentRegressor];
    // console.log(coloring_information);
    let dimension0: number[], dimension1: number[];
    let xData: number[] = []
    let yData: number[] = [];
    if (algorithm == 'tsne') {
      dimension0 = curVisualizationResponse.tsne["dimension0"];
      dimension1 = curVisualizationResponse.tsne["dimension1"];
    } else {
      dimension0 = curVisualizationResponse.pca['dimension0'];
      dimension1 = curVisualizationResponse.pca['dimension1'];
    }

    var chunkSize = Math.ceil(Math.max(...dimension0) - Math.min(...dimension0)) / 50;
    var i = Math.min(...dimension0);
    while (i < Math.max(...dimension0)) {
      xData.push(Math.round(i * 1000) / 1000);
      i += chunkSize;
    }
    i = Math.min(...dimension1);
    var chunkSize = Math.ceil(Math.max(...dimension1) - Math.min(...dimension1)) / 50;
    while (i < Math.max(...dimension1)) {
      yData.push(Math.round(i * 1000) / 1000);
      i += chunkSize;
    }
    let fullData: number[][] = [];
    for (var i = 0; i < dimension0.length; i++) {
      fullData.push([this.findClosest(xData, dimension0[i]), this.findClosest(yData, dimension1[i]), coloring_information[i]]);
    }
    let heatMapOptions: echarts.EChartsOption = {
      tooltip: {},
      title: {
        text: this.currentRegressor + " : " + this.currentMetric,
        padding: [5, 5, 5, 5],
        top: '2%',
        left: '1%'
      },
      toolbox: {
        feature: {
          dataView: {
            readOnly: false,
            title: 'Data View'
          },
          saveAsImage: {
            type: 'png',
            name: this.currentRegressor + '_' + this.currentMetric + '_' + this.chartType,
            title: 'Download Plot',
            pixelRatio: 2
          },
        },
      },
      dataZoom: {
        type: "inside",
        xAxisIndex: 0,
        yAxisIndex: 0
      },
      xAxis: [{
        type: 'category',
        data: xData,
        name: 'Dimension 1',
        nameLocation: 'middle',
        nameGap: 40,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 400,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
      }],
      yAxis: [{
        type: 'category',
        data: yData,
        name: 'Dimension 2',
        nameLocation: 'middle',
        nameGap: 50,
        nameRotate: 90,
        nameTextStyle: {
          lineHeight: 14,
          fontWeight: 400,
          fontSize: 16,
          color: 'black'
        },
        axisLabel: {
          fontWeight: 'bold',
          color: '#030359'
        },
      }],
      visualMap: [{
        dimension: 2,
        type: 'continuous',
        min: 0,
        max: 1,
        calculable: true,
        realtime: true,
        inRange: {
          color: [
            '#313695',
            '#4575b4',
            '#74add1',
            '#abd9e9',
            '#e0f3f8',
            '#ffffbf',
            '#fee090',
            '#fdae61',
            '#f46d43',
            '#d73027',
            '#a50026'
          ]
        },
        top: 'center',
        text: ['Prediction - Bad', 'Prediction - Good'],
        orient: 'vertical',
        right: 10,
      }],
      series: [
        {
          name: this.currentMetric,
          type: 'heatmap',
          data: fullData,
          emphasis: {
            itemStyle: {
              borderColor: '#333',
              borderWidth: 1
            }
          }
        }
      ]
    };
    this.chartOptions = heatMapOptions;
  }

  findNearestIndex(array: number[], value: number) {
    let i: number = 0;
    for (i = 0; i < array.length; i++) {
      if (value > array[i])
        return i;
    }
    return array.length - 1;
  }


  is3DVisualization() {
    return this.chartType.includes("3d");
  }

  findClosest(arr: number[], target: number) {
    let n = arr.length;

    // Corner cases
    if (target <= arr[0])
      return 0;
    if (target >= arr[n - 1])
      return n - 1;

    // Doing binary search
    let i = 0, j = n, mid = 0;
    while (i < j) {
      mid = Math.floor((i + j) / 2);

      if (arr[mid] == target)
        return mid;

      // If target is less than array
      // element,then search in left
      if (target < arr[mid]) {

        // If target is greater than previous
        // to mid, return closest of two
        if (mid > 0 && target > arr[mid - 1])
          return this.getClosest(arr, mid - 1, mid, target);

        // Repeat for left half
        j = mid;
      }

      // If target is greater than mid
      else {
        if (mid < n - 1 && target < arr[mid + 1])
          return this.getClosest(arr, mid, mid + 1, target);
        i = mid + 1; // update i
      }
    }

    // Only single element left after search
    // console.log('closest to ' + target + ' is at index ' + mid + ' with value ' + arr[mid]);
    return mid;
  }

  getClosest(arr: number[], idx1: number, idx2: number, target: number) {
    if (target - arr[idx1] >= arr[idx2] - target)
      return idx2;
    else
      return idx1;
  }

}



