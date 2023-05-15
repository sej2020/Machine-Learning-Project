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
  resultsData!: IResults;
  visualizationResponse!: IDataVisualizationResponse;
  visualizationResponse2d!: IDataVisualizationResponse;

  constructor(private uploadRequestService: UploadRequestService, private router: Router) { }

  defaultChartTypes: string[] = ['boxplot', 'cv_line_chart', 'train_test_error'];
  visualizationChartTypes: string[] = ['tsne_visualization_2d', 'pca_visualization_2d', 'tsne_visualization_3d', 'pca_visualization_3d', 'tsne_heatmap', 'pca_heatmap'];
  visualizationTypes: string[] = ['default_visualization', 'data_centric_visualization'];

  requestId: string = "";
  currentMetric: string = "";
  currentRegressor: string = "";
  chartType: string = "cv_line_chart";
  visualizationType: string = this.visualizationTypes[0];

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
        this.yAxisData = this.resultsData.data['metrics_list'];
        this.currentMetric = this.yAxisData[0];
        this.regressorList = this.resultsData.data['regressor_list'];
        this.currentRegressor = this.regressorList[0];
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
    console.log(this.requestId);
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
    this.chartType = value;
    this.updateChartOptions();
  }

  changeInVisualizationType(value: any) {
    this.visualizationType = value;
    this.chartType = this.visualizationType == 'default_visualization' ? this.defaultChartTypes[0] : this.visualizationChartTypes[0];
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
    let curVisualizationResponse = this.is3DVisualization()? this.visualizationResponse: this.visualizationResponse2d;
    let coloring_information = curVisualizationResponse.coloring_data[this.currentMetric][this.currentRegressor];
  
    if (algorithm == 'tsne') {
      dimension1 = curVisualizationResponse.tsne['dimension0'];
      dimension2 = curVisualizationResponse.tsne['dimension1'];
      if (dimension==3) {
        dimension3 = curVisualizationResponse.tsne['dimension2'];
      }
    } else {
      dimension1 = curVisualizationResponse.pca['dimension0'];
      dimension2 = curVisualizationResponse.pca['dimension1'];
      if (dimension==3) {
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
        text: this.currentRegressor + " : " + this.chartType + ' - ' + this.currentMetric,
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
        // inRange: {
        //   color: ['#24b7f2', '#f2c31a', '#ff0066']
        // },
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
    for (var i = 0; i < this.visualizationResponse.pca['dimension0'].length; i++) {
      var curColor = this.getColor(coloring_information[i]);
      colors.push(curColor);
    }
    var trace1 = {
      x: dimension1,
      y: dimension2,
      z: dimension3,
      mode: 'markers',
      marker: {
        size: 8,
        opacity: 0.8,
        color: colors
      },
      type: "scatter3d"
    };
    var plotData = [trace1];
    var layout = {
      margin: {
        l: 0,
        r: 0,
        b: 0,
        t: 0
      }
    };
    Plotly.newPlot('plotly-div', plotData, layout);
  }

  getColor(colorValue: number) {
    if (Math.abs(colorValue) < 0.25) {
      return '#096921';
    } else if (Math.abs(colorValue) < 0.75) {
      return '#929608';
    } else {
      return '#963a08';
    }
  }

  getHeatMap(algorithm: string) {
    let curVisualizationResponse = this.is3DVisualization()? this.visualizationResponse: this.visualizationResponse2d;
    let coloring_information = curVisualizationResponse.coloring_data[this.currentMetric][this.currentRegressor];
    let dimension0: number[], dimension1: number[];
    let xData: string[] = []
    let yData: string[] = [];
    if (algorithm == 'tsne') {
      dimension0 = curVisualizationResponse.tsne["dimension0"];
      dimension1 = curVisualizationResponse.tsne["dimension1"];
    } else {
      dimension0 = curVisualizationResponse.pca['dimension0'];
      dimension1 = curVisualizationResponse.pca['dimension1'];
    }

    for (var i=Math.min(...dimension0); i<Math.max(...dimension0); i = i+1) {
      xData.push(i.toFixed(2));
    }
    for (var i=Math.min(...dimension1); i<Math.max(...dimension1); i = i+1) {
      yData.push(i.toFixed(2));
    }

    let fullData = [];
    for(var i=0; i<xData.length; i++) {
      fullData.push([dimension0[i], dimension1[i], coloring_information[i]]);
    }

    let heatMapOptions: echarts.EChartsOption = {
      tooltip: {},
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
        // nameRotate: 90,
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
      visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        realtime: false,
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
        }
      },
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
          },
          progressive: 1000,
          animation: false
        }
      ]
    };
    this.chartOptions = heatMapOptions;
  }

  is3DVisualization() {
    return this.chartType.includes("3d");
  }

}



