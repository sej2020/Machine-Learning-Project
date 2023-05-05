import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { Component, OnInit } from '@angular/core';
import { IDataVisualizationResponse, UploadRequestService } from 'src/app/Services/upload-request.service';
import { IResults } from '../result-page/result-page.component';
import { Router } from '@angular/router';
import * as echarts from 'echarts';
import { aggregate } from '@manufac/echarts-simple-transform';
import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";
import { transform } from "echarts-stat";
declare let Plotly: any;

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
  chartType: string = "cv_line_chart";
  visualizationType: string = this.visualizationTypes[0];

  chartOptions!: echarts.EChartsOption;
  // highChartOptions!: Highcharts.Options;
  // Highcharts: typeof Highcharts = Highcharts;
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
    console.log('echarts.version: ', echarts.version);
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
        this.updateChartOptions();
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
        nameTextStyle:{
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
        axisLabel:{
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        type: 'value',
        splitArea:{
          interval: 'auto'
        },
        axisLine:{
          show: true
        },
        axisLabel:{
          fontWeight: 'bold',
          color: '#030359',
        },
        name: this.currentMetric,
        nameLocation: 'middle',
        nameGap: 50,
         nameTextStyle:{
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black',
          padding : [0, 10, 0, 0]
        },
      },
      legend: {
        data: ['Train ' + this.currentMetric, 'Test ' + this.currentMetric],
        top: '10%',
        itemHeight: 16,
        textStyle:{
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
      tooltip:{
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
            name: this.currentRegressor+'_'+this.currentMetric+'_'+this.chartType,
            title: 'Download Plot',
            pixelRatio: 2
          },
        },
      },
      series: [
        {
          name: 'Train ' + this.currentMetric,
          data:this.trainTestErrorData[this.currentMetric][this.currentRegressor]['train'],
          tooltip:{
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
          tooltip:{
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
        textStyle:{
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
            name: this.currentRegressor+'_'+this.currentMetric+'_'+this.chartType,
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
        nameTextStyle:{
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
        axisLabel:{
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        type: 'value',
        axisLine:{
          show: true
        },
        axisLabel:{
          fontWeight: 'bold',
          color: '#030359'
        },
        name: currentMetric,
        nameLocation: 'middle',
        nameGap: 30,
         nameTextStyle:{
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
            name: this.currentRegressor+'_'+this.currentMetric+'_'+this.chartType,
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
        nameTextStyle:{
          lineHeight: 14,
          fontWeight: 800,
          fontSize: 16,
          color: 'black'
        },
        axisLabel:{
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        type: 'category',
        splitArea: {
          show: true
        },
        axisLabel:{
          fontWeight: 'bold',
          color: '#030359'
        },
        nameGap: 30,
         nameTextStyle:{
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


  getScatterPlot(algorithm: string, dimension:number) {
    let coloring_information = this.visualizationResponse.coloring_data[this.currentMetric][this.currentRegressor];
    let minColorValue = Math.min(...Object.values(coloring_information));
    let maxColorValue = Math.max(...Object.values(coloring_information));
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

    if (dimension === 3) {
      for (var i = 0; i < dimension1.length; i++) {
          data.push([dimension1[i], dimension2[i], dimension3[i], coloring_information[i]]);
      }
      return this.get3DScatterPlotOptions(data);
    }
    else{
        for (var i = 0; i < dimension1.length; i++) {
        data.push([dimension1[i], dimension2[i], coloring_information[i]]);
        }
        return this.get2DScatterPlotOptions(data, minColorValue, maxColorValue);
    }
  }

  get2DScatterPlotOptions(data: any, minColorValue: number, maxColorValue: number) {
    let scatterPlotOption: echarts.EChartsOption = {
      title: {
        text: this.currentRegressor+" : "+this.chartType+' - ' + this.currentMetric,
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
            name: this.currentRegressor+'_'+this.currentMetric+'_'+this.chartType,
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
        textStyle:{
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
        inRange:{
          color: ['#24b7f2','#f2c31a','#ff0066'  ]
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
        nameTextStyle:{
          lineHeight: 14,
          fontWeight: 400,
          fontSize: 16,
          color: 'black'
        },
        axisLabel:{
          fontWeight: 'bold',
          color: '#030359'
        },
      },
      yAxis: {
        name: 'Dimension 2',
        nameLocation: 'start',
        nameGap: 40,
        nameTextStyle:{
          lineHeight: 14,
          fontWeight: 400,
          fontSize: 16,
          color: 'black'
        },
        axisLabel:{
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

  get3DScatterPlotOptions(data:any) {
    console.log(this.visualizationResponse.pca['dimension1']);
    let coloring_information = this.visualizationResponse.coloring_data[this.currentMetric][this.currentRegressor]
    var colors = [];
    for (var i=0; i<this.visualizationResponse.pca['dimension0'].length; i++) {
      var curColor = this.getColor(coloring_information[i]);
      colors.push(curColor);
    }
    var trace1 = {
      x: this.visualizationResponse.tsne['dimension0'].map(String), 
      y: this.visualizationResponse.tsne['dimension1'].map(String),
      z: this.visualizationResponse.tsne['dimension2'].map(String),
      mode: 'markers',
      marker: {
        size: 12,
        line: {
                color: 'rgba(217, 217, 217, 0.14)',
                width: 0.5
              },
        opacity: 0.8,
        color: colors
      },
      type: "scatter3d"
    };
    var plotData = [trace1];
    var layout = {margin: {
      l: 0,
      r: 0,
      b: 0,
      t: 0
      }};
    Plotly.newPlot('plotly-div', plotData, layout);
  }

  // get3DScatterPlotOptions(data: any) {
  //   console.log("am i coming here?", this.is3DVisualization());
  //   this.highChartOptions = {
  //     chart: {
  //       renderTo: 'container',
  //       margin: 100,
  //       type: 'scatter3d',
  //       animation: false,
  //       options3d: {
  //         enabled: true,
  //         alpha: 10,
  //         beta: 30,
  //         depth: 250,
  //         viewDistance: 5,
  //         fitToPlot: false,
  //       },
  //     },
  //     title: {
  //       text: '3D scatter chart',
  //     },
  //     // yAxis: {
  //     //   min: -50,
  //     //   max: 50,
  //     // },
  
  //     // xAxis: {
  //     //   min: -50,
  //     //   max: 50,
  //     //   gridLineWidth: 1,
  //     // },
  
  //     // zAxis: {
  //     //   min: -50,
  //     //   max: 50,
  //     //   showFirstLabel: false,
  //     // },
  
  //     series: [
  //       {
  //         type: 'scatter3d',
  //         data: data,
  //         colorByPoint: true,
  //       },
  //     ],
  //   };
  // }

  getColor(colorValue: number) {
    if (Math.abs(colorValue) <0.25) {
      return '#096921';
    } else if (Math.abs(colorValue) <0.75) {
      return '#929608';
    } else {
      return '#963a08';
    }
    // return colorValue;
  }

  is3DVisualization() {
    return this.chartType.includes("3d");
  }

}



