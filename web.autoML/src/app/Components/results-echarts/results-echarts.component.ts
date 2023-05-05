import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { Component, OnInit } from '@angular/core';
import { EChartsOption } from 'echarts';
import * as echarts from 'echarts';
import { aggregate } from '@manufac/echarts-simple-transform';
import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";
import { IDataVisualizationResponse, UploadRequestService } from 'src/app/Services/upload-request.service';
import { IResults } from '../result-page/result-page.component';
import { Router } from '@angular/router';
import { transform } from "echarts-stat";
import { min } from 'rxjs/operators';
// import * from 'echarts-gl';
// import 'echarts-gl/src/chart/scatter3D';
// import { Scatter3DChart } from 'echarts-gl/dist/charts/*';
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
  chartType: string = "cv_line_chart";
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
    return dataPercentChartOptions;
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

    let cvLineplotOptions: EChartsOption = {
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

    return cvLineplotOptions;
  }

  getBoxPlotCharOptions(currentMetric: string, dataSource: [string[]]) {
    let boxplotOptions: EChartsOption = {
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

    return boxplotOptions;
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

    if(dimension === 3){
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
    let scatterPlotOption: EChartsOption = {
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
      // series: [
      //   {
      //     type: 'scatter3D',
      //     symbolSize: symbolSize,
      //     encode: {
      //       x: 'Country',
      //       y: 'Life Expectancy',
      //       z: 'Income',
      //       tooltip: [0, 1, 2, 3, 4]
      //     }
      //   }
      // ]
    };
    return scatter3DPlotOption;
  }

  getColor(colorValue: number) {
    return colorValue
  }


}



