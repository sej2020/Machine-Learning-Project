import { Component, OnInit, Renderer2 } from '@angular/core';
import { EChartsOption, registerTransform } from 'echarts';
import { ScriptService } from 'src/app/Services/script.service';
import { TransformComponent } from 'echarts/components';
import * as echarts from 'echarts';
import { aggregate } from '@manufac/echarts-simple-transform';
import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";


@Component({
  selector: 'app-results-echarts',
  templateUrl: './results-echarts.component.html',
  styleUrls: ['./results-echarts.component.scss']
})
export class ResultsEchartsComponent implements OnInit {

  // regressorData = [[134.744, 101.139, 107.769, 83.486, 103.296, 103.997, 96.735, 91.425, 139.113, 118.779], [61.9045328587085, 52.5798970391187, 60.320216595453, 61.8208707545006, 59.9380889894142, 45.8944453371561, 55.2555857413487, 81.7090951014166, 74.4910264391988, 61.1379666741692], [19.7060123749999,
  //   34.8677096054421, 22.1665278051947, 29.5878210025915, 28.0377016623376, 16.1137678474025, 48.667168025974, 32.0334957922077, 48.022053707013, 22.552428327922]]

  regressorData = [
    [
        "metricValue",
        "regressorName",
        "metricName"
    ],
    [
        100.2,
        "LinearRegression",
        "MSE"
    ],
    [
        9.2,
        "LinearRegression",
        "MSE"
    ],
    [
        8.2,
        "LinearRegression",
        "MSE"
    ],
    [
        100.2,
        "LogisticRegression",
        "MSE"
    ],
    [
        90.2,
        "LogisticRegression",
        "MSE"
    ],
    [
        80.2,
        "LogisticRegression",
        "MSE"
    ],
    [
        200.2,
        "ADP",
        "MSE"
    ],
    [
        180.2,
        "ADP",
        "MSE"
    ],
    [
        160.2,
        "ADP",
        "MSE"
    ],
  [
      10.2,
      "LinearRegression",
      "RMSE"
  ],
  [
      9.2,
      "LinearRegression",
      "RMSE"
  ],
  [
      8.2,
      "LinearRegression",
      "RMSE"
  ],
  [
      100.2,
      "LogisticRegression",
      "RMSE"
  ],
  [
      90.2,
      "LogisticRegression",
      "RMSE"
  ],
  [
      80.2,
      "LogisticRegression",
      "RMSE"
  ],
  [
      200.2,
      "ADP",
      "RMSE"
  ],
  [
      180.2,
      "ADP",
      "RMSE"
  ],
  [
      160.2,
      "ADP",
      "RMSE"
  ]
  ]

  constructor() { }

  requestId:string = "";
  currentMetric:string = "MSE";

  chartOptions: EChartsOption = {}
  yAxisData: string[] = ["hello"];

  fetchRawData = () => {
    console.log("i got clicked");
  }

  onKey(event:any) {
    this.requestId = event.target.value;  
    console.log(this.requestId);
  }

  ngOnInit() {
    console.log(this.regressorData);
    let SCRIPT_PATH = 'https://fastly.jsdelivr.net/npm/echarts-simple-transform/dist/ecSimpleTransform.min.js';

    // this.yAxisData = ['ARDRegression', 'AdaBoostRegressor', 'BaggingRegressor'];

    echarts.registerTransform(aggregate as ExternalDataTransform);


    this.chartOptions = {
      title: {
        text: 'Regressors - Mean Squared Error'
      },
      tooltip: {
        trigger: 'axis',
        confine: true
      },
      toolbox:{
        show: true,
        feature:{
          dataView:{
            readOnly: false
          },
          saveAsImage: {}
        }
      },
      dataset: [
        {
          id: 'raw',
          source: this.regressorData
        }, {
          id: 'raw_select_metric',
          fromDatasetId: 'raw',
          transform: [
            {
              type: 'filter',
              config: {
                dimension: 'metricName',
                value: this.currentMetric
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
        name: 'metricName',
        nameLocation: 'middle',
        nameGap: 30,
        scale: true
      },
      yAxis: {
        type: 'category',
        // data: this.yAxisData,
        // splitArea: {
        //   show: true
        // }
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
  }
}



