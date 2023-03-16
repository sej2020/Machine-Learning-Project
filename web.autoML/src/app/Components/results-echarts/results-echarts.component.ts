import { Component } from '@angular/core';
import { EChartsOption, registerTransform} from 'echarts';
@Component({
  selector: 'app-results-echarts',
  templateUrl: './results-echarts.component.html',
  styleUrls: ['./results-echarts.component.scss']
})
export class ResultsEchartsComponent {


  chartOptions: EChartsOption ={}
  yAxisData: string[] = [];
  ngOnInit(){
  
    this.yAxisData = ['ARDRegression', 'AdaBoostRegressor', 'BaggingRegressor'];
    // echarts.registerTransform();
    this.chartOptions = {
      title:{
        text: 'Regressors - Mean Squared Error'
      },
      tooltip:{
        trigger: 'axis',
        confine: true
      },
      dataset: [
     {
        transform: {
            type: 'boxplot',
            config: { 
                itemNameFormatter: 'Option {value}' }
        }
    }, {
        fromDatasetIndex: 1,
        fromTransformResult: 1
    }],
      xAxis:{
        name : 'CV Folds',
        nameLocation: 'middle',
        nameGap: 30,
        scale: true,

      },
      yAxis:{
        type: 'category',
        data : this.yAxisData,
        splitArea:{
          show : true
        }

      },
      legend: {

        selected: { detail: false }
      },
      grid: {
        bottom: 100
      },
      series: [
        { 
          type: 'boxplot',
          name: 'boxplot',
          itemStyle: {
            color: '#a32',
            borderColor: '#429',
            borderWidth: 3
        },
        datasetIndex: 1,
        data : [[134.744,101.139,107.769,83.486,103.296,103.997,96.735,91.425,139.113,118.779],[61.9045328587085,52.5798970391187,60.320216595453,61.8208707545006,59.9380889894142,45.8944453371561,55.2555857413487,81.7090951014166,74.4910264391988,61.1379666741692],[19.7060123749999,
          34.8677096054421,
          22.1665278051947,
          29.5878210025915,
          28.0377016623376,
          16.1137678474025,
          48.667168025974,
          32.0334957922077,
          48.022053707013,
          22.552428327922
          ]],
        
}],
    };
  }
}
  
