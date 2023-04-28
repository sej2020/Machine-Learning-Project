import type { ExternalDataTransform } from "@manufac/echarts-simple-transform";

/**
 * Needed because of: https://github.com/ecomfe/echarts-stat/issues/35
 * Module augmentation: https://www.typescriptlang.org/docs/handbook/declaration-merging.html#module-augmentation
 */
declare module "echarts-stat" {
  let transform: {
    regression: ExternalDataTransform;
    histogram: ExternalDataTransform;
    clustering: ExternalDataTransform;
  };
}
declare module 'echarts-gl/charts';