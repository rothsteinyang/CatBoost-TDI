# 复现与验证

## 模型设置

[默认配置](../configs/paper_model.json)采用 8000 棵上限、学习率 0.08、深度 10；训练与早停验证按 8:2 划分，划分种子为 42、模型种子为 0，早停轮数为 100。九项特征的顺序见 [common/config.py](../common/config.py)。

分类变量沿用原代码的数值编码。`categorical_features` 可显式启用原生分类处理；`--refit-all` 可按选定树数重训全部可用样本。这些选项会改变默认训练协议。早停指标用于模型选择，不作为独立测试指标。

## TDI 实现

[thermal/tdi.py](../thermal/tdi.py)按论文公式计算类别温度权重、尺度聚合和指数模型普通克里金。默认扩散距离为 350 m；LC 需使用米制投影，扩散距离应为其像元边长的整数倍。`--scale-m` 必须显式指定，与扩散距离分别设置。

半变异函数为 `gamma(h>0)=nugget+partial_sill*(1-exp(-h/scale_m))`，95% 实用变程约为 `3*scale_m`。ArcGIS 参数定义见 [KrigingModelOrdinary](https://desktop.arcgis.com/en/arcmap/latest/analyze/arcpy-spatial-analyst/krigingmodelordinary-class.htm) 和 [Kriging](https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/kriging.htm)。

## 验证协议

`validate` 提供随机、空间和云掩膜验证。空间模式默认按列划分条带；原实验分区可通过 `--blocks` 指定。云掩膜约定 `0=晴空、1=模拟云、NoData=未知`，只对原有观测且特征完整的模拟云像元评分。

外层测试样本不参与早停。既有 TDI 是否使用了测试区域温度尚未核实，因此使用该 TDI 的分数属于给定特征条件下的模型验证。若评估完整流程，需仅用训练可见的 LST 重建 TDI；Python 验证接口提供 `feature_builder(visible_mask)` 回调，TDI 命令提供 `--visible-mask`。

[回归测试](../tests)覆盖观测保留、缺测处理、标定、网格、验证可见性、SHAP/消融和热扩散计算，使用独立的合成输入运行。试跑采用缩减参数，功能检查结果不等同于论文全部实验的精度复现。

原始单元格、预处理表达式和出处记录保存在 [provenance](../provenance/manifest.json)及其[预处理归档](../provenance/preprocessing/manifest.json)。其中记录的历史路径仅表示原始来源；运行入口和功能测试使用当前项目中的代码与数据。论文引用见根目录 [README](../../README.md)。
