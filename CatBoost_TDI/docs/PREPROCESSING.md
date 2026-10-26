# 输入与预处理

## 数据清单

[示例清单](../examples/data_manifest.json)采用 `schema_version=1`，通过 `../../assets` 引用项目数据，相对路径按清单目录解析。

项目自带的运行清单位于 [assets/workflow.json](../../assets/workflow.json)。`discover` 从数据目录的 `workflow.json`（或 `catalog.json` 指定的清单）选择场景并写出新清单，默认数据目录为项目根目录下的 `assets/`。

| 字段 | 内容 |
|---|---|
| `roi`、`crop_to_roi` | 研究区多边形及裁剪设置 |
| `static_features` | 共用的静态特征 |
| `scenes[].id`、`lst` | 场景标识及 LST 源 |
| `scenes[].reference` | 输出网格参考；缺省使用该景 LST |
| `scenes[].features` | 逐年或逐景特征，与静态特征合计覆盖全部 16 项 |
| `scenes[].lst_options` | 温度单位、阈值、云外扩及质量规则 |

每个栅格源可指定 `path`、`band`、`nodata`、`resampling` 和标定参数，波段编号从 1 起。`scale/offset` 与 `use_metadata_scale=true` 二选一；默认 scale 为 1、offset 为 0。缺测值在标定和插值前清理。默认排除原始值 `<= -200`，可用 `raw_minimum=null` 禁用此规则。

已校准的摄氏度 LST 应明确设为 `input_unit=C`、`output_unit=C`。新产品可配置 `cloud_mask`、`quality`、`zenith`；QC 必须给出允许值或拒绝位掩码。未提供的质量数据不会自动补齐；`already_screened` 只记录上游处理声明。

## 网格与变量

栅格配准到每景参考网格，ROI 按像元中心纳入。默认重采样方法如下，也可在源配置中显式覆盖。

| 方法 | 变量 |
|---|---|
| 双线性 | NDVI、NDWI、NDBI、Albedo、Emissivity、DEM、TH |
| 最近邻 | LST、LC、EULUC、LCZ、BH、BD、SVF、TDI |
| 三次卷积 | AH、VNL |

云掩膜和 QC 始终用最近邻。云掩膜覆盖范围外视为未知并排除；阈值采用输出温度单位，`dilation_pixels=0` 禁用外扩。已有历史 LST 清单不重复应用未核实的云阈值。

合成顺序为：

```text
LST, NDVI, NDWI, NDBI, Albedo, Emissivity, LC, EULUC,
DEM, TH, BD, BH, SVF, AH, VNL, LCZ, TDI
```

可在场景中提供 `spectral_bands`，显式映射 `B02/B03/B04/B08/B11`，生成 NDVI、NDWI、NDBI 和 Albedo。指数在光谱网格计算后再重采样到 LST 网格；需自行核对源产品标定和获取日期。

## 输出

`prepare` 保存 `stacks/`、`preprocessing/` 和场景索引；`workflow` 另写 `reconstruction/`。报告记录实际输入、标定、网格、重采样及执行过的质量规则。

预处理 QA 为 `0=保留、1=原始缺失、2=云或掩膜未知、3=QC、4=角度、5=阈值、6=外扩、255=范围外`。该编码与重建 QA 不同。输入文件保持只读。

标定属性定义参见 [GDAL RasterBand](https://gdal.org/en/stable/doxygen/classGDALRasterBand.html)。
