# TDI + CatBoost 地表温度重建

用于城市 ECOSTRESS 地表温度（LST）的缺口重建。程序以热扩散指数（TDI）和环境特征训练 CatBoost，保留有效观测，仅填补研究区内、预测特征完整的缺测像元。

实现依据见文末论文及[复现说明](CatBoost_TDI/docs/REPRODUCIBILITY.md)。

## 环境

已在 Python 3.10 环境验证。新建conda环境并安装需要的包：

```powershell
Set-Location 'E:\ENVI\2025.6_TDI'
conda create -n TDI python=3.10
conda run -n TDI python -m pip install -r requirements.txt
```

## 数据

预处理入口接收独立的 LST、16 项特征栅格及研究区边界。默认模型使用九项特征：`NDVI, NDBI, Albedo, EULUC, DEM, AH, VNL, LCZ, TDI`。

使用随项目提供的数据包时，入口为根目录下的 `assets/workflow.json`，目录说明见 [assets/README.md](assets/README.md)。自备数据可参照[示例清单](CatBoost_TDI/examples/data_manifest.json)配置路径、波段和标定参数。相对路径按清单所在目录解析；示例清单通过 `../../assets` 引用项目数据。

数据格式、质量规则及重采样设置见[预处理说明](CatBoost_TDI/docs/PREPROCESSING.md)。流程从已导出的 LST 和特征产品开始；原始卫星产品解包及上游分类、城市参数生产需另行准备。

## 运行

从源栅格完成预处理与重建：

```powershell
conda run -n TDI python -m CatBoost_TDI workflow --manifest assets/workflow.json --config CatBoost_TDI/configs/paper_model.json --output output/reconstruction
```

清单路径可替换为编辑后的示例清单。快速检查单景时，可追加 `--scenes 2021_07_27_T14_54_06 --iterations 64 --depth 5 --threads 2 --max-samples 2000`；该设置用于检查流程。

已有符合波段约定的 17 波段数据时，可直接重建：

```powershell
conda run -n TDI python -m CatBoost_TDI reconstruct --input path/to/stack.tif --config CatBoost_TDI/configs/paper_model.json --output output/reconstruction_stack
```

其他命令包括 `inspect`、`discover`、`prepare`、`tdi`、`distance`、`validate` 和 `select-features`。`discover` 默认从 `assets/workflow.json` 选择场景并写出清单，例如：

```powershell
conda run -n TDI python -m CatBoost_TDI discover --scenes 2021_07_27_T14_54_06 --output output/scene_manifest.json
conda run -n TDI python -m CatBoost_TDI --help
```

在子命令后加 `--help` 可查看参数。已有输出默认保留，重跑时可更换目录或显式加 `--overwrite`。

## 输出

每景重建生成 `*_gapless.tif`、`*_qa.tif`、`*_model.cbm` 和 `*_report.json`，分别保存 LST、质量标识、模型和运行配置。重建 QA 编码为 `0=范围外、1=原始观测、2=已填补、3=仍缺失`。缺少预测特征的缺口保留为空。

`workflow` 另保存预处理 LST、17 波段栈、预处理 QA 和输入记录。LST 单位由清单明确指定；直接重建已有波段栈时保留输入单位。

## 代码结构

| 目录 | 功能 |
|---|---|
| `CatBoost_TDI/common/` | 配置、路径与栅格读写 |
| `CatBoost_TDI/preprocessing/` | 标定、质量控制、配准与波段合成 |
| `CatBoost_TDI/thermal/` | 扩散距离与 TDI |
| `CatBoost_TDI/reconstruction/` | CatBoost 训练和缺口填补 |
| `CatBoost_TDI/evaluation/` | 验证、SHAP、特征选择与消融 |
| `CatBoost_TDI/configs/`、`CatBoost_TDI/examples/` | 模型参数与数据清单 |
| `CatBoost_TDI/tests/` | 数值、栅格处理及重建功能测试 |
| `CatBoost_TDI/provenance/` | 原始代码块、表达式及出处记录 |
| `assets/` | 数据、目录索引与运行清单 |

流程入口为[预处理](CatBoost_TDI/preprocessing/pipeline.py)和[重建](CatBoost_TDI/reconstruction/pipeline.py)。公共配置可通过 `from CatBoost_TDI import ModelConfig` 导入。项目根目录和默认数据目录根据代码位置计算，整目录移动后无需修改绝对路径。

## 测试与复现范围

```powershell
conda run -n TDI python -m unittest discover -s CatBoost_TDI/tests -v
```

测试检查数值和数据处理行为。详见[复现与验证](CatBoost_TDI/docs/REPRODUCIBILITY.md)。

## 引用

Luo and Ren (2026). *A thermal diffusion-guided method for filling gaps in urban ECOSTRESS land surface temperature during summer*. Building and Environment, 287, 113920. [doi:10.1016/j.buildenv.2025.113920](https://doi.org/10.1016/j.buildenv.2025.113920).
