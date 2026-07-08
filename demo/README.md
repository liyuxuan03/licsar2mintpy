# LiCSAR to MintPy Processing Workflow

本项目演示如何将单个 LiCSAR frame 的 GeoTIFF 产品整理为 MintPy 可直接使用的 HDF5 输入，并继续运行 `smallbaselineApp.py` 完成时序 InSAR 处理。

## 项目内容

```text
.
├── prep_licsar.py
├── 指令.docx
└── 106A_09048_000909/
    ├── epochs/
    ├── interferograms/
    ├── metadata/
    └── mintpy/
        ├── inputs/
        ├── config/
        ├── qa/
        └── pic/
```

核心脚本为 `prep_licsar.py`，用于把 LiCSAR frame 目录转换为 MintPy 输入文件：

- `mintpy/inputs/ifgramStack.h5`
- `mintpy/inputs/geometryGeo.h5`
- `mintpy/config/mintpy_licsar.cfg`
- `mintpy/qa/*.csv` 和 `mintpy/qa/summary.json`

脚本会对干涉图进行覆盖率、平均相干性和闭合环误差筛查。被剔除的干涉图会记录在 QA 文件中，并写入 `ifgramStack.h5:/dropIfgram`。

## 环境依赖

需要先准备可运行 MintPy 的 Python 环境。脚本直接依赖：

- Python 3
- NumPy
- MintPy

确认 MintPy 命令可用：

```bash
python -c "import mintpy; print(mintpy.__version__)"
smallbaselineApp.py -h
```

如果需要从 LiCSAR 下载数据，还需要准备你自己的 `downloadv2.py`。该下载脚本没有包含在本项目根目录中。

## 1. 下载 LiCSAR 数据

以下命令来自原始 Word 指令，示例 frame 为 `106A_09048_000909`。

```bash
python downloadv2.py \
  --frames 106A_09048_000909 \
  --dates 20240101-20240215 \
  --products epochs cc diff_pha unw \
  --proxy http://127.0.0.1:7892 \
  --no-verify-ssl
```

如需下载全部产品，可使用：

```bash
python downloadv2.py \
  --frames 106A_09048_000909 \
  --dates 19000101-19000102 \
  --products all \
  --proxy http://127.0.0.1:7892 \
  --no-verify-ssl
```

下载后，frame 目录应至少包含：

```text
<frame_dir>/
├── interferograms/
│   └── YYYYMMDD_YYYYMMDD/
│       ├── YYYYMMDD_YYYYMMDD.geo.unw.tif
│       ├── YYYYMMDD_YYYYMMDD.geo.cc.tif
│       └── YYYYMMDD_YYYYMMDD.geo.diff_pha.tif
└── metadata/
    ├── <frame>.geo.E.tif
    ├── <frame>.geo.N.tif
    ├── <frame>.geo.U.tif
    ├── <frame>.geo.hgt.tif
    ├── <frame>.geo.landmask.tif
    ├── baselines
    └── metadata.txt
```

## 2. 生成 MintPy 输入文件

进入项目根目录：

```bash
cd /path/to/demo
```

运行预处理脚本：

```bash
python prep_licsar.py \
  106A_09048_000909 \
  --outdir 106A_09048_000909/mintpy
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--outdir` | `<frame_dir>/mintpy` | 输出目录 |
| `--min-unw-valid-ratio` | `0.05` | 干涉图有效解缠像元比例阈值 |
| `--min-mean-coherence` | `0.05` | 有效解缠像元内平均相干性阈值 |
| `--disable-loop-closure` | 关闭 | 禁用闭合环筛查 |
| `--max-loop-count` | `0` | 最大闭合环检查数量，`0` 表示不限制 |
| `--force-avg-incidence` | 关闭 | 使用 `metadata.txt` 中的平均入射角 |
| `--orbit-direction` | `UNKNOWN` | 手动指定轨道方向，可选 `ASCENDING`、`DESCENDING`、`UNKNOWN` |
| `--verbose` | 关闭 | 输出更详细日志 |

## 3. 生成并修改 MintPy 配置

进入 MintPy 输出目录：

```bash
cd 106A_09048_000909/mintpy
```

生成完整配置文件：

```bash
smallbaselineApp.py -g
```

然后根据需要修改 `smallbaselineApp.cfg`。推荐参数片段如下：

```ini
# MintPy config for LiCSAR-prepared HDF5
mintpy.load.processor                 = auto
mintpy.load.updateMode                = yes
mintpy.network.coherenceBased         = yes
mintpy.network.minCoherence           = 0.15
mintpy.network.keepMinSpanTree        = yes
mintpy.reference.minCoherence         = 0.15
mintpy.unwrapError.method             = no
mintpy.networkInversion.weightFunc    = var
mintpy.networkInversion.waterMaskFile = auto
mintpy.networkInversion.minNormVelocity = auto
mintpy.networkInversion.maskDataset   = no
mintpy.networkInversion.minTempCoh    = 0.20
mintpy.networkInversion.minNumPixel   = 100
mintpy.troposphericDelay.method       = no
mintpy.deramp                         = linear
mintpy.deramp.maskFile                = auto
mintpy.topographicResidual            = no
mintpy.geocode                        = no
```

说明：

- `prep_licsar.py` 已经生成 `inputs/ifgramStack.h5` 和 `inputs/geometryGeo.h5`。
- 脚本不会生成假的 `connectComponent`。
- 在没有真实连通域文件或经过验证的解缠误差修正流程前，建议保持 `mintpy.unwrapError.method = no`。

## 4. 运行 MintPy

在 `mintpy` 目录下运行：

```bash
smallbaselineApp.py
```

运行完成后，重点检查：

```text
mintpy/
├── velocity.h5
├── timeseries.h5
├── temporalCoherence.h5
├── maskTempCoh.h5
├── qa/
└── pic/
```

## 示例 QA 结果

当前示例 `106A_09048_000909` 的 `mintpy/qa/summary.json` 显示：

| 项目 | 数值 |
| --- | ---: |
| 扫描干涉图数量 | 10 |
| 成功载入干涉图数量 | 10 |
| 最终保留干涉图数量 | 8 |
| 初筛剔除数量 | 0 |
| 闭合环剔除数量 | 2 |
| 跳过干涉图数量 | 0 |

闭合环筛查剔除的干涉图记录在 `mintpy/qa/bad_ifgrams.csv` 中。

