# LiCSAR2MintPy 使用说明

本项目用于将 LiCSAR 单帧 GeoTIFF 产品转换为 MintPy 可直接进行时序分析的 HDF5 输入文件。完整流程包括两个主要步骤：

1. 使用 `downloadv2.py` 下载 LiCSAR 产品；
2. 使用 `prep_licsar.py` 将 LiCSAR 产品转换为 MintPy 输入文件，并使用 `smallbaselineApp.py` 进行 SBAS 时序反演。

---

## 1. 流程概览

```text
LiCSAR 在线产品
    │
    ├── downloadv2.py
    │       下载 LiCSAR metadata / interferograms / epochs / png 等产品
    │
本地 LiCSAR frame 目录
    │
    ├── prep_licsar.py
    │       将 LiCSAR GeoTIFF 产品转换为 MintPy HDF5
    │       生成 QA 表格和 MintPy 配置文件
    │
MintPy 工作目录
    │
    ├── inputs/ifgramStack.h5
    ├── inputs/geometryGeo.h5
    ├── qa/*.csv
    ├── qa/summary.json
    └── config/mintpy_licsar.cfg
    │
    └── smallbaselineApp.py
            执行 SBAS 时序反演
```

---

## 2. 环境要求

建议在已经安装 MintPy 的 Python 或 conda 环境中运行本流程。

```bash
conda activate insar
```

如需使用下载脚本，建议安装以下依赖：

```bash
pip install requests beautifulsoup4 tqdm
```

其中：

- `requests`：用于访问 LiCSAR 服务器并下载数据；
- `beautifulsoup4`：用于解析网页或 manifest 中的产品链接；
- `tqdm`：用于显示下载进度条；
- `MintPy`：用于读取 GeoTIFF、写出 HDF5，并执行时序分析。

检查 MintPy 命令是否可用：

```bash
smallbaselineApp.py -h
info.py -h
view.py -h
plot_network.py -h
```

---

## 3. LiCSAR 输入目录要求

`prep_licsar.py` 要求输入为单个 LiCSAR frame 目录，目录结构应类似：

```text
<frame_dir>/
├── interferograms/
│   ├── YYYYMMDD_YYYYMMDD/
│   │   ├── YYYYMMDD_YYYYMMDD.geo.unw.tif
│   │   ├── YYYYMMDD_YYYYMMDD.geo.cc.tif
│   │   └── YYYYMMDD_YYYYMMDD.geo.diff_pha.tif   # 可选
│   └── ...
├── metadata/
│   ├── <frame>.geo.E.tif
│   ├── <frame>.geo.N.tif
│   ├── <frame>.geo.U.tif
│   ├── <frame>.geo.hgt.tif
│   ├── <frame>.geo.landmask.tif                 # 可选
│   ├── baselines
│   └── metadata.txt
└── epochs/                                      # 可选
```

MintPy 时序分析所需的最小产品组合为：

```text
unw cc metadata
```

其中：

- `unw`：解缠相位产品；
- `cc`：相干性产品；
- `metadata`：包含几何文件、基线文件、`metadata.txt`、landmask 等信息。

---

## 4. 使用 `downloadv2.py` 下载 LiCSAR 产品

### 4.1 基本下载命令

```bash
python downloadv2.py \
  --frames <FRAME_ID> \
  --dates <START_DATE-END_DATE> \
  --products unw cc metadata \
  --output <OUTPUT_DIR>
```

示例：

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products unw cc metadata \
  --output /data/licsar
```

下载完成后，输出目录通常为：

```text
/data/licsar/
└── 124/
    └── 124A_06996_091406/
        ├── metadata/
        └── interferograms/
```

### 4.2 查询可用轨道

```bash
python downloadv2.py --list-orbits
```

该命令只列出 LiCSAR 服务器中可用的 orbit 编号，不下载数据。

### 4.3 查询某个轨道下的 frame

```bash
python downloadv2.py \
  --orbits 124 \
  --list-frames
```

该命令用于查看指定 orbit 下包含哪些 LiCSAR frame。

### 4.4 查询某个 frame 的影像日期

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --list-epochs
```

该命令用于查看指定 frame 中包含哪些 epoch 日期。

### 4.5 预览下载计划

正式下载前建议使用 `--dry-run` 检查将要下载的文件：

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products unw cc metadata \
  --output /data/licsar \
  --dry-run
```

该命令会打印匹配到的文件、下载 URL 和本地保存路径，但不会真正下载数据。

### 4.6 下载全部匹配产品

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products all \
  --output /data/licsar
```

`all` 会下载所有匹配到的 LiCSAR 产品。若只是进行 MintPy 时序分析，一般下载 `unw cc metadata` 即可。

### 4.7 使用代理或关闭 SSL 验证（推荐，但需本地使用代理）

使用 HTTP 代理：

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products unw cc metadata \
  --output /data/licsar \
  --proxy http://127.0.0.1:7890
```

使用 SOCKS5 代理：

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products unw cc metadata \
  --output /data/licsar \
  --proxy socks5://127.0.0.1:7890
```

如果出现 SSL 证书错误，可以加入 `--no-verify-ssl`：

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products unw cc metadata \
  --output /data/licsar \
  --proxy http://127.0.0.1:7890 \
  --no-verify-ssl
```

### 4.8 `downloadv2.py` 参数说明

| 参数 | 示例 | 说明 |
|---|---|---|
| `--orbits`, `-o` | `--orbits 124` | 指定 orbit 编号。支持逗号分隔或范围形式，如 `1-175`。 |
| `--frames`, `-f` | `--frames 124A_06996_091406` | 指定 LiCSAR frame ID。实际下载时推荐使用该参数。 |
| `--dates`, `-d` | `--dates 20200101-20250101` | 指定日期范围，也可指定单个日期。 |
| `--products`, `-p` | `--products unw cc metadata` | 指定下载产品类型。 |
| `--output`, `-O` | `--output /data/licsar` | 指定本地输出目录。 |
| `--proxy` | `--proxy http://127.0.0.1:7890` | 指定 HTTP 或 SOCKS5 代理。 |
| `--delay` | `--delay 1.0` | 每个 HTTP 请求之间的延迟时间，单位为秒，默认 `0.5`。 |
| `--retries` | `--retries 5` | 每个文件的最大重试次数，默认 `3`。 |
| `--no-resume` | `--no-resume` | 禁用断点续传，重新下载完整文件。 |
| `--timeout` | `--timeout 120` | HTTP 请求超时时间，单位为秒，默认 `60`。 |
| `--no-verify-ssl` | `--no-verify-ssl` | 关闭 SSL 证书验证。 |
| `--list-orbits` | `--list-orbits` | 只列出可用轨道，不下载数据。 |
| `--list-frames` | `--list-frames` | 只列出指定轨道下的 frame。 |
| `--list-epochs` | `--list-epochs` | 只列出指定 frame 的 epoch 日期。 |
| `--dry-run` | `--dry-run` | 只显示下载计划，不实际下载。 |
| `--quiet`, `-q` | `--quiet` | 减少命令行输出。 |

可选产品类型包括：

```text
all unw cc diff_pha metadata dem hgt los_E los_N los_U inc epochs png
```

---

## 5. 使用 `prep_licsar.py` 生成 MintPy HDF5

### 5.1 基本命令

```bash
python prep_licsar.py \
  <frame_dir> \
  --outdir <mintpy_output_dir>
```

示例：

```bash
python prep_licsar.py \
  /data/licsar/124/124A_06996_091406 \
  --outdir /data/licsar/124/124A_06996_091406/mintpy
```

运行完成后会生成：

```text
mintpy/
├── inputs/
│   ├── ifgramStack.h5
│   └── geometryGeo.h5
├── qa/
│   ├── pair_table.csv
│   ├── bad_ifgrams.csv
│   ├── loop_closure_table.csv
│   ├── skipped_pairs.csv
│   └── summary.json
└── config/
    └── mintpy_licsar.cfg
```

### 5.2 推荐命令

```bash
python prep_licsar.py \
  /data/licsar/124/124A_06996_091406 \
  --outdir /data/licsar/124/124A_06996_091406/mintpy \
```

该命令适合作为首次运行的默认配置，给定数据路径和输出路径即可运行。



### 5.3 `prep_licsar.py` 参数说明

| 参数 | 示例 | 说明 |
|---|---|---|
| `frame_dir` | `/data/licsar/124/124A_06996_091406` | LiCSAR frame 根目录。 |
| `--outdir` | `--outdir ./mintpy` | MintPy 输出目录，默认是 `<frame_dir>/mintpy`。 |
| `--wavelength` | `--wavelength 0.05546576` | 雷达波长，单位为米。默认是 Sentinel-1 波长。 |
| `--cc-max` | `--cc-max 255` | 当无法根据 dtype 或 metadata 判断 coherence 缩放方式时使用的兜底除数。 |
| `--min-unw-valid-ratio`, `--min-valid-ratio` | `--min-unw-valid-ratio 0.05` | 可信几何掩膜内有效解缠像素比例下限。 |
| `--min-mean-coherence`, `--min-mean-coh` | `--min-mean-coherence 0.05` | 有效解缠像素上的平均相干性下限。 |
| `--unw-zero-eps` | `--unw-zero-eps 1e-6` | 绝对值小于等于该值的解缠相位会被视为无效。 |
| `--keep-zero-unw-valid` | `--keep-zero-unw-valid` | 将有限的 0 值解缠相位像素视为有效。一般不建议使用。 |
| `--disable-loop-closure` | `--disable-loop-closure` | 关闭 loop closure 筛选。 |
| `--loop-min-valid-ratio` | `--loop-min-valid-ratio 0.02` | 一个闭合环参与评分所需的公共有效像素比例下限。 |
| `--loop-phase-threshold` | `--loop-phase-threshold 3.1415926` | 像素级闭合相位误差阈值，单位为弧度。 |
| `--loop-bad-pixel-ratio` | `--loop-bad-pixel-ratio 0.15` | 闭合环中坏像素比例阈值。 |
| `--loop-rms-threshold` | `--loop-rms-threshold 3.0` | 闭合相位 RMS 阈值，单位为弧度。 |
| `--loop-min-tested-loops` | `--loop-min-tested-loops 2` | 一个干涉图至少被多少个闭合环测试后才允许被剔除。 |
| `--loop-min-bad-loops` | `--loop-min-bad-loops 2` | 一个干涉图至少出现在多少个坏闭合环中才允许被剔除。 |
| `--loop-bad-loop-fraction` | `--loop-bad-loop-fraction 0.5` | 坏闭合环占已测试闭合环的比例阈值。 |
| `--max-loop-count` | `--max-loop-count 10000` | 最多评分的闭合环数量。`0` 表示不限制。 |
| `--force-avg-incidence` | `--force-avg-incidence` | 使用 `metadata.txt` 中的平均入射角。 |
| `--orbit-direction` | `--orbit-direction ASCENDING` | 手动设置轨道方向，可选 `ASCENDING`、`DESCENDING`、`UNKNOWN`。 |
| `--verbose` | `--verbose` | 输出详细日志。 |

---

## 6. 检查 MintPy 输入文件

进入 MintPy 输出目录：

```bash
cd /data/licsar/124/124A_06996_091406/mintpy
```

检查生成文件：

```bash
ls inputs/
ls qa/
ls config/
```

检查 HDF5 文件结构：

```bash
info.py inputs/ifgramStack.h5
info.py inputs/geometryGeo.h5
```

检查干涉图网络：

```bash
plot_network.py inputs/ifgramStack.h5
```

查看几何掩膜：

```bash
view.py inputs/geometryGeo.h5 waterMask
```

查看 QA 结果：

```bash
cat qa/summary.json
cat qa/bad_ifgrams.csv
head qa/pair_table.csv
```

`summary.json` 中需要重点关注：

```text
n_ifg_scanned
n_ifg_loaded
n_ifg_kept
n_ifg_initial_rejected
n_ifg_loop_rejected
n_ifg_skipped
geometry_valid_ratio
ifgram_mask_valid_ratio
```

如果 `n_ifg_kept` 太少，说明保留干涉图数量不足，可能导致 MintPy 网络不连通。此时可降低质量筛选阈值，或关闭 loop closure 后重新运行 `prep_licsar.py`。

---

## 7. MintPy 配置文件设置

`prep_licsar.py` 会生成：

```text
config/mintpy_licsar.cfg
```

同时，脚本已经生成 MintPy 可直接读取的输入文件：

```text
inputs/ifgramStack.h5
inputs/geometryGeo.h5
```

因此，通常不需要再执行 MintPy 的 `load_data` 步骤，建议从 `modify_network` 开始运行。

### 7.1 推荐的 `mintpy_licsar.cfg`

```cfg
# MintPy config for LiCSAR-prepared HDF5
mintpy.load.processor                 = auto
mintpy.load.updateMode                = yes
mintpy.network.coherenceBased  = yes  #[yes / no], auto for no, exclude interferograms with coherence < minCoherence
mintpy.network.minCoherence    = 0.15  #[0.0-1.0], auto for 0.7
mintpy.network.keepMinSpanTree = yes  #[yes / no], auto for yes, keep interferograms in Min Span Tree network
mintpy.reference.minCoherence  = 0.15   #[0.0-1.0], auto for 0.85, minimum coherence for auto method
mintpy.unwrapError.method          = no  #[bridging / phase_closure / bridging+phase_closure / no], auto for no
mintpy.networkInversion.weightFunc      = var #[var / fim / coh / no], auto for var
mintpy.networkInversion.waterMaskFile   = auto #[filename / no], auto for waterMask.h5 or no [if not found]
mintpy.networkInversion.minNormVelocity = auto #[yes / no], auto for yes, min-norm deformation velocity / phase
mintpy.networkInversion.maskDataset   = no #[coherence / connectComponent / rangeOffsetStd / azimuthOffsetStd / no], auto for no
mintpy.networkInversion.minTempCoh  = 0.20 #[0.0-1.0], auto for 0.7, min temporal coherence for mask
mintpy.networkInversion.minNumPixel = 100 #[int > 1], auto for 100, min number of pixels in mask above
mintpy.troposphericDelay.method = no  #[pyaps / height_correlation / gacos / no], auto for pyaps
mintpy.deramp          = linear  #[no / linear / quadratic], auto for no - no ramp will be removed
mintpy.deramp.maskFile = auto  #[filename / no], auto for maskTempCoh.h5, mask file for ramp estimation
mintpy.topographicResidual                   = no  #[yes / no], auto for yes
mintpy.geocode              = no  #[yes / no], auto for yes
```

### 7.2 关键配置说明

#### `mintpy.unwrapError.method = no`

`prep_licsar.py` 不生成假的 `connectComponent`。在没有真实 connected component 信息的情况下，不建议开启 MintPy 中依赖连通分量的解缠误差校正。潜在坏干涉图已经在预处理阶段通过覆盖率、相干性和可选 loop closure 进行筛选。

#### `mintpy.network.coherenceBased = yes`

`prep_licsar.py` 已经将 LiCSAR 相干性写入 `ifgramStack.h5`，因此 MintPy 可以根据 coherence 辅助网络筛选。

#### `mintpy.network.minCoherence = 0.2`

该参数用于网络筛选。`0.2` 是较常用的初始值。低相干区域可降至 `0.1`，高质量数据可提高到 `0.3`。

#### `mintpy.reference.minCoherence = 0.3`

参考点应选择稳定且相干性较高的像素，因此参考点相干性阈值建议略高于网络筛选阈值。如果研究区整体相干性较低，可改为 `0.2`。

#### `mintpy.networkInversion.maskDataset = coherence`

时序反演时使用 coherence 掩膜低质量像素。如果有效像素过少，可降低阈值：

```cfg
mintpy.networkInversion.maskThreshold = 0.1
```

或关闭该掩膜：

```cfg
mintpy.networkInversion.maskDataset = no
```

#### `mintpy.troposphericDelay.method = no`

确认基础流程正常后，可根据需要加入 ERA5、GACOS 等大气校正方法。

#### `mintpy.topographicResidual = no`

首次运行建议关闭 DEM 残余误差校正，以避免几何字段不完整或兼容性问题。

#### `mintpy.deramp = linear`

该设置用于去除线性坡度。对于大范围 LiCSAR frame，线性坡度可能来自轨道残差、大气长波误差或其他长波误差。若研究区存在真实长波形变，应对比 `linear` 和 `no` 两种设置。

#### `mintpy.geocode = no`

LiCSAR `.geo.*.tif` 产品本身已经是地理编码产品，`prep_licsar.py` 生成的也是 `geometryGeo.h5`，因此通常不需要 MintPy 再执行 geocode。

---

## 8. 运行 MintPy 时序分析

进入 MintPy 工作目录：

```bash
cd /data/licsar/124/124A_06996_091406/mintpy
```

推荐从 `modify_network` 开始运行：

```bash
smallbaselineApp.py config/mintpy_licsar.cfg --start modify_network
```

这样设置的原因是：`prep_licsar.py` 已经生成了 `inputs/ifgramStack.h5` 和 `inputs/geometryGeo.h5`，因此通常不需要再运行 `load_data`。

只运行某一个步骤：

```bash
smallbaselineApp.py config/mintpy_licsar.cfg --dostep velocity
```

指定起止步骤：

```bash
smallbaselineApp.py config/mintpy_licsar.cfg \
  --start modify_network \
  --stop invert_network
```

从某一步继续运行：

```bash
smallbaselineApp.py config/mintpy_licsar.cfg --start deramp
```

---

## 9. 完整示例

### 9.1 下载数据

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products unw cc metadata \
  --output /data/licsar \
  --retries 5 \
  --timeout 120
```

### 9.2 生成 MintPy HDF5

```bash
python prep_licsar.py \
  /data/licsar/124/124A_06996_091406 \
  --outdir /data/licsar/124/124A_06996_091406/mintpy \
  --min-unw-valid-ratio 0.05 \
  --min-mean-coherence 0.05 \
  --orbit-direction ASCENDING \
  --verbose
```

### 9.3 检查预处理结果

```bash
cd /data/licsar/124/124A_06996_091406/mintpy

info.py inputs/ifgramStack.h5
info.py inputs/geometryGeo.h5
plot_network.py inputs/ifgramStack.h5
view.py inputs/geometryGeo.h5 waterMask
cat qa/summary.json
cat qa/bad_ifgrams.csv
```

### 9.4 运行 MintPy

```bash
smallbaselineApp.py config/mintpy_licsar.cfg --start modify_network
```

### 9.5 查看结果

```bash
view.py velocity.h5 velocity
view.py temporalCoherence.h5
tsview.py timeseries.h5
```

---



## 10. 推荐默认策略

首次完整运行建议使用：

```bash
python downloadv2.py \
  --frames 124A_06996_091406 \
  --dates 20200101-20250101 \
  --products unw cc metadata \
  --output /data/licsar

python prep_licsar.py \
  /data/licsar/124/124A_06996_091406 \
  --outdir /data/licsar/124/124A_06996_091406/mintpy \

cd /data/licsar/124/124A_06996_091406/mintpy
smallbaselineApp.py config/mintpy_licsar.cfg --start modify_network
```

推荐原则：

- 至少下载 `unw cc metadata`；
- 预处理后检查 `qa/summary.json` 和 `qa/bad_ifgrams.csv`；
- 保持 `mintpy.unwrapError.method = no`；
- 从 `modify_network` 开始运行 MintPy；
- 根据网络连通性和结果质量调整 QA 阈值。
