# 车外数据集整理 (Exterior Dataset Manifest)
*(Visteon Cabin VLM · 车外数据线 · 袁傲杰)*

Qwen3-VL 微调用的车外(out-of-vehicle)训练 + held-out 数据清单。所有 GT 溯源、许可、下载出处、泄漏审计一表说清。

---

## 1. 训练数据块(sharegpt 格式)

格式:`{"conversations":[{"from":"human","value":"<image>\n..."},{"from":"gpt","value":"..."}], "images":[路径]}`

| 能力块 | 样本 | 图数 | 源数据集 | GT 来源 | 许可 |
|---|---|---|---|---|---|
| `exterior_cot_v4` | 2424 | 2424 | nuScenes CAM(6相机) | VLA CoT(Scene/Risk/Decision,AI 交叉核查 80% 忠实) | CC BY-NC-SA 4.0 |
| `cars` | 1997 | 1997 | Stanford Cars | make/model 标签引导 | 研究用 |
| `signs` | 1497 | 1497 | GTSRB(欧洲标志) | 类别标签 | CC0 |
| `tt100k_china` | 600 | 600 | TT100K(中国标志) | 类别→含义映射 | 研究用 |
| `textvqa` | 1500 | 1499 | TextVQA | 路牌/广告 OCR q+a | CC BY 4.0 |
| `landscape` | 778 | 778 | SUN397 户外 | 景观类型标签 | 研究用 |
| `trafficlight` | 142 | 142 | road-traffic | 红绿灯状态标签 | 研究用 |
| `jaad_vru` | 270 | 270 | JAAD | 逐帧 `cross` 行为属性 | 研究用(YorkU) |
| **合计** | **~9208** | **~9200** | — | — | — |

展平为 (image, q, a) 后:**31,631 条训练 QA**(多轮 sharegpt 拆单轮)。

## 2. 增强数据块(修 bias,YOLO/行为 GT)

| 块 | 条数 | 源 | 用途 |
|---|---|---|---|
| JAAD 横穿平衡负例 | 400帧×(yes/no) | JAAD clips 1–59 | 修 VRU 横穿 yes-bias(v24) |
| YOLO presence/计数 | 10,500 | ft_imgs 训练图 | 修行人 presence yes-bias(v26) |
| BDD100K 多样 | 10,500 | BDD100K 3500图 | 强化跨数据集泛化(v27) |

## 3. Held-out(评测,全部泄漏审计)

| held-out | 数量 | 源 | GT | 轴 | 泄漏 |
|---|---|---|---|---|---|
| nuScenes sweeps | 12,184 | nuScenes-mini sweeps | YOLOv8x 客观 | **in-domain** | ∩train=0 |
| nuImages-mini | 650 | nuImages-mini | YOLOv8x 客观 | **cross-dataset 同国** | 不同数据集 |
| `china_test` | 40 | TT100K held-out | 标志含义 | 中国标志 | ∩train=0 |
| `vru_test` | 40 | JAAD clips 60+ | crossing yes/no | 行人横穿 | ∩train=0 |
| `heldout_frozen` | 200 | 5类×40 | 分类(judge) | frozen 综合 | lab 服务器 |

## 4. 下载出处(可复现,均免登录直连)

- **nuScenes-mini**:`https://www.nuscenes.org/data/v1.0-mini.tgz`(4.2GB,含 samples+sweeps)
- **nuImages-mini**:`https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz`(118MB)
- **JAAD**:视频 `http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip`(3.1GB);标注 `github.com/ykotseruba/JAAD`
- **BDD100K**:HF 镜像 `dgural/bdd100k`(20k 图)
- **TT100K / GTSRB / Stanford Cars / TextVQA / SUN397**:各官方页(研究用)

## 5. 存储

- 打包 tar 在团队网盘 `网盘_数据集/`:`cabin_captions.tar.gz`(全部 sharegpt 标注,核心)、`cabin_source_imgs_used.tar.gz`(源图)、`cabin_derived_imgs.tar.gz`(生成图:JAAD抽帧/TT100K/交通灯/测试切片)、`cabin_code_docs.tar.gz`。
- **许可红线**:nuScenes/SVIRO/Stanford Cars 等为非商业/研究许可 → **不入公开 git**,只在网盘/yuque 内部流转。代码库仅存生成/训练/评测**代码**。

## 6. GT 溯源原则(零幻觉)
所有标注只从 authoritative GT 生成(标签/文件名/行为属性),不从视觉臆测;enum 约束 + `is_ground_truth` 溯源;held-out 一律 basename 泄漏审计。
