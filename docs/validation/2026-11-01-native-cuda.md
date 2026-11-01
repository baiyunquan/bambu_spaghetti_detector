# Nano 原生 CUDA 推理验证记录（2026-11-01）

已实现并部署三模型原生后端，保留 Python 3.10 主控、Python 3.6 worker、现有 UDS 协议及旧 torch 后端。
用户追加要求：暂停目标包含 spaghetti、warping、error；取消三个模型各 1 小时测试。
本次没有完成长期泄漏/swap 稳定性验证，也没有把中断的短回放记为一小时通过。

## 原因与实现

内存主要来自 CUDA 框架及模块初始化，图片缓冲仅占数 MiB。分段测量中，
使用 OpenCV CUDA stream 令 RSS 额外增加约 500 MiB；使用 Driver 主 context 时，
同一 YOLOv5 engine 从约 1.52 GiB 降至约 1.03 GiB。相比旧 torch 链路，
最终四组基准节省 1.53–2.08 GiB。保留 Python 主控即可达到本轮目标。

采用计划允许的小型 CUDA 预处理替代方案：OpenCV 主机 JPEG 解码，CUDA Driver
保留设备主 context，单 stream、固定 TensorRT/NMS 工作区；当前图片尺寸变化时同步后
替换输入缓冲，不保存历史尺寸。GPU 内核完成兼容 OpenCV CPU 舍入的 resize、letterbox、
RGB/归一化及连续 NCHW。原始 OpenCV CUDA resize 与 CPU 采样不一致的回归先失败后修复。
NVRTC 离线编译 compute_53/C++11 PTX，worker 不现场编译，不需要旧 GCC。

Obico、YOLOv8、YOLOv5 分别解码，v5 使用 objectness × 类别概率并保留四类原始编号。
engine/元数据 SHA256、绑定形状与 FP32 LINEAR 类型启动时校验。
GPU NMS 保持 >= 置信门槛、最高类别、同分候选编号顺序、同类 IoU > 门槛抑制，
不截断候选；25,200 个有效候选专项测试全部保留。
坏帧/非有限结果返回错误，主控清空连续命中证据，不把错误当作正常零检测。

## 测量

同一 engine、阈值、图片与环境，旧/新后端分进程串行测量，每组 100 帧。
P95 忽略最初 10 帧；完整 RSS/PSS、启动、可用内存、swap 记录见配套 JSON 和原始日志。

| engine | RSS GiB（旧 → 新） | 节省 GiB | P95 ms（旧 → 新） |
|---|---:|---:|---:|
| obico | 2.56 → 1.03 | 1.53 | 112.4 → 104.6 |
| yolov8_640 | 2.95 → 1.41 | 1.54 | 274.3 → 273.5 |
| yolov5_640 | 3.11 → 1.03 | 2.08 | 98.9 → 97.0 |
| yolov5_416 | 3.06 → 1.03 | 2.03 | 56.4 → 54.9 |

四组均满足稳态 RSS ≤1.5 GiB、启动峰值 ≤1.8 GiB、节省至少 800 MiB、
P95 劣化不超过 10%。原生基准进程 swap 均为 0；旧后端出现进程 swap。

## 正确性与部署后短测

- Obico416、YOLOv8 416/640、YOLOv5 416/640 各 295 张，共 1,475 次对照通过。
  RGB/NCHW 最大像素差为 0；框坐标、置信度、类别及多帧触发结果均满足规定容差。
- 五组真实相机 UDS 测试均通过默认 5 秒超时；720p 和 1536×1080 全链路预热后才监听。
- Nano GPU 专项 7 项通过；本地主机 70 项通过、11 项环境相关跳过、2 项子测试通过。
  部署代码通过 Python 3.6 compileall，主控实际短测运行于 Python 3.10。
- 部署后启动 8.02 秒；真实相机首帧 78.9 ms，连续 20 次请求最慢 87.8 ms。
  RSS 1.035 GiB，PSS 0.963 GiB，进程 swap 0。
- 坏 JPEG 明确报错、客户端重连后恢复；spaghetti/warping/error 各经 6 帧才触发，
  extrusor/part 不触发。进程动态库映射不含 torch/torchvision/libtorch/libc10。
  worker 正常释放并退出，未留下 socket。

## 部署与回滚

远程目录 `/home/jetson/Spaghetti_Failure_Detection` 已启用 `native_cuda`。
保留当前 YOLOv5 416、confidence=0.30、consecutive_hits=6、dry_run=true。
三份新 engine 在暂存目录串行构建后复制，原有 Obico/v5 engine 哈希保持一致。
五份元数据已纳入版本管理；部署后 42 份源文件/PTX 及五组 engine/元数据均与冻结输入一致。
两份 AGENTS.md 允许原生 CUDA GPU NMS 和拥有明确生命周期的 CUDA 输入缓冲。

备份：`/home/jetson/spaghetti-native-validation/backup-native-20261101-d1f6f11b/`，
包含原配置、原部署文件及两份原 engine 的 tar 归档和哈希清单。
需要回退后端时，先停止当前 worker，再恢复该目录下 `config.json` 到项目目录；
旧配置缺省使用 torch。未启用打印控制或启动此前处于 inactive 的主控服务。

原始记录位于远程 `spaghetti-native-validation/logs/acceptance-v4/`，本地副本位于
工作区 `diagnostics/native-memory/remote-logs/acceptance-v4/`。
`result-user-amended.json` 明确区分此次部署门槛通过和未完成的一小时扩展验收。
