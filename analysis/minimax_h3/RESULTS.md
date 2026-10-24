# MiniMax H3 上的 FreeFuse 探索（2026-10-24）

这次要回答的是：两套角色 LoRA 同时加载到 H3 后，FreeFuse 能否减少特征互相污染。先在韩国 DSW 的独立环境中跑通原生 H3，再固定提示词、种子和采样设置做有无 FreeFuse 的比较。**目前证据支持“在这组轻微动作的双角色视频中可以生效”，还不足以宣布通用的视频支持。**

第一段对照加载 Starfire 与 Albedo 两套 LoRA，各匹配 200 个 H3 原生目标层。不开 FreeFuse 时，Starfire 在种子 42 和 2026 中都出现了 Albedo 的白角，眼睛、手套和服装也有混合。随后分别只加载一套 LoRA 生成单人参考；这些参考展示各自的外观，但提示词不同，不是逐像素对齐的真值。Starfire 来自 Teen Titans 角色/画风 LoRA，并非专门只训练 Starfire 的权重。

H3 的文本、参考、音频、视频共用一条序列，线性层输入也是二维序列。因此这次没有直接套用图像模型的 token 长度假设，而是从原生 PackedLayout 取目标视频片段，只在该片段施加空间/时间 LoRA 遮罩和视频到角色文本的注意力约束。原生 QK 归一化、旋转位置编码和采样器保持原实现。为了避免为两万多个 token 构造完整的注意力偏置矩阵，将偏置分解到额外 Q/K 通道；数值检查覆盖了实际使用的 128 维注意力头及 BF16。

为了区分“区域约束无效”和“自动区域不准”，先跑了手工左右区域的诊断。Starfire 的白角消失，绿眼、紫色服装和画风更接近单角色参考。这说明约束机制可以发挥作用，但这段手工结果不能算作自动区域结果。

自动区域来自关闭两套 LoRA 的 H3 原生特征采集。检查了 3 个去噪索引、4 个层和 5 个温度，共 60 组遮罩。很多候选会混淆角色、产生大片背景归属或随时间改变身份，说明直接迁移图像参数不可靠。根据种子 42 的遮罩可视化，选择去噪索引 3、层索引 20、特征相似度温度 100000、3 个潜在时间位置的平均平滑。这些都是从零计数的索引；温度用于特征相似度的空间 softmax。最终遮罩为 37 × 24 × 24，没有手工补画、交换标签或空遮罩回退。

使用这组自动遮罩后，种子 42 的联合版本消除了 Starfire 的明显串角，绿眼与原画风也更接近参考。开关对照中，只开区域 LoRA 已有明显改善；只开注意力约束仍保留眼睛、手套和画风的混合；联合版本在这段样例中更接近两套单角色参考。构图也发生了变化，例如人物大小与前后关系，所以不能将结果描述成在完全相同构图下只修改了身份。

随后固定上述参数，用未参与选择的种子 2026 重新采集自动遮罩并生成视频。该种子的原生对照仍有串角，自动联合版本也消除了这一明显错误。两段视频的开头、中间和结尾都保留了角色区分，五个时间点的接触图和完整 MP4 一起提供。这里只能作定性判断；没有人工分割真值、身份分数或复杂运动的时间一致性基准。

所有实验均在用户指定的韩国 DSW、任务目录 /mnt/workspace/freefuse-h3-20261024 中执行，使用空闲 H200（主要为 GPU 4–7）。模型与 LoRA 文件完成大小和 SHA-256 核对。依赖装在专用环境，公共环境、其他任务数据和进程未修改。结果通过仅绑定回环地址的服务和 SSH 转发提供。

## 结论与边界

值得继续把 H3 作为 FreeFuse 的支持目标：本次区域 LoRA 和注意力约束确实能影响原生视频生成，自动区域也在两个种子上展示了角色隔离的改善。下一步重点是提高区域采集对参数、角色组合和运动的稳定性，再接入正式节点。本次仍是研究原型，尚未作为 Registry 发布版本。

未测试复杂运动、交叉遮挡、镜头切换、不同提示词、其他角色组合、参考帧生成或低精度/稀疏加速。音频没有质量评估；H3 是视频/音频联合模型，视频约束可能间接改变音频，不能声称音频不变。手部等原生生成瑕疵仍存在。两段视频不是大规模基准或普遍改善的统计证据。

## 固定设置与文件

| 项目 | 设置 |
|---|---|
| 原生模型 | minimax_h3_fl2va_bf16.safetensors（完整 BF16） |
| 文本编码器 | qwen3vl_32b_minimax_h3_bf16.safetensors |
| 原始输出 | 768 × 768，124 帧，24 fps |
| 采样 | res_multistep / simple，20 步，CFG 1 |
| LoRA | Starfire/Teen Titans、Albedo，各强度 1 |
| 主对照 | 两套 LoRA 全局叠加，关闭 FreeFuse |
| 自动遮罩 | 关闭 LoRA 采集；索引 3 / block 20 / 温度 100000 / 平滑窗口 3 |
| 注意力约束 | 正向 2，负向 4，作用于全部 50 个主 DiT 块 |
| 参数选择与复验 | 种子 42 选择参数，2026 固定参数复验；同一提示词 |
| ComfyUI | 7cebca52227c63b4180ceed448cbd7ad22903b6d |
| FreeFuse 基线 | 23310ae |
| 原生权重来源 | Comfy-Org/MiniMax-H3，e5eb578a89295337b8ff433a035929ce0279e0b6 |

- [种子42并排视频](pilot_comparison.mp4)、[种子2026并排视频](heldout_comparison.mp4)
- [开关对照中间帧](ablations.jpg)、[种子42时间线](pilot_timeline.jpg)、[种子2026时间线](heldout_timeline.jpg)
- [自动遮罩叠加视频](automatic_mask_overlay.mp4)（红：Starfire，绿：Albedo；底图是关闭 LoRA 的采集视频）
- 原始视频、潜变量及 run.json 位于 validation 下各独立运行目录。
- 模型/LoRA 清单、版本、提示词、token 位置、运行计时与视频哈希见 [provenance.json](provenance.json)。
- 计时包含采样入口内的模型装载，未计 VAE 解码；自动区域采集另有开销。本次并行运行，未做严格速度基准。

## 检查记录

H3 的 9 项检查全部通过（h3-tests-final.log）。原生小模型的采集无扰动、零偏置一致性、负 CFG 隔离、二维 packed LoRA 遮罩、Unicode/特殊 token 对齐、参考/音频片段布局以及偏置数值对照通过。偏置检查包含 FP32/BF16、64/128 维注意力头。完整模型的两套 LoRA 均匹配 200 个目标，全部主结果完成 20 步并解码出 124 帧。

现有图像测试在这个新环境中为 34 通过、1 失败。失败是 MOSS 的 Unicode token 位置测试：当前 ComfyUI/Transformers 组合返回不支持 offset_mapping 的慢 tokenizer。同一失败已在未修改的 23310ae 独立工作区复现，未把它当作 H3 改动引起的回归，也未在本次顺带修改 MOSS。

## 复现入口

这是原生 ComfyUI Python 实验入口，尚未注册用户节点。将 ComfyUI 与 FreeFuse 放在同一任务目录，模型放在 models/{diffusion_models,text_encoders,vae,loras}，把源码中的公开 manifest 复制到任务的 reference/。使用任务专用环境，安装 ComfyUI requirements、av、pytest，并提供 run-python 启动器指向该环境。

在 FreeFuse 根目录执行，GPU 编号按实际空闲资源设置：

```bash
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_native.py --out native_global
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_collection.py --out collection_seed42
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/mask_sweep.py
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_routing.py --mask s3_b20_t100000 --mode combined --out auto_combined

PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_native.py --seed 2026 --out holdout_native_global
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_collection.py --seed 2026 --out collection_seed2026
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/build_fixed_mask.py --seed 2026
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_routing.py --seed 2026 --mask seed2026_s3_b20_t100000 --mode combined --out holdout_auto_combined

PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_native.py --names starfire --out reference_starfire
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_native.py --names albedo --out reference_albedo
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_routing.py --mask s3_b20_t100000 --mode route --out auto_route
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_routing.py --mask s3_b20_t100000 --mode bias --out auto_bias
PYTHONPATH=$PWD ../run-python analysis/minimax_h3/run_routing.py --mask manual --mode combined --out manual_combined
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../run-python -m pytest -q analysis/minimax_h3/test_support.py
../run-python analysis/minimax_h3/render_review.py
../run-python analysis/minimax_h3/review_server.py
```

官方参考：[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)、[ComfyUI 原生教程](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)、[官方 Comfy 权重](https://huggingface.co/Comfy-Org/MiniMax-H3)。LoRA 版本与作者见公开清单。
