# Demo 说明

## 文件

- `Q1T1_mount_demo_handload_cross17_17s.mp4`：完整 17 秒真实 native 状态回放；
- `demo_preview.gif`：面向 README 的 9 秒轻量预览；
- `demo_poster.png`：跨后前倾支撑帧；
- `light_dr_ppo16_rejected_nominal.mp4`：未通过晋升评测的 Mild-DR PPO16 候选。

视频来自真实 MuJoCo native 执行后保存的 `qpos/qvel`。视频生成只做运动学和渲染，不重新积分、不切换 policy、不 reset、不固定 root、不添加外力。

## 主 Demo 指标

| 指标 | 实测 |
|---|---:|
| 起点 | 保存的局部扶撑跨腿初态 |
| 跨越完成 | 约 4.90 s |
| 跨后连续未摔倒保持 | 约 12.10 s |
| 右手合法承重 | 末 1 秒约 14.2 N |
| seat | 末 1 秒约 52.9 N |
| 左脚 / 右脚 | 末 1 秒约 13.3 / 36.4 N |
| 最大关节越限 | 约 9.25 mrad |
| T1 最大倾角 | 约 5.33° |
| 正式合法 hold | 0 s |

动作仍然存在左脚和右脚借地、左肘壳、右髋壳以及踝/腿壳等未核准接触。该视频用于展示阶段性仿真能力，不构成普通站姿自主上马、合法坐稳或真机成功声明。

## 媒体权利

视频和 GIF 仅用于展示本项目研究结果。机器人名称、外观、商标以及相关模型权利归其权利人所有；详见仓库根目录 `THIRD_PARTY_NOTICES.md`。
