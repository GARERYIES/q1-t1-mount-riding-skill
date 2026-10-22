# 实验摘要

## 当前 Demo 选择

选用 B64 right2x handload checkpoint 的真实 native 连续轨迹：

- 起点：保存的局部扶撑跨腿初态；
- 约 4.90 s 完成右腿后髋近直腿跨越；
- 跨后连续未摔倒保持约 12.10 s；
- 最后 1 s 右手合法掌部/前臂平均竖直载荷约 14.2 N；
- 最后 1 s seat 平均约 52.9 N；
- 最后 1 s 左脚/右脚约 13.3/36.4 N；
- 最大关节越限约 9.25 mrad；
- T1 最大倾角约 5.33 deg；
- 无 hard guard，但存在持续未核准左肘、右髋和踝/腿壳接触。

正式“合法坐稳 3 秒”仍未完成，正式 hold 为 `0 s`。Demo 使用临时脚部支撑和前倾姿态，不声称普通站姿自主上马。

## 为什么未使用最新 439 PPO64

439 观测候选在严格 RSI 5 秒 native 筛选中没有形成合法 hold；右脚仍约 36.9 N 借地，seat 和臂部载荷的转移没有满足目标。候选未晋升，也不用于 Demo。

## Light-DR native screen v1

使用打包后的闭环 skill 在 nominal 和 3 个 light-DR seed 上各运行 17 秒：

- 4/4 均完成跨越；
- 4/4 均无 hard guard；
- 跨腿完成时间约 4.76-5.04 秒；
- 随机化后 seat/手部支撑明显退化，未核准 T1 接触上升；
- 因此当前 handload actor 不能直接晋升为鲁棒模型，也没有依据将其宣称为 DR-trained。

完整机器可读结果见 `docs/dr_light_screen_v1.json`。下一轮应以有界 PPO
微调改善这些 light-DR 失败，而不是扩大随机化范围。

## Bounded light-DR PPO v1

从 protected handload checkpoint 完整 warm start，完成 16/16 accepted
updates 和 65,536 environment steps。候选同样在 nominal 与三个 DR seed
上执行 17 秒 native 复测。跨腿保持，但 nominal seat 从约 51.1 N 降至
31.0 N，两个 DR seed 的 seat 降至 0 N，未核准 T1 接触仍约 134-197 N。

因此该候选被拒绝，不替换 Demo 模型，也不扩大训练。详见
`training/RESULTS_light_dr_ppo16.md`。

## 可复现证据

- `assets/Q1T1_mount_demo_handload_cross17_17s.mp4`
- `assets/demo_preview.gif`
- `assets/DEMO.md`
- `training/RESULTS_light_dr_ppo16.md`
- `docs/light_dr_ppo16/`

原厂 mesh、基础 Policy、reference motion、原始 native 轨迹和 checkpoint
缺少明确的再分发许可，因此不随 public 仓库发布。
