# Training Notes

本项目采用 reference-conditioned PPO + RSI，从已有跨腿/恢复 checkpoint 继续训练，不从随机网络开始。

## 已完成的轻量实验

- 128 个 RSI worlds；
- 32 steps/update；
- 16/16 accepted updates；
- 65,536 environment steps；
- 约 493 秒；
- 无 safety stop；
- 接触摩擦在标称 ±10% 内采样；
- PD 增益在标称 ±5% 内采样；
- 初始 root 位置约 ±1 cm；
- 初始关节角约 ±2°。

该候选在 nominal 和 3 个 light-DR seed 的 17 秒 native 评测中均保留了跨越，但 seat 承重下降、未核准接触仍高。候选被拒绝，没有替换 Demo 模型。完整结果见 `RESULTS_light_dr_ppo16.md`。

## 质量随机化限制

共享 MuJoCo Warp 模型没有 per-world mass 字段，因此质量随机化被明确标记为 deferred。项目没有通过隐藏外力伪造质量变化。

## 为什么不发布训练入口和完整数据

完整训练需要原厂模型、基础 Policy、reference motion、reset-state bank 和训练 source bank。这些文件缺少明确的第三方再分发许可，因此 public 仓库只发布训练协议、评价结果和 native 筛选工具，不打包受限资产。

获得授权后，可按照 `configs/demo_handload_cross17.json` 的路径合同放入资产，并使用原研究环境中的 PPO+RSI trainer。public 仓库不提供受限资产的下载镜像。

## 晋升规则

候选必须同时满足：

1. 不损伤跨腿完成时间和近直腿轨迹；
2. 不增加危险、自接触或未核准 T1 接触；
3. 合法手/seat 载荷不下降；
4. 脚地反力下降；
5. 关节和 T1 保持稳定；
6. 在 nominal 与多个轻量扰动 seed 上重复验证。

reward 增长、姿态外观或单次接触不能单独构成晋升依据。
