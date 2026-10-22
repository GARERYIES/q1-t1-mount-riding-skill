# Runtime data

以下文件和 benchmark 结果相关，但本次 public 发布不包含：

```text
reference_B.npz
initial_state_B_t3.npz
control_contract_v1.json
handload_cross17_trajectory.npz
handload_cross17_metrics.jsonl.gz
```

原因：

- reference motion 和初始状态来自受限仿真素材；
- native trajectory 和接触力记录包含原厂模型相关数据；
- 当前没有明确的第三方再分发许可。

授权复现时，需要把上述文件放在 `data/`，并保持
`configs/demo_handload_cross17.json` 中的 SHA-256 合同一致。

公开仓库仍保留 `docs/light_dr_ppo16/` 中的聚合指标和训练结果，用于说明负结果与评价口径。
