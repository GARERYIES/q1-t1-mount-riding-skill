# Checkpoints

受保护的 actor 和训练 checkpoint 不随 public 仓库发布，因为它们建立在原厂仿真资产和基础 Policy 之上，当前没有明确的再分发许可。

公开记录中的模型版本：

| 模型 | SHA-256 | 状态 |
|---|---|---|
| Demo handload actor | `9a671c257b765c37237c65736148fe2abb468624828401893559a47c56afb0b0` | Demo 采用 |
| Handload checkpoint | `c6ca93721d0b36315830ff8c7cada7801c8e9bb26f0f735decdcfd39d3de2991` | 保护父模型 |
| Light-DR PPO16 actor | `47f2e8a931341c88b6c6ae3f0c62d12d5450567d931be397c1c70c37fd6968d0` | 拒绝晋升 |

获得授权后，将 Demo actor 放到 `checkpoints/handload_actor.ts`。
