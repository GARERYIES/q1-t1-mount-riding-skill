# Model assets

原厂 Q1/T1 STL、mesh 和组合 MuJoCo XML 不随 public 仓库分发。

官方 Q1/T1 URDF、SDK 和说明文档：

https://dev.primebot.com/download

官方 URDF 不保证与本研究使用的组合场景完全一致。使用前需要重新核对：

- body 与 joint 坐标系；
- 质量、惯量和碰撞几何；
- actuator 顺序；
- 关节范围和 PD；
- Q1/T1 组合场景中的接触对和 support surface。

获得授权后，将组合场景放在 `model/scene.xml`，mesh 放在 `model/meshes/`。
