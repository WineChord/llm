# 空间智能与三维表示

二维图像是三维场景经相机投影后的结果。单张图中的像素位置并不直接给出真实距离、尺度、遮挡后结构或可通行空间。空间智能因此需要把视觉语义与几何坐标连接起来：

$$
\text{pixels}
\rightarrow
\text{rays / depth / points}
\rightarrow
\text{objects and scene}
\rightarrow
\text{language and action}.
$$

它既不是普通 VQA 的自然外推，也不等于生成一段具有立体感的视频。一个系统要在空间中可靠行动，必须维护坐标、相机、尺度、时间与不确定性。

## 相机模型是第一份契约

针孔相机把相机坐标中的点 $P_c=(X,Y,Z)$ 投影到像素：

$$
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
\sim
K
\begin{bmatrix}
X/Z\\Y/Z\\1
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
$$

若已知深度 $Z$，可反投影：

$$
X=(u-c_x)Z/f_x,\qquad
Y=(v-c_y)Z/f_y.
$$

畸变、rolling shutter、resize 和 crop 都会改变像素到光线的映射。模型即使预测了正确 box，若使用过期或错误内参，三维位置仍会偏移。

```python
def backproject(u, v, depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    if depth <= 0 or min(fx, fy) <= 0:
        raise ValueError("depth and focal lengths must be positive")
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return x, y, depth
point = backproject(420, 290, 2., (500., 500., 320., 240.))
assert point == (.4, .2, 2.)
```

这段代码只给出相机坐标。要进入世界或机器人坐标，还需外参 $T_{wc}\in SE(3)$；单位、轴方向和矩阵左右乘约定必须固定。

## 深度从哪里来

### 双目与多视角

已知基线 $b$ 与焦距 $f$，理想校正双目中的深度近似

$$
Z=\frac{fb}{d},
$$

其中 $d$ 是视差。视差很小时，深度误差会被放大；无纹理、反光与遮挡区域也难以匹配。

Structure-from-Motion 与 multi-view stereo 从多张图的对应点共同估计相机位姿与结构。它们利用视角变化恢复几何，但尺度可能只有相似变换意义，动态物体又会破坏静态场景假设。

### 主动传感

LiDAR、结构光和 ToF 直接测量距离，提供更稳定尺度，却有稀疏、反射、量程和成本限制。深度图与 RGB 往往来自不同光心，需要标定与时间同步。

### 单目学习

单目深度模型从数据先验预测深度或相对深度。它能在一张图上工作，但真实尺度、透明物体和领域外相机仍可能不可靠。评测必须区分 metric depth、relative depth 与 ordinal relation。

## 点云、体素与隐式场

三维场景有多种计算表示：

| 表示 | 优点 | 主要代价 |
| --- | --- | --- |
| 点云 | 保留测量、稀疏直接 | 无天然邻域和表面 |
| voxel | 规则网格、卷积友好 | 分辨率立方增长 |
| mesh | 显式表面与拓扑 | 重建和更新复杂 |
| implicit field | 连续查询、紧凑 | 渲染/优化成本 |
| Gaussian primitives | 快速可微渲染 | 几何与编辑语义需额外约束 |

[PointNet](https://arxiv.org/abs/1612.00593) 用对称聚合处理无序点集，[PointNet++](https://arxiv.org/abs/1706.02413)进一步建立局部层级。点的排列不应改变结果，但采样密度、坐标尺度和局部邻域仍决定表示。

[NeRF](https://arxiv.org/abs/2003.08934) 用神经场表示位置与方向到密度、颜色的映射，并沿光线体渲染：

$$
C(r)
=
\int_{t_n}^{t_f}
T(t)\sigma(r(t))c(r(t),d)\,dt,
$$

$$
T(t)
=
\exp\left(
-\int_{t_n}^{t}\sigma(r(s))\,ds
\right).
$$

它擅长从已知视角图像重建可渲染场景，但经典逐场景优化、静态假设和视角外推限制了实时交互。

[3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) 用带位置、协方差、颜色与透明度的显式 Gaussian primitives，实现高质量实时渲染。渲染速度不等于物理可交互性；碰撞、材质、对象身份和可编辑拓扑仍需额外表示。

## 从重建到语义场景

几何回答“在哪里”，语义回答“是什么、能做什么”。两者结合可以形成：

- 3D detection 与 instance segmentation；
- open-vocabulary 3D grounding；
- 语言查询的对象地图；
- affordance 与可抓取区域；
- 跨视角身份跟踪；
- 面向导航和操作的 scene graph。

二维基础模型可以把语义特征投影到 3D，再通过多视角一致性融合。但若 2D segmentation 错误或位姿漂移，错误会在地图中累积。开放词表相似度也不能替代精确几何和可操作性验证。

语言描述“杯子在桌子左边”时，还要明确参考系：

- 图像坐标的左；
- 相机朝向的左；
- 机器人 base frame 的左；
- 人或对象自身坐标的左。

自然语言中的视角省略是空间推理的重要歧义源。

## 多视角与对象恒常性

同一对象在不同视角下外观改变，短暂离开视野后仍应保持身份。系统通常组合：

- 几何投影与可见性；
- 外观/语义 embedding；
- 运动模型；
- 时间和置信更新；
- object-level memory。

只把多张图拼入 VLM 上下文，模型可能比较外观，却不一定知道相机之间的变换。加入 pose token 或把特征投影到共同坐标可以提供几何约束，但标定误差必须显式建模。

一个长期场景记忆至少保存

$$
\mathcal M_t
=
\{(o_i,\ \hat T_i,\ \Sigma_i,\ \text{attributes}_i,\ \text{last seen}_i)\},
$$

其中 $\Sigma_i$ 表示位置或状态不确定性。环境变化后，旧对象不能因为曾被观察就永久当作当前事实。

## 生成式三维与可交互世界

文本或图像条件的 3D 生成可以输出 mesh、NeRF、Gaussian、multi-view image 或可导航场景。评价时应拆开：

- 单视图视觉质量；
- 多视角一致性；
- 几何完整性与尺度；
- 可编辑对象和材质；
- 碰撞与物理属性；
- 相机/动作控制响应；
- 长时间状态持久性。

一组跨视角连贯图片可以营造三维感，却没有显式可查询几何；一个可渲染 NeRF 也可能没有对象级语义和动力学。生成式世界与决策世界模型的边界见[表示预测与生成式世界](../../world-models/predictive-generative-worlds.md)。

## 空间表示怎样进入 VLA

机器人策略可以直接从图像端到端输出动作，也可以显式构建深度、点云或对象状态。两条路线的差异不是“是否智能”，而是状态接口：

- 端到端策略减少手工模块，却难以定位几何错误；
- 显式状态便于约束和规划，却会累积 perception/association 误差；
- 混合系统可让 VLM 做语义目标，几何 planner 和低层 controller 保证可执行性。

无论哪条路线，动作必须注明 world/base/end-effector/camera frame，详见[状态、动作与策略](../../embodied/state-action-policies.md)。

## 评测矩阵

| 能力 | 代表测量 | 必要切片 |
| --- | --- | --- |
| 深度 | AbsRel、RMSE、尺度误差 | 室内/室外、透明、远距 |
| 位姿 | ATE/RPE | 速度、回环、动态物体 |
| 重建 | geometry/render quality | 新视角、遮挡、薄结构 |
| 语义 | 3D mAP/IoU、grounding | 长尾对象、开放词汇 |
| 记忆 | identity/state consistency | 离开视野、环境变化 |
| 交互 | navigation/manipulation success | 闭环恢复与校准漂移 |
| 系统 | map update/latency/memory | 多相机、长时运行 |

空间问答还应要求可视化坐标或三维证据。只比较自然语言答案，很难区分真实几何推理与场景常识猜测。

## 失效模式

- resize/crop 后内参未更新；
- depth 单位或坐标轴混淆；
- 相机、IMU 与机器人时间不同步；
- 相机运动被误认为对象运动；
- 单目先验给出合理但错误的绝对尺度；
- 地图融合把同一对象重复建图；
- object memory 在环境变化后过期；
- 新视角渲染逼真，但几何不能支持碰撞和规划；
- 语言方位未绑定参考系；
- benchmark 的合成相机和真实镜头分布差异被忽略。

空间智能的可靠性来自从像素到坐标、从坐标到记忆、从记忆到行动的整条可审计链，而不是某一个 3D 指标。

投影、坐标往返与视觉 token 的组合练习见[多模态手撕实现](../../practice/multimodal.md)。

## Reference {#reference}

- [Hartley and Zisserman, Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Qi et al., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [Qi et al., PointNet++: Deep Hierarchical Feature Learning on Point Sets](https://arxiv.org/abs/1706.02413)
- [Mildenhall et al., NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)
- [Ranftl et al., Towards Robust Monocular Depth Estimation](https://arxiv.org/abs/1907.01341)
- [Peng et al., OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/abs/2211.15654)
- [Hong et al., 3D-LLM: Injecting the 3D World into Large Language Models](https://arxiv.org/abs/2307.12981)
- [Majumdar et al., OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://arxiv.org/abs/2404.05080)
