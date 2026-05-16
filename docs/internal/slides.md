# OffshoreWind-ControlRL 演讲幻灯片

---

## 标题页
**项目名称**：Robust and Safe PPO Control for a Simplified Floating Offshore Wind Platform
**演讲者**：__[姓名]__
**时间**：__[日期]__
**地点**：__[会议/工作坊]__

---

## 1. 项目背景 & 研究动机
- **现实需求**：浮式海上风机平台需在强风/波浪敲击下保持姿态。
- **技术挑战**：\
  - 实时控制 → 高度动态。
  - 传统类 PID/PD 经验法则对极端环境有局限。
- **创新点**：\
  - 结合 **Proximal Policy Optimization (PPO)** 与 **安全过滤**。
  - 在可控可验证的简化动力学模型上实现全闭环 RL 方案。

---

## 2. 研究目标 & 评估指标
| 目标 | 说明 |
|------|------|
| **可行性** | 在课设规模下实现完整的 RL 控制链路。 |
| **稳健性** | 同时评估四种风波情景（正常、强、可变、OOD）。 |
| **安全性** | 坚持绝对安全阈值 |α|≤0.3rad. |
| **资源效率** | 控制能源（action²）的占比。 |
| **稳定性** | 随机种子 vs 训练步数对性能的影响。 |

---

## 3. 环境 & MDP 公式化
```
State : s = [θ, θ̇, w, q]            (4‑D)
Action: a ∈ [-1, 1] → F = a * 0.5 N·m
Dynamics (Euler) :
  θ_{t+1} = θ_t + θ̇_t * dt
  θ̇_{t+1} = θ̇_t + dt/m * (F + w + q - cθ̇_t - kθ_t)
Reward : r = -θ² – 0.5 θ̇² – 0.1 F² + safety_penalty
Done : |θ|>0.3 rad  OR  episode_len≥1000
```
- **安全惩罚**：若 |θ| > 0.9·0.3 ,  额外 -10.0
- **控制量**：s = action (连续). |

---

## 4. 基线控制器
| 控制器 | 说明 |
|------|------|
| **NoControl** | 始终输出 0。
| **PD** | `a = clip(-Kp·θ - Kd·θ̇, -1, 1)`  (Kp=5.0, Kd=2.0)。 |
| **PPO** | 使用 Stable‑Baselines3 的 MlpPolicy， 默认 lr=3e‑4。
| **PPO+Safety** | 在 PPO 之后加入 1‑step 预测 safety filter。
| **PPO‑Randomized** | 训练时随机化环境参数(damping, stiffness, wind_noise)。 |

---

## 5. 训练方案
| 参数 | 值 |
|------|-----|
| 训练步数 | 500 k (从 1k → 1 m) |
| 随机种子 | 0、1、2 |
| 终值 | ‑1,000 采样 |
| 优化器 | Adam (β₁=0.9, β₂=0.999) |
| 记忆 | 1M 条 | 
| 学习率衰减 |  

+ **域随机化**：每次 episode reset 时重新 draw 参数。
+ **Safety Filter**：
```
pred_theta = θ + θ̇*dt + 0.5*damping*dt**2
if |pred_theta|>0.3 :  action = 0
elif |pred_theta|>0.24 : action = action/2
```

---

## 6. 评估与结果
- **情形**：normal, strong, variable, out‑of‑distribution.
- **评估周期**：20 次自举测试。
- **主要指标**：return、failure_rate、control_energy、mean_abs_theta。

| Scenario | Winner | avg_return | fail% | energy | mean_abs_theta |
|----------|--------|------------|-------|--------|----------------|
| normal   | **PD**   | -2.59 | 0.0 | 25.1 | 0.020 |
| strong   | **PD**   | -10.10 | 0.0 | 98.8 | 0.040 |
| variable | **PD**   | -10.90 | 0.0 | 154.3 | 0.064 |
| OOD      | **PPO+Safety** | -30.19 | 40.0 | - | 0.114 |

- **关键点**：
  1. PD 在所有平稳情形几乎零失败，且能耗最低。
  2. PPO 在非分布外情况有 18 % 的返回提升，但失效率高达 40 %。
  3. 域随机化对凯文“性能”提升有限，且不解决 seed‑0 失败。

---

## 7. 全局性能图示
（请查看 `results/figures/`）
- 学习曲线 (log reward)
- 控制能耗 box‑plot
- 失败率柱状图
- 位置/动作时间序列

---

## 8. 结论 & 贡献
| 贡献 | 说明 |
|------|------|
| **可复现性** | 代码、数据、模型均已公开。 |
| **多种子评估** | 关闭了单 seed 曲杆。 |
| **安全过滤器** | 让 RL 在 OOD 情境下保留竞争力。 |
| **简化模型** | 为教育和快速原型提供可靠基准。 |
| **开源实现** | 完整的 GitHub 文档与示例脚本。 |

---

## 9. 局限 & 下一步
- **模型简化**：无真实 hydrodynamics；可接 3‑D 结构与 FEA。
- **策略安全**：采用更精细的预测安全约束（MPC、VaR）。
- **分布式风机**：扩展至多机群控制与协同。 |
- **多目标**：包括功率输出、能耗、可靠性综合优化。

---

## 10. Q&A
> 有哪些观点需要进一步讨论？ |
> 对安全过滤的更严格策略？ |
> 进一步测试极端风速（>1.2·σ）？ |

---

## 11. 致谢
- DeepSense、RL‑PSF、FloatingFarmYaw 受启发。 |
- GitHub 仓库: https://github.com/your‑org/OffshoreWind_ControlRL |
- 所有测试环境与评估脚本均为开源。 |

---

## 12. 附录（可选）
- 代码示例：`src/training/train_ppo.py` 运行方式。
- 数据生成脚本：`src/envs/floating_platform_env.py`。 |
- 贡献者名单与工作分工。 |

---

**结束语**：本项目提供了从简化 MDP → 训练 PPO → 评估 → 可视化 的全流程示例。对教学与快速原型极具参考价值。 |