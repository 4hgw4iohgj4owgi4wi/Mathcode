# 赛题 2 问题一：拥堵瓶颈识别模型公式整理

> 本文档依据 `Framework.md` 中“有限容量双边排队网络—道路回溢—因果瓶颈识别”的统一框架整理，并以 `q1_bottleneck_analysis.py` 的实际计算逻辑为准。所有公式均采用可直接用于 Markdown 或论文的 LaTeX 格式。

## 1. 模型结构

将机场陆侧交通中心抽象为由入口道路、车辆入口、车辆缓冲区、上客区和离场道路组成的有限容量网络：

$$
G=(V,E).
$$

交通方式集合为

$$
\mathcal M
=
\{\text{taxi},\text{ride-hailing},\text{private-car},\text{bus}\},
$$

分别表示出租车、网约车、私家车和机场巴士。离散时间集合为

$$
\mathcal T=\{0,1,\ldots,T-1\},
$$

代码中取时间步长 $\Delta t=1\ \mathrm{min}$，仿真时长为 $T=240\ \mathrm{min}$。

系统的主要流动过程为

$$
\text{旅客到达}
\longrightarrow
\text{分方式候车}
\longrightarrow
\text{车辆进入缓冲区}
\longrightarrow
\text{车辆完成上客}
\longrightarrow
\text{离场道路}.
$$

## 2. 主要符号

| 符号 | 含义 |
|---|---|
| $m$ | 交通方式，$m\in\mathcal M$ |
| $t$ | 离散时间，$t\in\mathcal T$ |
| $\lambda_t$ | 时刻 $t$ 的总旅客到达率 |
| $s_m$ | 交通方式 $m$ 的旅客分担率 |
| $a_m$ | 交通方式 $m$ 的平均载客量 |
| $d_{m,t}$ | 时刻 $t$ 选择方式 $m$ 的旅客到达量 |
| $u_{m,t}$ | 时刻 $t$ 到达交通中心外部的车辆数 |
| $\tau_m$ | 方式 $m$ 的车辆供给响应延迟 |
| $\eta_m$ | 方式 $m$ 的车辆供给响应系数 |
| $P_{m,t}$ | 方式 $m$ 的候车旅客队列 |
| $E_{m,t}$ | 方式 $m$ 在入口外部等待的车辆队列 |
| $H_{m,t}$ | 方式 $m$ 在内部缓冲区中的车辆数 |
| $N_t$ | 离场道路中的车辆占用量 |
| $K_m^H$ | 方式 $m$ 的内部缓冲区容量 |
| $\mu_m^G$ | 方式 $m$ 的入口服务能力 |
| $\mu_m^C$ | 方式 $m$ 的上客区服务能力 |
| $C^{\mathrm{in}}$ | 公共入口道路通行能力 |
| $C^{\mathrm{out}}$ | 离场道路通行能力 |
| $K^{\mathrm{out}}$ | 离场道路存储容量 |
| $g_{m,t}$ | 时刻 $t$ 实际进入内部缓冲区的车辆数 |
| $y_{m,t}$ | 时刻 $t$ 完成上客并进入离场道路的车辆数 |
| $x_{m,t}$ | 时刻 $t$ 完成运输匹配的旅客数 |
| $B_t^{\mathrm{in}}$ | 入口道路容量约束造成的阻塞量 |
| $B_t^{\mathrm{out}}$ | 离场存储容量约束造成的阻塞量 |
| $J$ | 系统广义延误成本 |
| $S_i$ | 设施或节点 $i$ 的表面拥堵指数 |
| $\mathcal E_i$ | 设施或资源 $i$ 的因果瓶颈弹性 |

## 3. 旅客需求与车辆供给

### 3.1 多高斯峰旅客到达率

代码使用多个高斯峰叠加描述航班集中到达引起的客流波动：

$$
\lambda_t
=
b+\sum_{k=1}^{K}
h_k
\exp\left[
-\frac{1}{2}
\left(
\frac{t-c_k}{w_k}
\right)^2
\right],
$$

其中：

- $b$ 为基础客流强度；
- $c_k$ 为第 $k$ 个客流高峰的中心时刻；
- $w_k$ 为高峰宽度；
- $h_k$ 为高峰增量。

### 3.2 分交通方式旅客需求

方式 $m$ 的期望旅客到达率为

$$
\lambda_{m,t}^{P}=s_m\lambda_t,
\qquad
\sum_{m\in\mathcal M}s_m=1.
$$

实际旅客到达量服从泊松分布：

$$
d_{m,t}
\sim
\operatorname{Poisson}\left(\lambda_{m,t}^{P}\right).
$$

### 3.3 延迟响应的车辆供给

定义车辆供给所响应的需求时刻

$$
\ell_m(t)
=
\begin{cases}
0, & t<\tau_m,\\
t-\tau_m, & t\geq \tau_m.
\end{cases}
$$

方式 $m$ 的期望车辆到达率为

$$
\lambda_{m,t}^{V}
=
\eta_m
\frac{\lambda_{m,\ell_m(t)}^{P}}{a_m}.
$$

实际车辆供给量为

$$
u_{m,t}
\sim
\operatorname{Poisson}
\left(
\max\{\lambda_{m,t}^{V},0.05\}
\right).
$$

该式反映了车辆供给对旅客需求的滞后响应。$\eta_m<1$ 表示供给不足，$\eta_m>1$ 表示车辆供给相对充足。

## 4. 有限容量双边排队网络

### 4.1 到达更新

每个时间步开始时，先将新到达的旅客和车辆加入队列：

$$
\bar P_{m,t}=P_{m,t}+d_{m,t},
$$

$$
\bar E_{m,t}=E_{m,t}+u_{m,t}.
$$

其中 $\bar P_{m,t}$ 和 $\bar E_{m,t}$ 分别表示加入本时段新需求后的旅客队列和外部车辆队列。

### 4.2 离场道路车辆释放

离场道路首先按其通行能力释放已有车辆：

$$
q_t^{\mathrm{out}}
=
\min\{N_t,C^{\mathrm{out}}\},
$$

$$
\bar N_t
=
N_t-q_t^{\mathrm{out}}
=
\max\{N_t-C^{\mathrm{out}},0\}.
$$

$\bar N_t$ 为本时段上客车辆进入离场道路之前的道路占用量。

### 4.3 车辆入口请求

方式 $m$ 的内部缓冲区剩余容量为

$$
R_{m,t}^{H}
=
\max\{K_m^H-H_{m,t},0\}.
$$

考虑外部车辆队列与缓冲区空间后，车辆的原始入口需求为

$$
\widehat g_{m,t}
=
\min\{\bar E_{m,t},R_{m,t}^{H}\}.
$$

进一步考虑方式专用入口的服务能力：

$$
g_{m,t}^{\mathrm{req}}
=
\min\{\widehat g_{m,t},\mu_m^G\}.
$$

### 4.4 公共入口道路容量分配

所有方式共用入口道路。定义入口道路比例分配系数：

$$
\alpha_t^{\mathrm{in}}
=
\begin{cases}
\displaystyle
\min\left\{
1,
\frac{C^{\mathrm{in}}}
{\sum_{m\in\mathcal M}g_{m,t}^{\mathrm{req}}}
\right\},
&
\sum_m g_{m,t}^{\mathrm{req}}>0,
\\[3mm]
1,
&
\sum_m g_{m,t}^{\mathrm{req}}=0.
\end{cases}
$$

方式 $m$ 实际获准进入缓冲区的车辆数为

$$
g_{m,t}
=
\alpha_t^{\mathrm{in}}g_{m,t}^{\mathrm{req}}.
$$

入口更新后的车辆状态为

$$
E_{m,t+1}
=
\bar E_{m,t}-g_{m,t},
$$

$$
\bar H_{m,t}
=
H_{m,t}+g_{m,t}.
$$

该比例分配机制保证容量不足时各交通方式按照请求量同比缩减。

### 4.5 上客服务请求

由缓冲区车辆数量和候车旅客数量共同决定可执行的上客车辆数：

$$
\widehat y_{m,t}
=
\min
\left\{
\bar H_{m,t},
\frac{\bar P_{m,t}}{a_m}
\right\}.
$$

考虑上客区服务能力后：

$$
y_{m,t}^{\mathrm{req}}
=
\min\{\widehat y_{m,t},\mu_m^C\}.
$$

这同时满足车辆可用约束、旅客需求约束和上客区服务能力约束。

### 4.6 离场道路接收约束

离场道路的剩余存储空间为

$$
R_t^{\mathrm{out}}
=
\max\{K^{\mathrm{out}}-\bar N_t,0\}.
$$

定义离场道路比例接收系数：

$$
\alpha_t^{\mathrm{out}}
=
\begin{cases}
\displaystyle
\min\left\{
1,
\frac{R_t^{\mathrm{out}}}
{\sum_{m\in\mathcal M}y_{m,t}^{\mathrm{req}}}
\right\},
&
\sum_m y_{m,t}^{\mathrm{req}}>0,
\\[3mm]
1,
&
\sum_m y_{m,t}^{\mathrm{req}}=0.
\end{cases}
$$

实际完成上客并进入离场道路的车辆数为

$$
y_{m,t}
=
\alpha_t^{\mathrm{out}}y_{m,t}^{\mathrm{req}}.
$$

完成运输匹配的旅客数为

$$
x_{m,t}
=
\min\{\bar P_{m,t},a_my_{m,t}\}.
$$

因此，代码实现了框架中的载客能力约束：

$$
x_{m,t}\leq a_my_{m,t}.
$$

### 4.7 状态转移方程

旅客队列状态更新为

$$
P_{m,t+1}
=
\bar P_{m,t}-x_{m,t}
=
P_{m,t}+d_{m,t}-x_{m,t}.
$$

内部车辆缓冲区状态更新为

$$
H_{m,t+1}
=
\bar H_{m,t}-y_{m,t}
=
H_{m,t}+g_{m,t}-y_{m,t}.
$$

入口外部车辆队列更新为

$$
E_{m,t+1}
=
E_{m,t}+u_{m,t}-g_{m,t}.
$$

离场道路占用量更新为

$$
N_{t+1}
=
\bar N_t+\sum_{m\in\mathcal M}y_{m,t},
$$

等价地，

$$
N_{t+1}
=
N_t-q_t^{\mathrm{out}}
+\sum_{m\in\mathcal M}y_{m,t}.
$$

当 $R_t^{\mathrm{out}}$ 不足时，$y_{m,t}$ 被同比压缩，上客区车辆无法释放，进而占满内部缓冲区并阻止入口车辆进入，由此形成

$$
\text{离场道路拥堵}
\rightarrow
\text{上客区受阻}
\rightarrow
\text{内部缓冲区满载}
\rightarrow
\text{入口外部排队}.
$$

## 5. 阻塞量与容量负荷率

### 5.1 共享道路阻塞量

入口道路容量不足产生的阻塞量为

$$
B_t^{\mathrm{in}}
=
\sum_{m\in\mathcal M}
\left(
g_{m,t}^{\mathrm{req}}-g_{m,t}
\right).
$$

离场道路存储空间不足产生的阻塞量为

$$
B_t^{\mathrm{out}}
=
\sum_{m\in\mathcal M}
\left(
y_{m,t}^{\mathrm{req}}-y_{m,t}
\right).
$$

系统总阻塞量为

$$
B_t=B_t^{\mathrm{in}}+B_t^{\mathrm{out}}.
$$

### 5.2 方式专用设施阻塞量

方式 $m$ 的入口服务能力阻塞量为

$$
B_{m,t}^{G}
=
\max\{\widehat g_{m,t}-g_{m,t}^{\mathrm{req}},0\}.
$$

方式 $m$ 的上客区服务能力阻塞量为

$$
B_{m,t}^{C}
=
\max\{\widehat y_{m,t}-y_{m,t}^{\mathrm{req}},0\}.
$$

### 5.3 节点容量负荷率

公共入口道路负荷率为

$$
\rho_t^{\mathrm{in}}
=
\frac{\sum_m g_{m,t}^{\mathrm{req}}}
{C^{\mathrm{in}}}.
$$

离场道路负荷率为

$$
\rho_t^{\mathrm{out}}
=
\frac{\sum_m y_{m,t}^{\mathrm{req}}}
{C^{\mathrm{out}}}.
$$

方式 $m$ 的入口负荷率为

$$
\rho_{m,t}^{G}
=
\frac{\widehat g_{m,t}}{\mu_m^G}.
$$

方式 $m$ 的上客区负荷率为

$$
\rho_{m,t}^{C}
=
\frac{\widehat y_{m,t}}{\mu_m^C}.
$$

当 $\rho>1$ 时，表示对应设施的服务请求已超过标称服务能力。

## 6. 基础诊断指标

对任一诊断节点 $i$，其平均负荷率为

$$
\bar\rho_i
=
\frac{1}{T}\sum_{t=0}^{T-1}\rho_{i,t}.
$$

其 95% 分位负荷率为

$$
\rho_i^{95}
=
Q_{0.95}
\left(
\rho_{i,0},\rho_{i,1},\ldots,\rho_{i,T-1}
\right),
$$

其中 $Q_{0.95}(\cdot)$ 表示样本的 95% 分位数。

最大队列长度为

$$
Q_i^{\max}
=
\max_{t\in\mathcal T}Q_{i,t}.
$$

代码中各节点对应的队列定义为

$$
Q_{i,t}
=
\begin{cases}
\displaystyle\sum_m E_{m,t},
& i=\text{入口道路},
\\[2mm]
N_t,
& i=\text{离场道路},
\\
E_{m,t},
& i=\text{方式 }m\text{ 的入口},
\\
P_{m,t},
& i=\text{方式 }m\text{ 的上客区}.
\end{cases}
$$

定义回溢指示变量

$$
I_{i,t}^{\mathrm{spill}}
=
\begin{cases}
1, & B_{i,t}>10^{-6},\\
0, & B_{i,t}\leq 10^{-6}.
\end{cases}
$$

节点回溢概率为

$$
P_i^{\mathrm{spill}}
=
\frac{1}{T}
\sum_{t=0}^{T-1}
I_{i,t}^{\mathrm{spill}}.
$$

离场道路存储占用率的最大值为

$$
R_{\mathrm{storage}}^{\max}
=
\frac{\max_t N_t}{K^{\mathrm{out}}}.
$$

## 7. 表面拥堵指数

对序列 $z_i$ 采用极差归一化：

$$
\widetilde z_i
=
\frac{z_i-z_{\min}}
{z_{\max}-z_{\min}}.
$$

若 $z_{\max}=z_{\min}$，则令所有 $\widetilde z_i=0$。

为避免极端负荷率支配综合指标，代码先对 95% 分位负荷率作截断：

$$
\rho_i^{95,*}
=
\min\{\rho_i^{95},4\}.
$$

节点 $i$ 的表面拥堵指数为

$$
S_i
=
0.4\widetilde{\rho_i^{95,*}}
+0.3\widetilde{Q_i^{\max}}
+0.3P_i^{\mathrm{spill}}.
$$

$S_i$ 越大，表示该节点的高负荷、长队列和回溢现象越明显，但不一定意味着该节点是拥堵根因。

## 8. 系统广义延误成本

定义系统在时刻 $t$ 的旅客排队总量和车辆排队总量：

$$
P_t
=
\sum_{m\in\mathcal M}P_{m,t},
$$

$$
V_t
=
\sum_{m\in\mathcal M}
\left(E_{m,t}+H_{m,t}\right),
$$

离场道路占用量直接记为 $N_t$。

根据 `Framework.md`，广义成本的一般形式为

$$
J
=
\sum_{t=0}^{T-1}
\left(
c_pP_t+c_vV_t+c_nN_t+c_bB_t
\right).
$$

代码采用的具体权重为

$$
c_p=1,\qquad
c_v=1.2,\qquad
c_n=2,\qquad
c_b=4.
$$

因此，实际计算公式为

$$
J
=
\sum_{t=0}^{T-1}
\left[
P_t
+1.2V_t
+2N_t
+4\left(B_t^{\mathrm{in}}+B_t^{\mathrm{out}}\right)
\right].
$$

该目标将旅客等待、车辆等待、道路占用和容量阻塞统一换算为无量纲的广义延误成本，用于比较不同资源扩容前后的系统变化。

## 9. 系统运行指标

仿真期内完成服务的旅客总数为

$$
X
=
\sum_{t=0}^{T-1}
\sum_{m\in\mathcal M}
x_{m,t}.
$$

旅客总需求为

$$
D
=
\sum_{t=0}^{T-1}
\sum_{m\in\mathcal M}
d_{m,t}.
$$

旅客服务率为

$$
R_{\mathrm{service}}
=
\frac{X}{D}.
$$

根据离散形式的 Little 定律，平均旅客等待时间估计为

$$
\bar W_P
=
\frac{
\sum_t\sum_m P_{m,t}
}{
\max\{X,1\}
}.
$$

车辆供给总数为

$$
U
=
\sum_{t=0}^{T-1}
\sum_{m\in\mathcal M}
u_{m,t}.
$$

代码中的平均车辆等待时间估计为

$$
\bar W_V
=
\frac{
\sum_t\sum_m
\left(E_{m,t}+H_{m,t}\right)
}{
\max\{U,1\}
}.
$$

最大旅客队列为

$$
Q_P^{\max}
=
\max_t
\sum_m P_{m,t}.
$$

最大车辆队列为

$$
Q_V^{\max}
=
\max_t
\sum_m
\left(E_{m,t}+H_{m,t}\right).
$$

## 10. 因果瓶颈弹性

### 10.1 容量扰动实验

对候选资源 $i$ 的容量或供给执行相对扰动

$$
C_i^{\prime}
=(1+\varepsilon)C_i,
\qquad
\varepsilon=0.10.
$$

若资源 $i$ 为某种交通方式的车辆供给，则执行

$$
u_{i,t}^{\prime}
=(1+\varepsilon)u_{i,t}.
$$

在其他输入和随机场景保持不变的条件下重新运行仿真，得到扰动后的系统成本 $J_i^{+10\%}$。

### 10.2 成本下降率

资源 $i$ 扩容后的系统成本下降率为

$$
r_i
=
\frac{J_0-J_i^{+10\%}}{J_0},
$$

其中 $J_0$ 为基准情景的广义延误成本。

以百分数表示时：

$$
r_i^{(\%)}
=
100\%
\times
\frac{J_0-J_i^{+10\%}}{J_0}.
$$

### 10.3 因果瓶颈弹性

定义资源 $i$ 的因果瓶颈弹性为

$$
\mathcal E_i
=
-\frac{\Delta J/J_0}{\Delta C_i/C_i}
=
\frac{J_0-J_i^{+10\%}}
{0.1J_0}.
$$

其解释如下：

- $\mathcal E_i$ 越大，增加该资源越能降低系统总延误，该资源越可能是根因瓶颈；
- $\mathcal E_i\approx 0$，说明增加该资源对系统改善有限；
- $\mathcal E_i<0$，说明局部扩容反而提高系统成本，可能将更多流量推向下游瓶颈。

## 11. 表面拥堵与因果弹性联合判定

代码使用以下分段规则识别节点角色：

$$
\operatorname{Role}(i)
=
\begin{cases}
\text{根因瓶颈},
& \mathcal E_i\geq 0.15,
\\
\text{次级瓶颈},
& 0.02\leq\mathcal E_i<0.15,
\\
\text{表面拥堵},
& \mathcal E_i\leq0
\ \text{且}\ 
S_i\geq0.45,
\\
\text{非瓶颈},
& \text{其他情况}.
\end{cases}
$$

联合诊断逻辑可概括为：

| 表面拥堵指数 $S_i$ | 因果弹性 $\mathcal E_i$ | 解释 |
|---|---|---|
| 高 | 高 | 拥堵现象明显，扩容后系统显著改善，属于根因瓶颈 |
| 低或中 | 高 | 表面排队不一定突出，但属于隐蔽性根因瓶颈 |
| 高 | 低或负 | 主要由上下游回溢导致，属于表面拥堵 |
| 低 | 低 | 非关键资源 |

对于出租车、网约车和机场巴士的车辆供给资源，若

$$
\mathcal E_m>0.02,
$$

则将其加入瓶颈候选集，并使用对应上客区的表面拥堵指数作为绘图横坐标。若 $\mathcal E_m\geq0.15$，判定为根因运力瓶颈，否则判定为次级运力瓶颈。

## 12. 多随机种子稳定性分析

设重复仿真次数为 $R=30$，资源 $i$ 在第 $r$ 次仿真中的弹性为 $\mathcal E_i^{(r)}$。

平均因果瓶颈弹性为

$$
\overline{\mathcal E}_i
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathcal E_i^{(r)}.
$$

弹性的样本标准差为

$$
\sigma_i
=
\sqrt{
\frac{1}{R-1}
\sum_{r=1}^{R}
\left(
\mathcal E_i^{(r)}
-\overline{\mathcal E}_i
\right)^2
}.
$$

弹性为正的概率为

$$
p_i^{+}
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbf 1
\left(
\mathcal E_i^{(r)}>0
\right).
$$

资源成为第一瓶颈的概率为

$$
p_i^{\mathrm{Top1}}
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbf 1
\left[
i=
\arg\max_j
\mathcal E_j^{(r)}
\right].
$$

若某资源同时具有较大的 $\overline{\mathcal E}_i$、较小的 $\sigma_i$ 和较高的 $p_i^{\mathrm{Top1}}$，则说明其瓶颈地位具有较强的随机稳定性。

## 13. 模型计算流程

问题一的完整计算流程可写为

$$
\boxed{
\begin{aligned}
&\text{生成旅客需求与延迟车辆供给}
\\
&\Longrightarrow
\text{运行有限容量双边排队网络}
\\
&\Longrightarrow
\text{计算负荷率、队列与回溢概率}
\\
&\Longrightarrow
\text{计算表面拥堵指数 }S_i
\\
&\Longrightarrow
\text{逐项扩容 }10\%\text{ 并计算 }\mathcal E_i
\\
&\Longrightarrow
\text{联合 }(S_i,\mathcal E_i)\text{ 判定瓶颈类型}
\\
&\Longrightarrow
\text{通过 }30\text{ 次随机重复实验检验稳定性}.
\end{aligned}
}
$$

## 14. 与 `Framework.md` 的一致性说明

1. 本代码已实现框架中的有限容量双边排队网络、车辆供给延迟、道路有限存储、回溢传播、表面拥堵指数、容量扰动弹性和联合瓶颈判定。
2. 框架中的动态 Logit 交通方式选择尚未在问题一代码中启用。当前代码使用固定方式分担率 $s_m$，因此本文公式按固定分担率表述。
3. 框架允许采用元胞传输模型或有限存储容量道路模型。当前代码采用后一种形式，仅显式描述公共入口道路和离场道路。
4. 框架列出的平均等待时间、95% 分位等待时间、满载持续时间和系统延误贡献率并未全部写入节点诊断表；当前代码实际用于联合诊断的指标是 95% 分位负荷率、最大队列和回溢概率。
5. 广义成本权重 $(1,1.2,2,4)$ 为代码中的比较权重，主要服务于容量扰动排序。正式论文中应说明这些权重的标定依据或补充权重敏感性分析。
