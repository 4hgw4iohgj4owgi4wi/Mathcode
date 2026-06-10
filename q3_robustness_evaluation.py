"""问题3：评估问题2公平约束动态调度方案的鲁棒性。

运行：python q3_robustness_evaluation.py
输出：Mathcode/outputs/q3/ 下的 CSV、中文图表与结果摘要。
"""

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q1_bottleneck_analysis import CITIES, CN_CITY, MODES, generate_inputs, validate_configs
from q2_dynamic_dispatch import (
    TrafficCenter,
    base_control,
    select_mpc_action,
    static_control,
)


OUTPUT = Path(__file__).resolve().parent / "outputs" / "q3"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 六类场景覆盖随机、需求、供给、设施、信息和复合冲击。
SCENARIOS = {
    "random_fluctuation": {
        "name": "正常随机波动",
        "repeats": 3,
    },
    "demand_surge": {
        "name": "客流激增25%",
        "repeats": 3,
        "demand_factor": 1.25,
        "start": 45,
        "end": 215,
    },
    "vehicle_shortage": {
        "name": "营运车辆短缺30%",
        "repeats": 3,
        "supply_factor": {"taxi": 0.70, "ride_hailing": 0.70, "bus": 0.80},
        "start": 55,
        "end": 210,
    },
    "exit_incident": {
        "name": "离场能力骤降35%",
        "repeats": 3,
        "exit_factor": 0.65,
        "start": 90,
        "end": 150,
    },
    "forecast_delay": {
        "name": "预测信息滞后15分钟",
        "repeats": 3,
        "forecast_lag": 15,
    },
    "compound_extreme": {
        "name": "复合极端冲击",
        "repeats": 3,
        "demand_factor": 1.20,
        "supply_factor": {"taxi": 0.80, "ride_hailing": 0.80, "bus": 0.85},
        "exit_factor": 0.75,
        "forecast_lag": 10,
        "start": 65,
        "end": 190,
    },
}

POLICIES = ("baseline", "static", "dynamic_balanced")
METRICS = [
    "generalized_cost",
    "service_rate",
    "avg_passenger_wait_min",
    "p95_passenger_wait_min",
    "avg_driver_wait_min",
    "p95_driver_wait_min",
    "driver_fairness_jain",
    "service_opportunity_gap",
    "max_exit_occupancy",
    "congestion_minutes",
    "emergency_minutes",
    "emergency_shifted_passengers",
]


def apply_interval(frame, start, end, factors):
    """在指定冲击区间内按交通方式缩放输入。"""
    result = frame.astype(float).copy()
    mask = (result.index >= start) & (result.index < end)
    if np.isscalar(factors):
        result.loc[mask, list(MODES)] *= factors
    else:
        for mode, factor in factors.items():
            result.loc[mask, mode] *= factor
    return result


def delayed_forecast(frame, lag):
    """用滞后观测替代真实预测，检验控制器对信息误差的耐受性。"""
    if not lag:
        return frame.copy()
    return frame.shift(lag).bfill()


def build_scenario_inputs(city, seed, spec):
    demand, supply = generate_inputs(city, seed)
    start, end = spec.get("start", 0), spec.get("end", CITIES[city]["horizon"])
    if "demand_factor" in spec:
        demand = apply_interval(demand, start, end, spec["demand_factor"])
    if "supply_factor" in spec:
        supply = apply_interval(supply, start, end, spec["supply_factor"])
    lag = spec.get("forecast_lag", 0)
    return demand, supply, delayed_forecast(demand, lag), delayed_forecast(supply, lag)


def recovery_time(log, storage, event_end):
    """事故结束后连续5分钟保持离场道路低于80%容量所需时间。"""
    if event_end is None:
        return 0
    safe = log["exit_occupancy"].to_numpy() < 0.8 * storage
    for time in range(event_end, max(event_end, len(safe) - 4)):
        if safe[time : time + 5].all():
            return time - event_end
    return len(safe) - event_end


def emergency_overlay(sim, control, city, spec, time):
    """结构性冲击下的轻量应急覆盖层，不改变问题2常态MPC。"""
    structural = any(key in spec for key in ("demand_factor", "supply_factor", "exit_factor"))
    start, end = spec.get("start"), spec.get("end")
    if not structural or start is None or not (max(0, start - 10) <= time < end + 25):
        return control

    result = deepcopy(control)
    road_ratio = sim.exit_occupancy / sim.cfg["exit_storage"]
    passenger_queue = sum(sim.passenger.values())
    passenger_threshold = 720 if city == "Hongqiao" else 850
    road_risk = road_ratio >= 0.72 or ("exit_factor" in spec and start <= time < end)
    service_risk = passenger_queue >= passenger_threshold or "supply_factor" in spec
    if not road_risk and not service_risk:
        return control

    # 应急运力与分流均设置上限，避免以无限增援换取不现实的结果。
    result["private_diversion"] = max(
        result["private_diversion"], 0.30 if city == "Hongqiao" else 0.25
    )
    result["bus_extra"] = max(result["bus_extra"], 0.50 if city == "Hongqiao" else 0.65)
    result["fairness_weight"] = max(result["fairness_weight"], 0.90)

    if road_risk:
        # 道路风险优先：减少小客车进入核心区，保留高载客量巴士。
        for mode, factor in {"taxi": 0.74, "ride_hailing": 0.78, "private_car": 0.62}.items():
            result["release"][mode] *= factor
            result["gate"][mode] *= min(1.0, factor + 0.08)
    elif service_risk and road_ratio < 0.65:
        # 道路仍有余量时，提高可用营运车辆释放，缩短旅客长尾等待。
        floors = (
            {"taxi": 0.84, "ride_hailing": 0.76}
            if city == "Hongqiao"
            else {"taxi": 0.99, "ride_hailing": 0.85}
        )
        for mode, floor in floors.items():
            result["release"][mode] = max(result["release"][mode], floor)
            result["gate"][mode] = max(result["gate"][mode], min(1.0, floor + 0.08))

    result["emergency_mode"] = True
    return result


def emergency_mode_shift(demand_row, control, city):
    """应急模式下引导少量小客车旅客转乘加班巴士，总需求保持不变。"""
    result = demand_row.astype(float).copy()
    if not control.get("emergency_mode", False):
        return result, 0.0
    rates = (
        {"taxi": 0.04, "ride_hailing": 0.06}
        if city == "Hongqiao"
        else {"taxi": 0.06, "ride_hailing": 0.08}
    )
    shifted = 0.0
    for mode, rate in rates.items():
        amount = result[mode] * rate
        result[mode] -= amount
        result["bus"] += amount
        shifted += amount
    return result, shifted


def run_stress_policy(city, demand, supply, forecast_demand, forecast_supply, policy, spec):
    """运行带设施冲击和预测误差的调度仿真。"""
    sim, control = TrafficCenter(city), None
    normal_exit = sim.cfg["exit_road"]
    start, end = spec.get("start"), spec.get("end")
    shifted_total = 0.0
    for time in range(len(demand)):
        in_incident = start is not None and start <= time < end
        sim.cfg["exit_road"] = normal_exit * (spec.get("exit_factor", 1.0) if in_incident else 1.0)

        if policy == "baseline":
            control = base_control()
        elif policy == "static":
            control = static_control(city)
        elif time % 5 == 0:
            control = select_mpc_action(
                sim,
                time,
                forecast_demand,
                forecast_supply,
                city,
                "dynamic_balanced",
            )
            control = emergency_overlay(sim, control, city, spec, time)
        demand_now, shifted = emergency_mode_shift(demand.iloc[time], control, city)
        shifted_total += shifted
        sim.step(time, demand_now, supply.iloc[time], control)
        sim.logs[-1]["emergency_mode"] = int(control.get("emergency_mode", False))
        sim.logs[-1]["emergency_shifted_passengers"] = shifted

    log = pd.DataFrame(sim.logs)
    summary = sim.summary(policy)
    summary["emergency_minutes"] = int(log["emergency_mode"].sum())
    summary["emergency_shifted_passengers"] = shifted_total
    summary["recovery_time_min"] = recovery_time(
        log,
        CITIES[city]["exit_storage"],
        end if "exit_factor" in spec else None,
    )
    return summary, log


def run_experiments():
    rows, trajectories = [], {}
    for city in CITIES:
        for scenario, spec in SCENARIOS.items():
            for repeat in range(spec["repeats"]):
                # 各冲击场景使用共同随机数，隔离冲击本身造成的性能变化。
                seed = CITIES[city]["seed"] + 1000 + repeat
                inputs = build_scenario_inputs(city, seed, spec)
                for policy in POLICIES:
                    summary, log = run_stress_policy(city, *inputs, policy, spec)
                    summary.update(
                        {
                            "scenario": scenario,
                            "scenario_cn": spec["name"],
                            "repeat": repeat,
                            "seed": seed,
                        }
                    )
                    rows.append(summary)
                    if scenario == "compound_extreme" and repeat == 0:
                        trajectories[(city, policy)] = log
                print(f"完成：{CN_CITY[city]} / {spec['name']} / 第{repeat + 1}次")
    return pd.DataFrame(rows), trajectories


def compare_policies(raw):
    """逐次实验比较动态方案相对无调度和固定规则的抗冲击收益。"""
    index = ["city", "scenario", "scenario_cn", "repeat", "seed"]
    wide = raw.pivot(index=index, columns="policy", values=METRICS + ["recovery_time_min"])
    rows = []
    for key, row in wide.iterrows():
        result = dict(zip(index, key))
        base, static, dynamic = (
            row.xs("baseline", level="policy"),
            row.xs("static", level="policy"),
            row.xs("dynamic_balanced", level="policy"),
        )
        result.update(
            {
                "cost_reduction_pct": 100 * (1 - dynamic["generalized_cost"] / base["generalized_cost"]),
                "cost_gain_vs_static_pct": 100
                * (1 - dynamic["generalized_cost"] / static["generalized_cost"]),
                "passenger_wait_reduction_pct": 100
                * (1 - dynamic["avg_passenger_wait_min"] / base["avg_passenger_wait_min"]),
                "driver_wait_reduction_pct": 100
                * (1 - dynamic["avg_driver_wait_min"] / base["avg_driver_wait_min"]),
                "service_delta_pp": 100 * (dynamic["service_rate"] - base["service_rate"]),
                "fairness_delta": dynamic["driver_fairness_jain"] - base["driver_fairness_jain"],
                "congestion_minutes_avoided": base["congestion_minutes"]
                - dynamic["congestion_minutes"],
                "dynamic_recovery_time_min": dynamic["recovery_time_min"],
                "baseline_recovery_time_min": base["recovery_time_min"],
                "dynamic_cost": dynamic["generalized_cost"],
                "dynamic_service_rate": dynamic["service_rate"],
                "dynamic_passenger_wait": dynamic["avg_passenger_wait_min"],
                "dynamic_p95_passenger_wait": dynamic["p95_passenger_wait_min"],
                "dynamic_driver_wait": dynamic["avg_driver_wait_min"],
                "dynamic_p95_driver_wait": dynamic["p95_driver_wait_min"],
                "dynamic_fairness": dynamic["driver_fairness_jain"],
                "dynamic_opportunity_gap": dynamic["service_opportunity_gap"],
                "dynamic_congestion_minutes": dynamic["congestion_minutes"],
                "dynamic_emergency_minutes": dynamic["emergency_minutes"],
                "dynamic_shifted_passengers": dynamic["emergency_shifted_passengers"],
            }
        )
        # 安全服务底线同时约束服务率、公平性、长尾等待和道路拥堵。
        result["safety_pass"] = (
            dynamic["service_rate"] >= 0.90
            and dynamic["driver_fairness_jain"] >= 0.98
            and dynamic["p95_passenger_wait_min"] <= 45
            and dynamic["congestion_minutes"] <= 60
        )
        rows.append(result)
    return pd.DataFrame(rows)


def add_retention_score(comparison):
    """相对正常场景评价冲击后性能保持程度，满分为100。"""
    result = comparison.copy()
    score = np.zeros(len(result))
    weights = {
        "dynamic_cost": 0.30,
        "dynamic_passenger_wait": 0.20,
        "dynamic_driver_wait": 0.15,
        "dynamic_p95_passenger_wait": 0.15,
        "dynamic_service_rate": 0.10,
        "dynamic_fairness": 0.10,
    }
    for city in CITIES:
        city_mask = result["city"] == city
        nominal = result[city_mask & (result["scenario"] == "random_fluctuation")]
        for metric, weight in weights.items():
            reference = nominal[metric].mean()
            values = result.loc[city_mask, metric]
            ratio = values / reference if metric in ("dynamic_service_rate", "dynamic_fairness") else reference / values
            score[city_mask] += weight * np.minimum(ratio, 1.0)
    result["retention_score"] = 100 * score
    return result


def aggregate_results(raw, comparison):
    summary = (
        raw.groupby(["city", "scenario", "scenario_cn", "policy"])[METRICS + ["recovery_time_min"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in col if x).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    robust = (
        comparison.groupby(["city", "scenario", "scenario_cn"])
        .agg(
            cost_reduction_mean=("cost_reduction_pct", "mean"),
            cost_reduction_std=("cost_reduction_pct", "std"),
            cost_gain_vs_static_mean=("cost_gain_vs_static_pct", "mean"),
            passenger_wait_reduction_mean=("passenger_wait_reduction_pct", "mean"),
            driver_wait_reduction_mean=("driver_wait_reduction_pct", "mean"),
            service_delta_mean=("service_delta_pp", "mean"),
            fairness_delta_mean=("fairness_delta", "mean"),
            congestion_avoided_mean=("congestion_minutes_avoided", "mean"),
            dynamic_recovery_mean=("dynamic_recovery_time_min", "mean"),
            baseline_recovery_mean=("baseline_recovery_time_min", "mean"),
            retention_score_mean=("retention_score", "mean"),
            safety_pass_rate=("safety_pass", "mean"),
            emergency_minutes_mean=("dynamic_emergency_minutes", "mean"),
            shifted_passengers_mean=("dynamic_shifted_passengers", "mean"),
        )
        .reset_index()
    )
    return summary, robust


def plot_robustness(robust, trajectories):
    scenarios = list(SCENARIOS)
    labels = [SCENARIOS[s]["name"] for s in scenarios]

    matrix = robust.pivot(index="city", columns="scenario", values="retention_score_mean").reindex(
        index=list(CITIES), columns=scenarios
    )
    fig, ax = plt.subplots(figsize=(13, 4))
    image = ax.imshow(matrix, cmap="YlGn", vmin=70, vmax=100, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=18)
    ax.set_yticks(range(len(CITIES)), [CN_CITY[c] for c in CITIES])
    for i in range(len(CITIES)):
        for j in range(len(scenarios)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.1f}", ha="center", va="center")
    ax.set_title("优化后动态调度方案性能保持分数")
    fig.colorbar(image, ax=ax, label="鲁棒性保持分数")
    fig.tight_layout()
    fig.savefig(OUTPUT / "robustness_score_matrix.png", dpi=180)
    plt.close(fig)

    for city in CITIES:
        data = robust.query("city == @city").set_index("scenario").reindex(scenarios)
        x = np.arange(len(data))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].bar(x - 0.2, data["cost_reduction_mean"], 0.4, label="综合成本下降")
        axes[0].bar(x + 0.2, data["passenger_wait_reduction_mean"], 0.4, label="旅客等待下降")
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_ylabel("相对无调度改善（%）")
        axes[0].legend()
        axes[1].bar(x - 0.2, data["retention_score_mean"], 0.4, label="性能保持分数")
        axes[1].bar(x + 0.2, 100 * data["safety_pass_rate"], 0.4, label="安全底线通过率")
        axes[1].axhline(80, ls="--", color="#D62728", label="80分参考线")
        axes[1].set(ylabel="分数 / 通过率（%）", ylim=(0, 105))
        axes[1].set_xticks(x, labels, rotation=18)
        axes[1].legend()
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{CN_CITY[city]}：多场景鲁棒性评估")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{city}_robustness_overview.png", dpi=180)
        plt.close(fig)

        base, dynamic = trajectories[(city, "baseline")], trajectories[(city, "dynamic_balanced")]
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(base["time"], base["passenger_queue"], label="无调度")
        axes[0].plot(dynamic["time"], dynamic["passenger_queue"], label="优化动态调度")
        axes[0].set_ylabel("旅客排队人数")
        axes[0].legend()
        axes[1].plot(base["time"], base["exit_occupancy"], label="无调度")
        axes[1].plot(dynamic["time"], dynamic["exit_occupancy"], label="优化动态调度")
        axes[1].axhline(
            0.9 * CITIES[city]["exit_storage"], ls="--", color="grey", label="拥堵阈值"
        )
        axes[1].axvspan(65, 190, color="#E45756", alpha=0.12, label="复合冲击区间")
        axes[1].set(xlabel="时间（分钟）", ylabel="离场道路车辆数")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.suptitle(f"{CN_CITY[city]}：复合极端冲击下的系统状态")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{city}_compound_trajectory.png", dpi=180)
        plt.close(fig)


def write_report(robust):
    lines = [
        "# 问题3计算结果摘要",
        "",
        "采用随机波动、客流激增、车辆短缺、道路事故、信息滞后和复合极端冲击六类场景，",
        "比较含轻量应急覆盖层的公平约束动态调度、固定规则和无调度方案。",
        "",
    ]
    for city in CITIES:
        data = robust.query("city == @city").set_index("scenario").reindex(list(SCENARIOS))
        worst = data["retention_score_mean"].idxmin()
        lines.extend(
            [
                f"## {CN_CITY[city]}",
                "",
                f"- 最低性能保持分数：{data.loc[worst, 'retention_score_mean']:.2f}，"
                f"发生于{SCENARIOS[worst]['name']}；",
                f"- 所有场景平均成本下降：{data['cost_reduction_mean'].mean():.2f}%；",
                f"- 最差场景成本下降：{data['cost_reduction_mean'].min():.2f}%；",
                f"- 安全服务底线平均通过率：{data['safety_pass_rate'].mean():.2%}；",
                f"- 结构性冲击下平均应急模式持续："
                f"{data.loc[data['emergency_minutes_mean'] > 0, 'emergency_minutes_mean'].mean():.1f}分钟；",
                "",
                "| 场景 | 成本下降 | 相对固定规则成本改善 | 旅客等待下降 | 驾驶员等待下降 | "
                "保持分数 | 安全底线通过率 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for scenario, row in data.iterrows():
            lines.append(
                f"| {SCENARIOS[scenario]['name']} | {row['cost_reduction_mean']:.2f}% | "
                f"{row['cost_gain_vs_static_mean']:.2f}% | "
                f"{row['passenger_wait_reduction_mean']:.2f}% | "
                f"{row['driver_wait_reduction_mean']:.2f}% | "
                f"{row['retention_score_mean']:.2f} | {row['safety_pass_rate']:.0%} |"
            )
        lines.append("")
    (OUTPUT / "problem3_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    validate_configs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw, trajectories = run_experiments()
    comparison = add_retention_score(compare_policies(raw))
    scenario_summary, robust = aggregate_results(raw, comparison)

    raw.to_csv(OUTPUT / "robustness_raw.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(OUTPUT / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(OUTPUT / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    plot_robustness(robust, trajectories)
    write_report(robust)

    display = robust[
        [
            "city",
            "scenario_cn",
            "cost_reduction_mean",
            "retention_score_mean",
            "safety_pass_rate",
        ]
    ].copy()
    display["city"] = display["city"].map(CN_CITY)
    print(display.round(2).to_string(index=False))
    print(f"\n结果已写入：{OUTPUT}")


if __name__ == "__main__":
    main()
