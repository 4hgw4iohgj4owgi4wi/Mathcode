"""问题2：机场交通中心多主体动态调度。

运行：python q2_dynamic_dispatch.py
输出：Mathcode/outputs/q2/ 下的 CSV、中文图表和结果摘要。
"""

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q1_bottleneck_analysis import (
    CITIES,
    CN_CITY,
    CN_MODE,
    MODES,
    generate_inputs,
    proportional_limit,
    validate_configs,
)


OUTPUT = Path(__file__).resolve().parent / "outputs" / "q2"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

POLICY_CN = {
    "baseline": "无调度",
    "static": "固定规则调度",
    "dynamic_efficiency": "效率优先动态调度",
    "dynamic_balanced": "公平约束动态调度",
}

# 驾驶员单次服务收益仅用于构造可比的收益机会指标。
FARE = {"taxi": 85.0, "ride_hailing": 90.0}


def priority_limit(request, capacity, priority):
    """共享容量不足时按优先权分配，并保证分配量不超过实际请求。"""
    if sum(request.values()) <= capacity:
        return request.copy()
    remaining, result = capacity, {m: 0.0 for m in MODES}
    active = {m for m in MODES if request[m] > 0}
    while active and remaining > 1e-9:
        weight_sum = sum(priority[m] * (request[m] - result[m]) for m in active)
        if weight_sum <= 0:
            break
        used = 0.0
        for m in list(active):
            room = request[m] - result[m]
            add = min(room, remaining * priority[m] * room / weight_sum)
            result[m] += add
            used += add
            if request[m] - result[m] <= 1e-9:
                active.remove(m)
        if used <= 1e-9:
            break
        remaining -= used
    return result


def add_cohort(cohorts, time, amount):
    if amount > 1e-9:
        cohorts.append([time, float(amount)])


def serve_cohorts(cohorts, amount, time, records):
    """按FIFO服务队列，并以(等待时间, 数量)记录加权等待分布。"""
    remaining = amount
    while cohorts and remaining > 1e-9:
        arrival, count = cohorts[0]
        served = min(count, remaining)
        records.append((time - arrival, served))
        cohorts[0][1] -= served
        remaining -= served
        if cohorts[0][1] <= 1e-9:
            cohorts.pop(0)


def weighted_quantile(records, q):
    if not records:
        return 0.0
    data = sorted(records)
    target, cumulative = q * sum(w for _, w in data), 0.0
    for value, weight in data:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(data[-1][0])


def jain_index(values):
    values = np.maximum(np.asarray(values, dtype=float), 1e-6)
    return values.sum() ** 2 / (len(values) * np.square(values).sum())


def base_control():
    return {
        "release": {m: 1.0 for m in MODES},
        "gate": {m: 1.0 for m in MODES},
        "private_diversion": 0.0,
        "bus_extra": 0.0,
        "fairness_weight": 0.0,
        "dynamic_curb": False,
        "priority": False,
        "label": "baseline",
    }


def static_control(city):
    """依据问题1瓶颈给出不随状态变化的固定规则方案。"""
    if city == "Hongqiao":
        meter, diversion, bus_extra = 0.72, 0.18, 0.20
    else:
        meter, diversion, bus_extra = 0.80, 0.10, 0.25
    return {
        "release": {
            "taxi": meter,
            "ride_hailing": meter * 0.92,
            "private_car": max(0.60, meter - 0.05),
            "bus": 1.0,
        },
        "gate": {
            "taxi": min(1.0, meter + 0.12),
            "ride_hailing": min(1.0, meter + 0.05),
            "private_car": min(1.0, meter + 0.05),
            "bus": 1.0,
        },
        "private_diversion": diversion,
        "bus_extra": bus_extra,
        "fairness_weight": 0.55,
        "dynamic_curb": True,
        "priority": True,
        "label": "static",
    }


def candidate_controls(city, policy):
    """生成滚动优化候选动作；两机场使用不同的控制边界。"""
    if city == "Hongqiao":
        meters, bus_levels, diversions = (0.52, 0.68, 0.84), (0.0, 0.18, 0.36), (0.08, 0.24)
    else:
        meters, bus_levels, diversions = (0.62, 0.78, 0.94), (0.0, 0.22, 0.44), (0.05, 0.18)

    if policy == "dynamic_efficiency":
        fairness_levels, rh_factors = (0.35,), (0.72, 0.90)
    else:
        # 公平方案允许对网约车实施更严格预约，提升其接单机会与等待收益。
        fairness_levels, rh_factors = (0.80,), (0.72, 0.90)

    actions = []
    for meter in meters:
        for bus_extra in bus_levels:
            for diversion in diversions:
                for fairness in fairness_levels:
                    for rh_factor in rh_factors:
                        rh_release = meter * rh_factor
                        actions.append(
                            {
                                "release": {
                                    "taxi": min(
                                        1.0,
                                        meter + (0.05 if city == "Xiaoshan" else 0.0),
                                    ),
                                    "ride_hailing": rh_release,
                                    "private_car": max(0.55, meter - 0.05),
                                    "bus": 1.0,
                                },
                                "gate": {
                                    "taxi": min(1.0, meter + 0.15),
                                    "ride_hailing": min(1.0, rh_release + 0.10),
                                    "private_car": min(1.0, meter + 0.08),
                                    "bus": 1.0,
                                },
                                "private_diversion": diversion,
                                "bus_extra": bus_extra,
                                "fairness_weight": fairness,
                                "dynamic_curb": True,
                                "priority": True,
                                "label": "candidate",
                            }
                        )
    return actions


class TrafficCenter:
    """在问题一有限容量排队网络上增加可控准入、泊位和公交调度。"""

    def __init__(self, city):
        self.city, self.cfg = city, deepcopy(CITIES[city])
        self.passenger = {m: 0.0 for m in MODES}
        self.remote = {m: 0.0 for m in MODES}
        self.external = {m: 0.0 for m in MODES}
        self.stage = {m: 0.0 for m in MODES}
        self.exit_occupancy = 0.0
        self.pending_bus = np.zeros(self.cfg["horizon"] + 90)

        self.passenger_cohorts = {m: [] for m in MODES}
        self.vehicle_cohorts = {m: [] for m in MODES}
        self.passenger_waits = {m: [] for m in MODES}
        self.driver_waits = {m: [] for m in MODES}

        self.original_demand = {m: 0.0 for m in MODES}
        self.accepted_vehicles = {m: 0.0 for m in MODES}
        self.served_people = {m: 0.0 for m in MODES}
        self.served_vehicles = {m: 0.0 for m in MODES}
        self.passenger_area = {m: 0.0 for m in MODES}
        self.driver_area = {m: 0.0 for m in MODES}
        self.remote_area = {m: 0.0 for m in MODES}

        self.road_area = self.block_area = 0.0
        self.diverted_people = self.extra_bus_dispatched = 0.0
        self.adjustment_cost = 0.0
        self.logs, self.last_control = [], None

    def clone(self):
        return deepcopy(self)

    def curb_capacity(self, control):
        base = self.cfg["curb"]
        if not control["dynamic_curb"]:
            return base.copy()

        road_modes, flexible_ratio = ("taxi", "ride_hailing", "private_car"), 0.18
        capacity = {m: base[m] for m in MODES}
        pool = sum(base[m] * flexible_ratio for m in road_modes)
        score = {}
        for m in road_modes:
            passenger_pressure = self.passenger[m] / (self.cfg["load"][m] * base[m] + 1)
            driver_pressure = (self.external[m] + self.stage[m]) / (base[m] + 1)
            driver_weight = control["fairness_weight"] if m != "private_car" else 0.15
            score[m] = 1 + passenger_pressure + driver_weight * driver_pressure
        total = sum(score.values())
        for m in road_modes:
            capacity[m] = base[m] * (1 - flexible_ratio) + pool * score[m] / total
        return capacity

    def control_change(self, control):
        if self.last_control is None:
            return 0.0
        change = abs(control["private_diversion"] - self.last_control["private_diversion"])
        change += 2 * abs(control["bus_extra"] - self.last_control["bus_extra"])
        change += sum(
            abs(control["release"][m] - self.last_control["release"][m]) for m in MODES
        )
        return change

    def step(self, time, demand_row, supply_row, control, record=True):
        demand_now = {m: float(demand_row[m]) for m in MODES}
        supply_now = {m: float(supply_row[m]) for m in MODES}

        # 私家车转入停车楼，不再占用航站楼前即停即走区。
        diverted = demand_now["private_car"] * control["private_diversion"]
        demand_now["private_car"] -= diverted
        supply_now["private_car"] *= 1 - control["private_diversion"]
        self.diverted_people += diverted

        # 加班巴士按调度提前量进入系统。
        bus_due = self.pending_bus[time]
        supply_now["bus"] += bus_due
        bus_arrival = time + self.cfg["supply_lag"]["bus"]
        self.pending_bus[bus_arrival] += control["bus_extra"]
        self.extra_bus_dispatched += control["bus_extra"]

        for m in MODES:
            self.original_demand[m] += float(demand_row[m])
            self.passenger[m] += demand_now[m]
            add_cohort(self.passenger_cohorts[m], time, demand_now[m])

            # 预约准入：未释放车辆留在远端，不计作机场内驾驶员排队。
            self.remote[m] += supply_now[m]
            need = max(self.passenger[m] / self.cfg["load"][m] - self.external[m] - self.stage[m], 0)
            release_target = control["release"][m] * supply_now[m] + 0.18 * need
            released = min(self.remote[m], release_target)
            self.remote[m] -= released
            self.external[m] += released
            self.accepted_vehicles[m] += released
            add_cohort(self.vehicle_cohorts[m], time, released)

        self.exit_occupancy -= min(self.exit_occupancy, self.cfg["exit_road"])

        gate_raw = {
            m: min(self.external[m], max(self.cfg["stage"][m] - self.stage[m], 0.0))
            for m in MODES
        }
        gate_request = {
            m: min(gate_raw[m], self.cfg["gate"][m] * control["gate"][m]) for m in MODES
        }
        gate_priority = {
            m: 1
            + self.passenger[m] / (self.cfg["load"][m] * self.cfg["gate"][m] + 1)
            + control["fairness_weight"]
            * (self.external[m] + self.stage[m])
            / (self.cfg["gate"][m] + 1)
            for m in MODES
        }
        admitted = (
            priority_limit(gate_request, self.cfg["entry_road"], gate_priority)
            if control["priority"]
            else proportional_limit(gate_request, self.cfg["entry_road"])
        )
        for m in MODES:
            self.external[m] -= admitted[m]
            self.stage[m] += admitted[m]

        curb = self.curb_capacity(control)
        curb_raw = {m: min(self.stage[m], self.passenger[m] / self.cfg["load"][m]) for m in MODES}
        curb_request = {m: min(curb_raw[m], curb[m]) for m in MODES}
        exit_space = max(self.cfg["exit_storage"] - self.exit_occupancy, 0.0)
        exit_priority = {
            m: 1
            + self.passenger[m] / (self.cfg["load"][m] * curb[m] + 1)
            + control["fairness_weight"] * self.stage[m] / (curb[m] + 1)
            for m in MODES
        }
        served_vehicle = (
            priority_limit(curb_request, exit_space, exit_priority)
            if control["priority"]
            else proportional_limit(curb_request, exit_space)
        )

        served_person = {}
        for m in MODES:
            served_person[m] = min(self.passenger[m], served_vehicle[m] * self.cfg["load"][m])
            self.stage[m] -= served_vehicle[m]
            self.passenger[m] -= served_person[m]
            self.served_people[m] += served_person[m]
            self.served_vehicles[m] += served_vehicle[m]
            serve_cohorts(self.passenger_cohorts[m], served_person[m], time, self.passenger_waits[m])
            serve_cohorts(self.vehicle_cohorts[m], served_vehicle[m], time, self.driver_waits[m])
        self.exit_occupancy += sum(served_vehicle.values())

        entry_block = sum(gate_request.values()) - sum(admitted.values())
        exit_block = sum(curb_request.values()) - sum(served_vehicle.values())
        passenger_q = sum(self.passenger.values())
        driver_q = sum(self.external.values()) + sum(self.stage.values())
        imbalance = abs(
            np.log1p(self.external["taxi"] + self.stage["taxi"])
            - np.log1p(self.external["ride_hailing"] + self.stage["ride_hailing"])
        )
        taxi_opportunity = self.served_vehicles["taxi"] / max(
            self.accepted_vehicles["taxi"], 1
        )
        ride_opportunity = self.served_vehicles["ride_hailing"] / max(
            self.accepted_vehicles["ride_hailing"], 1
        )
        fairness_gap = abs(taxi_opportunity - ride_opportunity)
        change = self.control_change(control)

        for m in MODES:
            self.passenger_area[m] += self.passenger[m]
            self.driver_area[m] += self.external[m] + self.stage[m]
            self.remote_area[m] += self.remote[m]
        self.road_area += self.exit_occupancy
        self.block_area += entry_block + exit_block
        self.adjustment_cost += change

        components = {
            "passenger_queue": passenger_q,
            "driver_queue": driver_q,
            "road": self.exit_occupancy,
            "block": entry_block + exit_block,
            "imbalance": imbalance,
            "fairness_gap": fairness_gap,
            "diversion": diverted,
            "bus_extra": control["bus_extra"],
            "adjustment": change,
        }

        if record:
            row = {
                "time": time,
                **components,
                "exit_occupancy": self.exit_occupancy,
                "private_diversion": control["private_diversion"],
                "bus_extra": control["bus_extra"],
            }
            for m in MODES:
                row.update(
                    {
                        f"p_{m}": self.passenger[m],
                        f"v_{m}": self.external[m] + self.stage[m],
                        f"release_{m}": control["release"][m],
                        f"curb_{m}": curb[m],
                        f"served_{m}": served_person[m],
                    }
                )
            self.logs.append(row)
        self.last_control = deepcopy(control)
        return components

    def driver_fairness(self):
        utilities = []
        for m in ("taxi", "ride_hailing"):
            accepted = max(self.accepted_vehicles[m], 1)
            service_opportunity = self.served_vehicles[m] / accepted
            avg_wait = self.driver_area[m] / accepted
            utilities.append(max(FARE[m] * service_opportunity - 0.6 * avg_wait, 0.1))
        return jain_index(utilities)

    def summary(self, policy):
        horizon = self.cfg["horizon"]
        passenger_records, driver_records = [], []
        for m in MODES:
            passenger_records.extend(self.passenger_waits[m])
            driver_records.extend(self.driver_waits[m])
            for arrival, count in self.passenger_cohorts[m]:
                passenger_records.append((horizon - arrival, count))
            for arrival, count in self.vehicle_cohorts[m]:
                driver_records.append((horizon - arrival, count))

        demand = sum(self.original_demand.values())
        served = sum(self.served_people.values()) + self.diverted_people
        accepted = sum(self.accepted_vehicles.values())
        p_area = sum(self.passenger_area.values())
        d_area = sum(self.driver_area.values())
        fairness = self.driver_fairness()
        generalized_cost = (
            p_area
            + 1.2 * d_area
            + 2 * self.road_area
            + 4 * self.block_area
            + 5 * self.diverted_people
            + 45 * self.extra_bus_dispatched
            + 30 * self.adjustment_cost
            + 5000 * (1 - fairness)
        )
        log = pd.DataFrame(self.logs)
        return {
            "city": self.city,
            "policy": policy,
            "generalized_cost": generalized_cost,
            "service_rate": served / demand,
            "avg_passenger_wait_min": p_area / max(served, 1),
            "p95_passenger_wait_min": weighted_quantile(passenger_records, 0.95),
            "avg_driver_wait_min": d_area / max(accepted, 1),
            "p95_driver_wait_min": weighted_quantile(driver_records, 0.95),
            "driver_fairness_jain": fairness,
            "service_opportunity_gap": abs(
                self.served_vehicles["taxi"] / max(self.accepted_vehicles["taxi"], 1)
                - self.served_vehicles["ride_hailing"]
                / max(self.accepted_vehicles["ride_hailing"], 1)
            ),
            "taxi_service_opportunity": self.served_vehicles["taxi"]
            / max(self.accepted_vehicles["taxi"], 1),
            "ride_hailing_service_opportunity": self.served_vehicles["ride_hailing"]
            / max(self.accepted_vehicles["ride_hailing"], 1),
            "max_passenger_queue": log["passenger_queue"].max(),
            "max_vehicle_queue": log["driver_queue"].max(),
            "max_exit_occupancy": log["exit_occupancy"].max(),
            "congestion_minutes": (log["exit_occupancy"] >= 0.9 * self.cfg["exit_storage"]).sum(),
            "diverted_private_passengers": self.diverted_people,
            "extra_buses": self.extra_bus_dispatched,
            "control_adjustment": self.adjustment_cost,
        }


def objective(components, weights):
    return sum(weights[k] * components[k] for k in weights)


def policy_weights(city, policy):
    if policy == "dynamic_efficiency":
        return {
            "passenger_queue": 1.0,
            "driver_queue": 0.65,
            "road": 2.7 if city == "Hongqiao" else 2.3,
            "block": 6.0,
            "imbalance": 15.0,
            "fairness_gap": 60.0,
            "diversion": 4.0,
            "bus_extra": 45.0,
            "adjustment": 18.0,
        }
    return {
        "passenger_queue": 1.0,
        "driver_queue": 1.15,
        "road": 2.8 if city == "Hongqiao" else 2.4,
        "block": 6.5,
        "imbalance": 65.0,
        "fairness_gap": 2000.0,
        "diversion": 4.5,
        "bus_extra": 48.0,
        "adjustment": 25.0,
    }


def select_mpc_action(sim, time, demand, supply, city, policy, forecast=30):
    """枚举可实施动作，在滚动预测窗口内选择综合成本最低者。"""
    weights, best_score, best = policy_weights(city, policy), np.inf, None
    end = min(time + forecast, len(demand))
    for action in candidate_controls(city, policy):
        trial, score = sim.clone(), 0.0
        for k in range(time, end):
            components = trial.step(k, demand.iloc[k], supply.iloc[k], action, record=False)
            score += objective(components, weights)
        terminal = sum(trial.passenger.values()) + 1.2 * (
            sum(trial.external.values()) + sum(trial.stage.values())
        )
        score += 3.0 * terminal
        if policy == "dynamic_balanced":
            opportunity_gap = abs(
                trial.served_vehicles["taxi"] / max(trial.accepted_vehicles["taxi"], 1)
                - trial.served_vehicles["ride_hailing"]
                / max(trial.accepted_vehicles["ride_hailing"], 1)
            )
            score += 30000 * opportunity_gap
        if score < best_score:
            best_score, best = score, action
    best = deepcopy(best)
    best["label"] = policy
    return best


def run_policy(city, demand, supply, policy):
    sim, control = TrafficCenter(city), None
    for time in range(len(demand)):
        if policy == "baseline":
            control = base_control()
        elif policy == "static":
            control = static_control(city)
        elif time % 5 == 0:
            control = select_mpc_action(sim, time, demand, supply, city, policy)
        sim.step(time, demand.iloc[time], supply.iloc[time], control)
    return sim, sim.summary(policy)


def plot_comparison(city, summary, logs):
    city_summary = summary.query("city == @city").copy()
    city_summary["方案"] = city_summary["policy"].map(POLICY_CN)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(city_summary))
    axes[0].bar(x - 0.18, city_summary["avg_passenger_wait_min"], 0.36, label="旅客")
    axes[0].bar(x + 0.18, city_summary["avg_driver_wait_min"], 0.36, label="驾驶员")
    axes[0].set(ylabel="平均等待时间（分钟）", title="等待时间")
    axes[0].set_xticks(x, city_summary["方案"], rotation=18)
    axes[0].legend()

    axes[1].bar(x, city_summary["driver_fairness_jain"], color="#54A24B")
    axes[1].set_ylim(0.8, 1.005)
    axes[1].set(ylabel="Jain公平指数", title="驾驶员收益机会公平性")
    axes[1].set_xticks(x, city_summary["方案"], rotation=18)

    base_cost = city_summary.iloc[0]["generalized_cost"]
    reduction = 100 * (1 - city_summary["generalized_cost"] / base_cost)
    axes[2].bar(x, reduction, color="#E45756")
    axes[2].set(ylabel="相对无调度成本下降（%）", title="综合运行效率")
    axes[2].set_xticks(x, city_summary["方案"], rotation=18)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{CN_CITY[city]}：调度方案综合比较")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{city}_policy_comparison.png", dpi=180)
    plt.close(fig)

    baseline, dynamic = logs[(city, "baseline")], logs[(city, "dynamic_balanced")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(baseline["time"], baseline["passenger_queue"], label="无调度")
    axes[0].plot(dynamic["time"], dynamic["passenger_queue"], label="公平约束动态调度")
    axes[0].set_ylabel("旅客排队人数")
    axes[0].legend()
    axes[1].plot(baseline["time"], baseline["exit_occupancy"], label="无调度")
    axes[1].plot(dynamic["time"], dynamic["exit_occupancy"], label="公平约束动态调度")
    axes[1].axhline(0.9 * CITIES[city]["exit_storage"], ls="--", color="grey", label="拥堵阈值")
    axes[1].set(xlabel="时间（分钟）", ylabel="离场道路车辆数")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"{CN_CITY[city]}：动态调度前后系统状态")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{city}_state_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(dynamic["time"], dynamic["release_taxi"], label="出租车")
    axes[0].plot(dynamic["time"], dynamic["release_ride_hailing"], label="网约车")
    axes[0].set_ylabel("预约释放比例")
    axes[0].legend()
    axes[1].plot(dynamic["time"], dynamic["bus_extra"], color="#D62728")
    axes[1].set_ylabel("巴士加班量\n（辆/分钟）")
    axes[2].plot(dynamic["time"], dynamic["private_diversion"], color="#2CA02C")
    axes[2].set(xlabel="时间（分钟）", ylabel="私家车分流比例")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"{CN_CITY[city]}：公平约束动态调度控制轨迹")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{city}_control_trajectory.png", dpi=180)
    plt.close(fig)


def summarize_controls(logs):
    """汇总各方案的实际控制强度，便于形成可执行调度表。"""
    rows = []
    for (city, policy), log in logs.items():
        rows.append(
            {
                "city": city,
                "policy": policy,
                "avg_taxi_release": log["release_taxi"].mean(),
                "avg_ride_hailing_release": log["release_ride_hailing"].mean(),
                "avg_private_car_release": log["release_private_car"].mean(),
                "avg_private_diversion": log["private_diversion"].mean(),
                "avg_bus_extra_per_min": log["bus_extra"].mean(),
                "max_exit_occupancy": log["exit_occupancy"].max(),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary, controls):
    lines = [
        "# 问题2计算结果摘要",
        "",
        "最终推荐方案为公平约束动态调度；效率优先方案用于展示公平约束的影响。",
        "",
    ]
    for city in CITIES:
        data = summary.query("city == @city").set_index("policy")
        base, final = data.loc["baseline"], data.loc["dynamic_balanced"]
        control = controls.query(
            "city == @city and policy == 'dynamic_balanced'"
        ).iloc[0]
        lines.extend(
            [
                f"## {CN_CITY[city]}",
                "",
                f"- 综合成本下降：{100 * (1 - final['generalized_cost'] / base['generalized_cost']):.2f}%",
                f"- 旅客平均等待：{base['avg_passenger_wait_min']:.2f} → "
                f"{final['avg_passenger_wait_min']:.2f} 分钟",
                f"- 驾驶员平均等待：{base['avg_driver_wait_min']:.2f} → "
                f"{final['avg_driver_wait_min']:.2f} 分钟",
                f"- 驾驶员公平指数：{base['driver_fairness_jain']:.3f} → "
                f"{final['driver_fairness_jain']:.3f}",
                f"- 出租车—网约车服务机会差：{base['service_opportunity_gap']:.3f} → "
                f"{final['service_opportunity_gap']:.3f}",
                f"- 离场道路拥堵持续：{base['congestion_minutes']:.0f} → "
                f"{final['congestion_minutes']:.0f} 分钟",
                f"- 动态方案平均控制：出租车释放{control['avg_taxi_release']:.1%}，"
                f"网约车释放{control['avg_ride_hailing_release']:.1%}，"
                f"私家车分流{control['avg_private_diversion']:.1%}，"
                f"加班巴士{final['extra_buses']:.1f}辆（等效量）",
                "",
                "| 方案 | 旅客平均等待 | 驾驶员平均等待 | 旅客95%等待 | "
                "驾驶员95%等待 | 公平指数 | 机会差 | 服务率 | 综合成本 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, row in data.iterrows():
            lines.append(
                f"| {POLICY_CN[policy]} | {row['avg_passenger_wait_min']:.2f} | "
                f"{row['avg_driver_wait_min']:.2f} | {row['p95_passenger_wait_min']:.1f} | "
                f"{row['p95_driver_wait_min']:.1f} | {row['driver_fairness_jain']:.3f} | "
                f"{row['service_opportunity_gap']:.3f} | {row['service_rate']:.2%} | "
                f"{row['generalized_cost']:.0f} |"
            )
        lines.append("")
    (OUTPUT / "problem2_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    validate_configs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, logs = [], {}
    policies = ("baseline", "static", "dynamic_efficiency", "dynamic_balanced")

    for city in CITIES:
        demand, supply = generate_inputs(city)
        for policy in policies:
            sim, summary = run_policy(city, demand, supply, policy)
            log = pd.DataFrame(sim.logs)
            log.to_csv(OUTPUT / f"{city}_{policy}_simulation.csv", index=False)
            summaries.append(summary)
            logs[(city, policy)] = log

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "policy_summary.csv", index=False, encoding="utf-8-sig")
    controls = summarize_controls(logs)
    controls.to_csv(OUTPUT / "control_summary.csv", index=False, encoding="utf-8-sig")
    for city in CITIES:
        plot_comparison(city, summary, logs)
    write_report(summary, controls)

    display = summary.copy()
    display["city"] = display["city"].map(CN_CITY)
    display["policy"] = display["policy"].map(POLICY_CN)
    print(
        display[
            [
                "city",
                "policy",
                "avg_passenger_wait_min",
                "avg_driver_wait_min",
                "driver_fairness_jain",
                "service_rate",
                "generalized_cost",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(f"\n结果已写入：{OUTPUT}")


if __name__ == "__main__":
    main()
