from __future__ import annotations

import io
import math
import random
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


st.set_page_config(
    page_title="Practical Elevator Traffic Tool",
    page_icon="🛗",
    layout="wide",
)


@dataclass
class LiftBankInput:
    bank_name: str
    building_type: str
    floors_served: int
    population_served: int
    number_of_lifts: int
    car_capacity_persons: int
    rated_speed_mps: float
    floor_height_m: float
    door_time_s: float
    passenger_transfer_time_s: float
    acceleration_mps2: float
    jerk_mps3: float
    main_terminal_floor: int = 0

    passenger_lift_pit_depth_m: float = 2.0
    passenger_lift_overhead_m: float = 4.5

    service_lift_pit_depth_m: float = 2.2
    service_lift_overhead_m: float = 4.8

    fireman_lift_pit_depth_m: float = 3.5
    fireman_lift_overhead_m: float = 5.0
    fireman_lift_car_width_mm: int = 1400
    fireman_lift_car_depth_mm: int = 2200
    fireman_lift_door_clear_mm: int = 1100


@dataclass
class Passenger:
    pid: int
    arrival: float
    origin: int
    destination: int
    direction: int
    assigned_car: int | None = None
    board_time: float | None = None
    exit_time: float | None = None
    wait_time: float | None = None
    journey_time: float | None = None


@dataclass(order=True)
class Event:
    time: float
    event_type: str = field(compare=False)
    passenger: Any = field(default=None, compare=False)


BUILDING_TYPES = ["Office", "Residential", "Mixed"]

DEFAULT_BANKS = pd.DataFrame([
    {
        "bank_name": "Tower A Passenger Lifts",
        "building_type": "Office",
        "floors_served": 33,
        "population_served": 963,
        "number_of_lifts": 4,
        "car_capacity_persons": 21,
        "rated_speed_mps": 4.0,
        "floor_height_m": 3.4,
        "door_time_s": 10.0,
        "passenger_transfer_time_s": 1.2,
        "acceleration_mps2": 1.0,
        "jerk_mps3": 1.5,
        "main_terminal_floor": 0,
        "passenger_lift_pit_depth_m": 2.0,
        "passenger_lift_overhead_m": 4.5,
        "service_lift_pit_depth_m": 2.2,
        "service_lift_overhead_m": 4.8,
        "fireman_lift_pit_depth_m": 3.5,
        "fireman_lift_overhead_m": 5.0,
        "fireman_lift_car_width_mm": 1400,
        "fireman_lift_car_depth_mm": 2200,
        "fireman_lift_door_clear_mm": 1100,
    },
    {
        "bank_name": "Tower B Passenger Lifts",
        "building_type": "Office",
        "floors_served": 37,
        "population_served": 1077,
        "number_of_lifts": 4,
        "car_capacity_persons": 21,
        "rated_speed_mps": 4.0,
        "floor_height_m": 3.4,
        "door_time_s": 10.0,
        "passenger_transfer_time_s": 1.2,
        "acceleration_mps2": 1.0,
        "jerk_mps3": 1.5,
        "main_terminal_floor": 0,
        "passenger_lift_pit_depth_m": 2.0,
        "passenger_lift_overhead_m": 4.5,
        "service_lift_pit_depth_m": 2.2,
        "service_lift_overhead_m": 4.8,
        "fireman_lift_pit_depth_m": 3.5,
        "fireman_lift_overhead_m": 5.0,
        "fireman_lift_car_width_mm": 1400,
        "fireman_lift_car_depth_mm": 2200,
        "fireman_lift_door_clear_mm": 1100,
    },
])

TRAFFIC_PROFILES: Dict[str, Dict[str, float | str]] = {
    "Office Morning Up-Peak": {
        "arrival_rate_per_sec": 0.25,
        "incoming": 0.85,
        "outgoing": 0.05,
        "interfloor": 0.10,
        "target_interval_s": 30.0,
        "target_hc_percent": 12.0,
        "applicable_to": "Office",
        "description": "Morning office traffic from lobby to upper floors.",
    },
    "Office Lunch / Two-Way": {
        "arrival_rate_per_sec": 0.30,
        "incoming": 0.40,
        "outgoing": 0.40,
        "interfloor": 0.20,
        "target_interval_s": 35.0,
        "target_hc_percent": 11.0,
        "applicable_to": "Office",
        "description": "Lunch traffic with mixed up, down and inter-floor movement.",
    },
    "Residential Morning Down-Peak": {
        "arrival_rate_per_sec": 0.18,
        "incoming": 0.20,
        "outgoing": 0.65,
        "interfloor": 0.15,
        "target_interval_s": 45.0,
        "target_hc_percent": 7.5,
        "applicable_to": "Residential",
        "description": "Residential morning traffic from apartments to lobby/parking.",
    },
    "Residential Evening Up-Peak": {
        "arrival_rate_per_sec": 0.18,
        "incoming": 0.60,
        "outgoing": 0.20,
        "interfloor": 0.20,
        "target_interval_s": 45.0,
        "target_hc_percent": 7.5,
        "applicable_to": "Residential",
        "description": "Residential evening return traffic from lobby/parking to apartments.",
    },
    "Mixed-Use Balanced": {
        "arrival_rate_per_sec": 0.28,
        "incoming": 0.45,
        "outgoing": 0.35,
        "interfloor": 0.20,
        "target_interval_s": 40.0,
        "target_hc_percent": 9.0,
        "applicable_to": "Mixed",
        "description": "Mixed-use profile with balanced up/down and inter-floor movement.",
    },
}


def scenario_is_applicable(building_type: str, scenario_name: str) -> bool:
    scenario_type = str(TRAFFIC_PROFILES[scenario_name]["applicable_to"])
    if building_type == "Mixed":
        return True
    return scenario_type == building_type or scenario_type == "Mixed"


def calculate_flight_time(distance_m: float, v_max: float, acceleration: float, jerk: float) -> float:
    if distance_m <= 0:
        return 0.0
    v_max = max(v_max, 0.1)
    acceleration = max(acceleration, 0.1)
    jerk = max(jerk, 0.1)
    distance_to_reach_speed = (v_max ** 2 / acceleration) + (v_max * (acceleration / jerk))
    if distance_m >= distance_to_reach_speed:
        return (distance_m / v_max) + (v_max / acceleration) + (acceleration / jerk)
    return 2 * math.sqrt(distance_m / acceleration) + (acceleration / jerk)


def estimated_min_overhead_m(speed_mps: float) -> float:
    buffer_stroke_m = speed_mps ** 2 / (2 * 9.81)
    return round((4200 + buffer_stroke_m * 1000 + 700) / 1000, 2)


def clone_bank(bank: LiftBankInput, **changes) -> LiftBankInput:
    data = bank.__dict__.copy()
    data.update(changes)
    return LiftBankInput(**data)


def run_practical_traffic_check(bank: LiftBankInput, control_method: str, scenario_name: str) -> Dict[str, float | str]:
    profile = TRAFFIC_PROFILES[scenario_name]
    passenger_load = max(2.0, bank.car_capacity_persons * 0.80)
    floors = max(1, bank.floors_served - 1)

    incoming = float(profile["incoming"])
    outgoing = float(profile["outgoing"])
    interfloor = float(profile["interfloor"])

    profile_pressure = 1.0
    if incoming >= 0.80 or outgoing >= 0.60:
        profile_pressure = 1.08
    elif interfloor >= 0.20:
        profile_pressure = 1.12

    probable_stops = floors * (1 - (1 - 1 / floors) ** passenger_load)
    highest_reversal_floor = floors - sum((i / floors) ** passenger_load for i in range(1, floors))

    single_floor_time = calculate_flight_time(
        bank.floor_height_m,
        bank.rated_speed_mps,
        bank.acceleration_mps2,
        bank.jerk_mps3,
    )

    zoning_factor = 1.0
    if floors > 35:
        zoning_factor = 0.92
    if floors > 50:
        zoning_factor = 0.86

    rtt = (
        2 * highest_reversal_floor * single_floor_time
        + probable_stops * bank.door_time_s
        + 2 * passenger_load * bank.passenger_transfer_time_s
    ) * profile_pressure * zoning_factor

    if control_method == "Destination Control":
        rtt *= 0.88
        awt_factor = 0.28
    else:
        awt_factor = 0.33

    interval = rtt / max(1, bank.number_of_lifts)
    five_min_capacity = (300 * passenger_load * bank.number_of_lifts) / max(1.0, rtt)
    hc_percent = (five_min_capacity / max(1, bank.population_served)) * 100
    awt = interval * awt_factor

    return {
        "Lift Bank": bank.bank_name,
        "Building Type": bank.building_type,
        "Traffic Scenario": scenario_name,
        "Control": control_method,
        "RTT (s)": round(rtt, 1),
        "Interval (s)": round(interval, 1),
        "AWT Approx. (s)": round(awt, 1),
        "Probable Stops": round(probable_stops, 1),
        "5-min HC (persons)": round(five_min_capacity, 0),
        "5-min HC (%)": round(hc_percent, 2),
    }


def pass_fail(row: Dict[str, float | str], profile: Dict[str, float | str]) -> str:
    interval = float(row["Interval (s)"])
    hc = float(row["5-min HC (%)"])
    target_interval = float(profile["target_interval_s"])
    target_hc = float(profile["target_hc_percent"])
    return "PASS" if interval <= target_interval and hc >= target_hc else "FAIL"


def find_solution_for_scenario(bank: LiftBankInput, scenario_name: str) -> Dict[str, str | float | int]:
    profile = TRAFFIC_PROFILES[scenario_name]
    current = run_practical_traffic_check(bank, "Conventional", scenario_name)
    if pass_fail(current, profile) == "PASS":
        return {
            "Lift Bank": bank.bank_name,
            "Traffic Scenario": scenario_name,
            "Current Result": "PASS",
            "Minimum Practical Solution": "No change required.",
            "Recommended Lifts": bank.number_of_lifts,
            "Recommended Capacity": bank.car_capacity_persons,
            "Recommended Speed": bank.rated_speed_mps,
            "Recommended Control": "Conventional",
            "Recommended Zoning": "Current zoning acceptable",
            "Result After Recommendation": "PASS",
        }

    capacity_options = sorted(set([bank.car_capacity_persons, 13, 16, 20, 21, 24, 26, 33, 40]))
    speed_options = sorted(set([bank.rated_speed_mps, 1.75, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]))
    lift_options = list(range(bank.number_of_lifts, min(bank.number_of_lifts + 8, 16) + 1))
    control_options = ["Conventional", "Destination Control"]

    # Practical priority order:
    # 1 destination control
    # 2 add lifts
    # 3 increase capacity
    # 4 increase speed
    # 5 combined optimized search

    candidates = []

    def add_candidate(test_bank: LiftBankInput, control: str, label: str):
        result = run_practical_traffic_check(test_bank, control, scenario_name)
        if pass_fail(result, profile) == "PASS":
            added_lifts = test_bank.number_of_lifts - bank.number_of_lifts
            added_capacity = test_bank.car_capacity_persons - bank.car_capacity_persons
            added_speed = test_bank.rated_speed_mps - bank.rated_speed_mps
            score = (
                added_lifts * 100
                + max(0, added_capacity) * 8
                + max(0, added_speed) * 15
                + (8 if control == "Destination Control" else 0)
            )
            candidates.append((score, test_bank, control, label, result))

    # Try only control system change
    add_candidate(bank, "Destination Control", "Use destination control system.")

    # Try adding lifts only
    for lifts in lift_options:
        add_candidate(clone_bank(bank, number_of_lifts=lifts), "Conventional", f"Increase number of lifts to {lifts}.")

    # Try increasing capacity only
    for cap in capacity_options:
        if cap >= bank.car_capacity_persons:
            add_candidate(clone_bank(bank, car_capacity_persons=cap), "Conventional", f"Increase car capacity to {cap} persons.")

    # Try increasing speed only
    for spd in speed_options:
        if spd >= bank.rated_speed_mps:
            add_candidate(clone_bank(bank, rated_speed_mps=spd), "Conventional", f"Increase speed to {spd} m/s.")

    # Try combined realistic combinations
    for lifts in lift_options:
        for cap in capacity_options:
            if cap < bank.car_capacity_persons:
                continue
            for spd in speed_options:
                if spd < bank.rated_speed_mps:
                    continue
                for control in control_options:
                    test_bank = clone_bank(
                        bank,
                        number_of_lifts=lifts,
                        car_capacity_persons=cap,
                        rated_speed_mps=spd,
                    )
                    zoning_note = "Apply zoning/sectoring" if test_bank.floors_served > 35 else "Current zoning acceptable"
                    label = (
                        f"Use {lifts} lifts, {cap} persons capacity, {spd} m/s speed, "
                        f"{control}, {zoning_note}."
                    )
                    add_candidate(test_bank, control, label)

    if not candidates:
        return {
            "Lift Bank": bank.bank_name,
            "Traffic Scenario": scenario_name,
            "Current Result": "FAIL",
            "Minimum Practical Solution": "No solution found within practical search range. Consider re-zoning, separate low/high rise banks, or specialist traffic study.",
            "Recommended Lifts": "Review",
            "Recommended Capacity": "Review",
            "Recommended Speed": "Review",
            "Recommended Control": "Review",
            "Recommended Zoning": "Separate zoning required",
            "Result After Recommendation": "FAIL",
        }

    candidates.sort(key=lambda x: x[0])
    _, best_bank, best_control, label, best_result = candidates[0]

    zoning = "Apply zoning/sectoring" if best_bank.floors_served > 35 else "Current zoning acceptable"

    return {
        "Lift Bank": bank.bank_name,
        "Traffic Scenario": scenario_name,
        "Current Result": "FAIL",
        "Minimum Practical Solution": label,
        "Recommended Lifts": best_bank.number_of_lifts,
        "Recommended Capacity": best_bank.car_capacity_persons,
        "Recommended Speed": best_bank.rated_speed_mps,
        "Recommended Control": best_control,
        "Recommended Zoning": zoning,
        "Result After Recommendation": pass_fail(best_result, profile),
    }


def final_recommendations_for_bank(bank: LiftBankInput) -> pd.DataFrame:
    rows = []
    for scenario_name in TRAFFIC_PROFILES:
        if scenario_is_applicable(bank.building_type, scenario_name):
            rows.append(find_solution_for_scenario(bank, scenario_name))
    return pd.DataFrame(rows)


def fire_lift_practical_check(bank: LiftBankInput) -> Dict[str, str | int | float]:
    stretcher_depth_ok = bank.fireman_lift_car_depth_mm >= 2134
    car_width_ok = bank.fireman_lift_car_width_mm >= 1400
    door_ok = bank.fireman_lift_door_clear_mm >= 1100
    pit_ok = bank.fireman_lift_pit_depth_m >= 3.5
    min_oh = estimated_min_overhead_m(bank.rated_speed_mps)
    oh_ok = bank.fireman_lift_overhead_m >= min_oh
    status = "PASS" if all([stretcher_depth_ok, car_width_ok, door_ok, pit_ok, oh_ok]) else "FAIL"

    return {
        "Lift Bank": bank.bank_name,
        "Fireman Lift Car W x D (mm)": f"{bank.fireman_lift_car_width_mm} x {bank.fireman_lift_car_depth_mm}",
        "Door Clear Opening (mm)": bank.fireman_lift_door_clear_mm,
        "Fireman Pit Depth (m)": bank.fireman_lift_pit_depth_m,
        "Fireman OH Provided (m)": bank.fireman_lift_overhead_m,
        "Estimated Min OH (m)": min_oh,
        "Width Check": "PASS" if car_width_ok else "FAIL",
        "Stretcher Depth Check": "PASS" if stretcher_depth_ok else "FAIL",
        "Door Check": "PASS" if door_ok else "FAIL",
        "Pit Check": "PASS" if pit_ok else "FAIL",
        "OH Check": "PASS" if oh_ok else "FAIL",
        "Final Result": status,
    }


def shaft_practical_check(bank: LiftBankInput) -> Dict[str, str | int | float]:
    estimated_car_width = 900 + bank.car_capacity_persons * 45
    estimated_car_depth = 1100 + bank.car_capacity_persons * 55
    estimated_single_shaft_width = estimated_car_width + 900
    estimated_total_core_width = estimated_single_shaft_width * bank.number_of_lifts + 600
    estimated_clear_depth = estimated_car_depth + 900
    min_oh = estimated_min_overhead_m(bank.rated_speed_mps)

    passenger_pit_status = "PASS" if bank.passenger_lift_pit_depth_m >= 1.6 else "FAIL"
    passenger_oh_status = "PASS" if bank.passenger_lift_overhead_m >= min_oh else "FAIL"

    service_pit_status = "PASS" if bank.service_lift_pit_depth_m >= 1.8 else "FAIL"
    service_oh_status = "PASS" if bank.service_lift_overhead_m >= min_oh else "FAIL"

    fireman_pit_status = "PASS" if bank.fireman_lift_pit_depth_m >= 3.5 else "FAIL"
    fireman_oh_status = "PASS" if bank.fireman_lift_overhead_m >= min_oh else "FAIL"

    final = "PASS" if all([
        passenger_pit_status == "PASS",
        passenger_oh_status == "PASS",
        service_pit_status == "PASS",
        service_oh_status == "PASS",
        fireman_pit_status == "PASS",
        fireman_oh_status == "PASS",
    ]) else "FAIL"

    return {
        "Lift Bank": bank.bank_name,
        "Estimated Car W x D (mm)": f"{int(estimated_car_width)} x {int(estimated_car_depth)}",
        "Estimated Bank Width (mm)": int(estimated_total_core_width),
        "Estimated Shaft Depth (mm)": int(estimated_clear_depth),
        "Estimated Min OH (m)": min_oh,
        "Passenger Pit Depth (m)": bank.passenger_lift_pit_depth_m,
        "Passenger OH Height (m)": bank.passenger_lift_overhead_m,
        "Passenger Pit Result": passenger_pit_status,
        "Passenger OH Result": passenger_oh_status,
        "Service Lift Pit Depth (m)": bank.service_lift_pit_depth_m,
        "Service Lift OH Height (m)": bank.service_lift_overhead_m,
        "Service Pit Result": service_pit_status,
        "Service OH Result": service_oh_status,
        "Fireman Lift Pit Depth (m)": bank.fireman_lift_pit_depth_m,
        "Fireman Lift OH Height (m)": bank.fireman_lift_overhead_m,
        "Fireman Pit Result": fireman_pit_status,
        "Fireman OH Result": fireman_oh_status,
        "Final Result": final,
    }


class PracticalElevatorSimulation:
    def __init__(self, bank: LiftBankInput, profile: Dict[str, float | str], horizon: int, seed: int = 42):
        self.bank = bank
        self.profile = profile
        self.horizon = horizon
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.events: List[Event] = []
        self.completed: List[Passenger] = []
        self.car_floors = [bank.main_terminal_floor] * max(1, bank.number_of_lifts)
        self.car_available_at = [0.0] * max(1, bank.number_of_lifts)

    def generate_passengers(self) -> None:
        rate = float(self.profile["arrival_rate_per_sec"])
        now = 0.0
        pid = 1
        floors = list(range(0, max(2, self.bank.floors_served)))

        while now < self.horizon:
            now += float(self.np_rng.exponential(1.0 / max(rate, 0.001)))
            if now >= self.horizon:
                break
            x = self.rng.random()
            incoming = float(self.profile["incoming"])
            outgoing = float(self.profile["outgoing"])
            if x < incoming:
                origin = self.bank.main_terminal_floor
                destination = self.rng.choice([f for f in floors if f != origin])
            elif x < incoming + outgoing:
                origin = self.rng.choice([f for f in floors if f != self.bank.main_terminal_floor])
                destination = self.bank.main_terminal_floor
            else:
                origin = self.rng.choice(floors)
                destination = self.rng.choice(floors)
                while destination == origin:
                    destination = self.rng.choice(floors)
            direction = 1 if destination > origin else -1
            heapq.heappush(self.events, Event(now, "PASSENGER", passenger=Passenger(pid, now, origin, destination, direction)))
            pid += 1

    def select_best_car(self, passenger: Passenger) -> int:
        best_car = 0
        best_eta = float("inf")
        for idx, current_floor in enumerate(self.car_floors):
            available_at = self.car_available_at[idx]
            travel_to_origin = calculate_flight_time(
                abs(passenger.origin - current_floor) * self.bank.floor_height_m,
                self.bank.rated_speed_mps,
                self.bank.acceleration_mps2,
                self.bank.jerk_mps3,
            )
            eta = max(passenger.arrival, available_at) + travel_to_origin
            if eta < best_eta:
                best_eta = eta
                best_car = idx
        return best_car

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.generate_passengers()
        while self.events:
            event = heapq.heappop(self.events)
            p: Passenger = event.passenger
            car = self.select_best_car(p)
            p.assigned_car = car + 1
            start_time = max(p.arrival, self.car_available_at[car])
            time_to_origin = calculate_flight_time(
                abs(p.origin - self.car_floors[car]) * self.bank.floor_height_m,
                self.bank.rated_speed_mps,
                self.bank.acceleration_mps2,
                self.bank.jerk_mps3,
            )
            board_time = start_time + time_to_origin + self.bank.door_time_s / 2
            trip_time = calculate_flight_time(
                abs(p.destination - p.origin) * self.bank.floor_height_m,
                self.bank.rated_speed_mps,
                self.bank.acceleration_mps2,
                self.bank.jerk_mps3,
            )
            exit_time = board_time + trip_time + self.bank.door_time_s / 2 + self.bank.passenger_transfer_time_s

            p.board_time = round(board_time, 2)
            p.exit_time = round(exit_time, 2)
            p.wait_time = round(board_time - p.arrival, 2)
            p.journey_time = round(exit_time - p.arrival, 2)
            self.car_floors[car] = p.destination
            self.car_available_at[car] = exit_time
            self.completed.append(p)

        passenger_df = pd.DataFrame([p.__dict__ for p in self.completed])
        if passenger_df.empty:
            return passenger_df, pd.DataFrame([{"Message": "No passengers generated. Increase horizon or arrival rate."}])
        summary_df = pd.DataFrame([{
            "Passengers Completed": len(passenger_df),
            "Mean Waiting Time (s)": round(passenger_df["wait_time"].mean(), 1),
            "90th Percentile Waiting Time (s)": round(passenger_df["wait_time"].quantile(0.9), 1),
            "Mean Journey Time (s)": round(passenger_df["journey_time"].mean(), 1),
            "Max Waiting Time (s)": round(passenger_df["wait_time"].max(), 1),
        }])
        return passenger_df, summary_df


def dataframe_to_pdf_table(df: pd.DataFrame, max_rows: int = 35):
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7, leading=8)
    header_style = ParagraphStyle("Header", parent=styles["Normal"], fontSize=7, leading=8, textColor=colors.white, fontName="Helvetica-Bold")
    clean_df = df.head(max_rows).copy().fillna("")
    header = [Paragraph(str(col), header_style) for col in clean_df.columns]
    rows = [[Paragraph(str(value), cell_style) for value in row] for row in clean_df.values.tolist()]
    col_width = 780 / max(1, len(clean_df.columns))
    table = Table([header] + rows, colWidths=[col_width] * len(clean_df.columns))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def create_pdf_report(project_name, prepared_by, input_df, results_df, recommendations_df, fire_df, shaft_df, simulation_summary_df=None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=25, leftMargin=25, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=15, textColor=colors.HexColor("#0F172A"), alignment=0)
    subtitle_style = ParagraphStyle("SubtitleStyle", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#475569"))
    section_style = ParagraphStyle("SectionStyle", parent=styles["Heading2"], fontSize=10, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=5)

    elements = [
        Paragraph("Practical Elevator Traffic Review Report", title_style),
        Paragraph(f"Project: {project_name}", subtitle_style),
        Paragraph(f"Prepared By: {prepared_by}", subtitle_style),
        Paragraph("Purpose: preliminary review, option comparison, and coordination support.", subtitle_style),
        Paragraph("Note: final traffic results, fire lift compliance and shaft sizes must be verified by the elevator specialist.", subtitle_style),
        Spacer(1, 8),
        Paragraph("1. Project Inputs", section_style),
        dataframe_to_pdf_table(input_df),
        Paragraph("2. Practical Traffic Results - PASS / FAIL", section_style),
        dataframe_to_pdf_table(results_df),
        Paragraph("3. Final Recommendations", section_style),
        dataframe_to_pdf_table(recommendations_df),
        Paragraph("4. Fireman Lift Preliminary Check", section_style),
        dataframe_to_pdf_table(fire_df),
        Paragraph("5. Separate Pit / Overhead Review", section_style),
        dataframe_to_pdf_table(shaft_df),
    ]
    if simulation_summary_df is not None and not simulation_summary_df.empty:
        elements.extend([
            Paragraph("6. Passenger Simulation Summary", section_style),
            dataframe_to_pdf_table(simulation_summary_df),
        ])
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def build_excel_report(input_df, results_df, recommendations_df, fire_df, shaft_df, simulation_summary_df, passenger_df) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="Inputs", index=False)
        results_df.to_excel(writer, sheet_name="PASS FAIL Results", index=False)
        recommendations_df.to_excel(writer, sheet_name="Final Recommendations", index=False)
        fire_df.to_excel(writer, sheet_name="Fireman Lift Check", index=False)
        shaft_df.to_excel(writer, sheet_name="Pit OH Separate Check", index=False)
        if simulation_summary_df is not None and not simulation_summary_df.empty:
            simulation_summary_df.to_excel(writer, sheet_name="Simulation Summary", index=False)
        if passenger_df is not None and not passenger_df.empty:
            passenger_df.head(5000).to_excel(writer, sheet_name="Passenger Logs", index=False)
    buffer.seek(0)
    return buffer.getvalue()


st.title("🛗 Practical Elevator Traffic Review Tool")
st.caption("Project-oriented tool for quick lift traffic review, shaft coordination, fireman lift checks, recommendations, and passenger simulation.")

with st.sidebar:
    st.header("Project")
    project_name = st.text_input("Project Name", "Radiant Tower")
    prepared_by = st.text_input("Prepared By", "ATGC Engineering")
    st.header("Simulation")
    sim_scenario_name = st.selectbox("Simulation Scenario", list(TRAFFIC_PROFILES.keys()))
    st.info(str(TRAFFIC_PROFILES[sim_scenario_name]["description"]))
    sim_horizon = st.number_input("Simulation Duration (seconds)", min_value=300, max_value=7200, value=1200, step=300)
    seed = st.number_input("Random Seed", min_value=1, max_value=999999, value=42, step=1)

st.markdown("## 1. Lift Bank Inputs")
st.write("Building type is selected from Office, Residential, or Mixed. Pit depth and overhead are separated for passenger, service, and fireman lifts.")

column_config = {
    "building_type": st.column_config.SelectboxColumn(
        "building_type",
        help="Select building type",
        options=BUILDING_TYPES,
        required=True,
    )
}

input_df = st.data_editor(DEFAULT_BANKS, num_rows="dynamic", use_container_width=True, hide_index=True, column_config=column_config)
input_df = input_df.dropna(subset=["bank_name"]).copy()
input_df["building_type"] = input_df["building_type"].where(input_df["building_type"].isin(BUILDING_TYPES), "Office")

int_cols = [
    "floors_served",
    "population_served",
    "number_of_lifts",
    "car_capacity_persons",
    "main_terminal_floor",
    "fireman_lift_car_width_mm",
    "fireman_lift_car_depth_mm",
    "fireman_lift_door_clear_mm",
]
float_cols = [
    "rated_speed_mps",
    "floor_height_m",
    "door_time_s",
    "passenger_transfer_time_s",
    "acceleration_mps2",
    "jerk_mps3",
    "passenger_lift_pit_depth_m",
    "passenger_lift_overhead_m",
    "service_lift_pit_depth_m",
    "service_lift_overhead_m",
    "fireman_lift_pit_depth_m",
    "fireman_lift_overhead_m",
]
for col in int_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0).astype(int)
for col in float_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0.0).astype(float)

banks = [LiftBankInput(**row) for row in input_df.to_dict(orient="records")]
if not banks:
    st.error("Please enter at least one lift bank.")
    st.stop()

st.markdown("## 2. Practical Traffic Results - PASS / FAIL")
result_rows = []
for bank in banks:
    for scenario_name, scenario_profile in TRAFFIC_PROFILES.items():
        if not scenario_is_applicable(bank.building_type, scenario_name):
            continue
        for control in ["Conventional", "Destination Control"]:
            row = run_practical_traffic_check(bank, control, scenario_name)
            row["Result"] = pass_fail(row, scenario_profile)
            row["Target Interval (s)"] = scenario_profile["target_interval_s"]
            row["Target HC (%)"] = scenario_profile["target_hc_percent"]
            result_rows.append(row)

results_df = pd.DataFrame(result_rows)
st.dataframe(results_df, use_container_width=True, hide_index=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Lift Banks", len(banks))
k2.metric("Total Lifts", int(input_df["number_of_lifts"].sum()))
k3.metric("Passed Rows", int((results_df["Result"] == "PASS").sum()))
k4.metric("Failed Rows", int((results_df["Result"] == "FAIL").sum()))

st.markdown("## 3. Final Recommendation - How to Solve Traffic")
recommendation_frames = [final_recommendations_for_bank(bank) for bank in banks]
recommendations_df = pd.concat(recommendation_frames, ignore_index=True) if recommendation_frames else pd.DataFrame()
st.dataframe(recommendations_df, use_container_width=True, hide_index=True)

st.markdown("## 4. Fireman Lift Practical Check")
fire_df = pd.DataFrame([fire_lift_practical_check(bank) for bank in banks])
st.dataframe(fire_df, use_container_width=True, hide_index=True)

st.markdown("## 5. Separate Pit Depth / Overhead Height Check")
shaft_df = pd.DataFrame([shaft_practical_check(bank) for bank in banks])
st.dataframe(shaft_df, use_container_width=True, hide_index=True)

st.markdown("## 6. Passenger Simulation")
selected_bank_name = st.selectbox("Select lift bank for simulation", input_df["bank_name"].tolist())
selected_bank = next(bank for bank in banks if bank.bank_name == selected_bank_name)

if st.button("Run Practical Simulation", type="primary"):
    simulator = PracticalElevatorSimulation(selected_bank, TRAFFIC_PROFILES[sim_scenario_name], horizon=int(sim_horizon), seed=int(seed))
    passenger_df, simulation_summary_df = simulator.run()
    st.session_state["passenger_df"] = passenger_df
    st.session_state["simulation_summary_df"] = simulation_summary_df

if "simulation_summary_df" in st.session_state:
    simulation_summary_df = st.session_state["simulation_summary_df"]
    passenger_df = st.session_state["passenger_df"]
    st.subheader("Simulation Summary")
    st.dataframe(simulation_summary_df, use_container_width=True, hide_index=True)
    if not passenger_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Waiting Time Sample")
            st.bar_chart(passenger_df["wait_time"].head(300))
        with c2:
            st.write("Destination Distribution")
            dest_counts = passenger_df.groupby("destination")["pid"].count().reset_index(name="passengers")
            st.bar_chart(dest_counts, x="destination", y="passengers")
        st.subheader("Passenger Log")
        st.dataframe(passenger_df.head(500), use_container_width=True, hide_index=True)
else:
    simulation_summary_df = None
    passenger_df = None

st.markdown("## 7. Export")

pdf_bytes = create_pdf_report(project_name, prepared_by, input_df, results_df, recommendations_df, fire_df, shaft_df, simulation_summary_df)
excel_bytes = build_excel_report(input_df, results_df, recommendations_df, fire_df, shaft_df, simulation_summary_df, passenger_df)

e1, e2, e3 = st.columns(3)
with e1:
    st.download_button("Download PDF", data=pdf_bytes, file_name="practical_elevator_traffic_review.pdf", mime="application/pdf", use_container_width=True)
with e2:
    st.download_button("Download Excel", data=excel_bytes, file_name="practical_elevator_traffic_review.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with e3:
    st.download_button("Download CSV", data=results_df.to_csv(index=False).encode("utf-8-sig"), file_name="practical_traffic_results_pass_fail.csv", mime="text/csv", use_container_width=True)

st.warning(
    "Practical note: Use this tool for early-stage review, project coordination and option comparison. "
    "Final lift selection, fireman lift compliance, shaft dimensions and traffic analysis shall be confirmed by the elevator specialist."
)
