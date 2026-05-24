from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image

# Set up page configuration
st.set_page_config(page_title="VT Engineering Review Platform", page_icon="🛗", layout="wide")

@dataclass
class LiftBankInput:
    bank_name: str
    building_type: str
    building_grade: str
    floors_served: int
    total_travel_height_m: float
    population_served: int
    population_per_floor: int
    number_of_lifts: int
    car_capacity_persons: int
    rated_speed_mps: float
    floor_height_m: float
    door_type: str
    door_clear_width_mm: int
    door_time_s: float
    passenger_transfer_time_s: float
    acceleration_mps2: float
    jerk_mps3: float
    main_terminal_floor: int = 0
    sky_lobby_floor: int = 0
    amenity_floor_numbers: str = ""
    zoning_strategy: str = "Single Zone"
    passenger_lift_pit_depth_m: float = 2.0
    passenger_lift_overhead_m: float = 4.5
    service_lift_pit_depth_m: float = 2.2
    service_lift_overhead_m: float = 4.8
    fireman_lift_pit_depth_m: float = 3.5
    fireman_lift_overhead_m: float = 5.0
    fireman_lift_car_width_mm: int = 1400
    fireman_lift_car_depth_mm: int = 2200
    fireman_lift_door_clear_mm: int = 1100

BUILDING_TYPES = ["Office", "Residential", "Hotel", "Hospital", "Mixed"]
BUILDING_GRADES = ["Prestige / Corporate Office", "Mainstream / Speculative Office", "Luxury Residential", "Standard Residential", "Hotel 4-5 Star", "Hospital", "Mixed Use"]
DOOR_TYPES = ["Center Opening", "Side Opening", "Telescopic", "Other"]
ZONING_OPTIONS = ["Single Zone", "Low / High Rise Split", "Express / Sky Lobby", "Separate Service Zone", "Mixed-Use Separated Banks"]

DEFAULT_BANKS = pd.DataFrame([{
    "bank_name": "Tower A Passenger Lifts", "building_type": "Office", "building_grade": "Mainstream / Speculative Office",
    "floors_served": 33, "total_travel_height_m": 108.8, "population_served": 963, "population_per_floor": 30,
    "number_of_lifts": 4, "car_capacity_persons": 21, "rated_speed_mps": 4.0, "floor_height_m": 3.4,
    "door_type": "Center Opening", "door_clear_width_mm": 1100, "door_time_s": 10.0, "passenger_transfer_time_s": 1.2,
    "acceleration_mps2": 1.0, "jerk_mps3": 1.5, "main_terminal_floor": 0, "sky_lobby_floor": 0,
    "amenity_floor_numbers": "", "zoning_strategy": "Single Zone", "passenger_lift_pit_depth_m": 2.0,
    "passenger_lift_overhead_m": 4.5, "service_lift_pit_depth_m": 2.2, "service_lift_overhead_m": 4.8,
    "fireman_lift_pit_depth_m": 3.5, "fireman_lift_overhead_m": 5.0, "fireman_lift_car_width_mm": 1400,
    "fireman_lift_car_depth_mm": 2200, "fireman_lift_door_clear_mm": 1100,
}])

TRAFFIC_PROFILES = {
    "Office Morning Up-Peak": {"incoming": 0.85, "outgoing": 0.10, "interfloor": 0.05, "applicable_to": "Office"},
    "Office Lunch / Two-Way": {"incoming": 0.40, "outgoing": 0.40, "interfloor": 0.20, "applicable_to": "Office"},
    "Residential Morning Down-Peak": {"incoming": 0.20, "outgoing": 0.65, "interfloor": 0.15, "applicable_to": "Residential"},
    "Residential Evening Up-Peak": {"incoming": 0.60, "outgoing": 0.20, "interfloor": 0.20, "applicable_to": "Residential"},
    "Hotel Two-Way Guest Movement": {"incoming": 0.45, "outgoing": 0.35, "interfloor": 0.20, "applicable_to": "Hotel"},
    "Mixed-Use Balanced": {"incoming": 0.45, "outgoing": 0.35, "interfloor": 0.20, "applicable_to": "Mixed"},
}

def benchmark_for(bank: LiftBankInput) -> Dict[str, float | str]:
    grade = bank.building_grade
    table = {
        "Prestige / Corporate Office": (25, 30, 12, 15, 90, 120),
        "Mainstream / Speculative Office": (30, 40, 11, 13, 90, 120),
        "Luxury Residential": (45, 45, 6, 7, 120, 120),
        "Standard Residential": (60, 60, 5, 7, 120, 120),
        "Hotel 4-5 Star": (30, 35, 10, 12, 120, 120),
        "Hospital": (35, 45, 10, 12, 120, 150),
        "Mixed Use": (40, 45, 9, 11, 120, 130),
    }
    awt_ex, awt_acc, hc_min, hc_target, attd_ideal, attd_max = table.get(grade, table["Mixed Use"])
    return {"awt_excellent": awt_ex, "awt_acceptable": awt_acc, "hc_min": hc_min, "hc_target": hc_target, "attd_ideal": attd_ideal, "attd_max": attd_max, "source": "CIBSE Guide D / ISO 8100-32 benchmark basis"}

def scenario_is_applicable(building_type: str, scenario_name: str) -> bool:
    applicable = TRAFFIC_PROFILES[scenario_name]["applicable_to"]
    if building_type == "Mixed": return True
    if building_type == "Hospital": return scenario_name in ["Hotel Two-Way Guest Movement", "Mixed-Use Balanced"]
    return applicable == building_type or applicable == "Mixed"

def clone_bank(bank: LiftBankInput, **changes) -> LiftBankInput:
    data = bank.__dict__.copy(); data.update(changes); return LiftBankInput(**data)

def calculate_flight_time(distance_m: float, v_max: float, acceleration: float, jerk: float) -> float:
    if distance_m <= 0: return 0.0
    v_max, acceleration, jerk = max(v_max, .1), max(acceleration, .1), max(jerk, .1)
    distance_to_reach_speed = (v_max ** 2 / acceleration) + (v_max * (acceleration / jerk))
    if distance_m >= distance_to_reach_speed:
        return (distance_m / v_max) + (v_max / acceleration) + (acceleration / jerk)
    return 2 * math.sqrt(distance_m / acceleration) + (acceleration / jerk)

def control_factor(control_method: str) -> Tuple[float, float]:
    return {"Conventional": (1.00, .33), "Hybrid": (.93, .30), "DCS": (.88, .28)}.get(control_method, (1.00, .33))

def door_efficiency_factor(bank: LiftBankInput) -> float:
    factor = 1.0
    if bank.door_clear_width_mm < 900: factor += .10
    elif bank.door_clear_width_mm < 1000: factor += .05
    elif bank.door_clear_width_mm >= 1100: factor -= .03
    if bank.door_type == "Side Opening": factor += .04
    elif bank.door_type == "Telescopic": factor += .06
    return max(.90, factor)

def zoning_efficiency_factor(bank: LiftBankInput) -> float:
    if bank.zoning_strategy == "Single Zone":
        if bank.floors_served > 50: return 1.08
        if bank.floors_served > 35: return 1.04
        return 1.00
    return {"Low / High Rise Split": .92, "Express / Sky Lobby": .88, "Mixed-Use Separated Banks": .90, "Separate Service Zone": .95}.get(bank.zoning_strategy, 1.0)

def profile_pressure_factor(scenario_name: str) -> float:
    p = TRAFFIC_PROFILES[scenario_name]
    if float(p["incoming"]) >= .80 or float(p["outgoing"]) >= .60: return 1.08
    if float(p["interfloor"]) >= .20: return 1.12
    return 1.0

def run_traffic(bank: LiftBankInput, control_method: str, scenario_name: str) -> Dict[str, float | str]:
    passenger_load = max(2.0, bank.car_capacity_persons * .80)
    floors = max(1, bank.floors_served - 1)
    stops = floors * (1 - (1 - 1 / floors) ** passenger_load)
    highest_reversal = floors - sum((i / floors) ** passenger_load for i in range(1, floors))
    avg_floor_height = bank.total_travel_height_m / max(1, bank.floors_served - 1) if bank.total_travel_height_m > 0 and bank.floors_served > 1 else bank.floor_height_m
    tf = calculate_flight_time(avg_floor_height, bank.rated_speed_mps, bank.acceleration_mps2, bank.jerk_mps3)
    rtt_base = (2 * highest_reversal * tf + stops * bank.door_time_s * door_efficiency_factor(bank) + 2 * passenger_load * bank.passenger_transfer_time_s)
    rtt_base *= profile_pressure_factor(scenario_name) * zoning_efficiency_factor(bank)
    if bank.sky_lobby_floor > 0 and bank.zoning_strategy == "Express / Sky Lobby":
        rtt_base += calculate_flight_time(bank.sky_lobby_floor * avg_floor_height, bank.rated_speed_mps, bank.acceleration_mps2, bank.jerk_mps3)
    if str(bank.amenity_floor_numbers).strip(): rtt_base *= 1.04
    rtt_factor, awt_factor = control_factor(control_method)
    rtt = rtt_base * rtt_factor
    interval = rtt / max(1, bank.number_of_lifts)
    hc_pax = (300 * passenger_load * bank.number_of_lifts) / max(1.0, rtt)
    hc_pct = hc_pax / max(1, bank.population_served) * 100
    awt = interval * awt_factor
    avg_trip_distance = (bank.total_travel_height_m or (bank.floor_height_m * floors)) * .55
    trip_time = calculate_flight_time(avg_trip_distance, bank.rated_speed_mps, bank.acceleration_mps2, bank.jerk_mps3)
    attd = awt + trip_time + bank.door_time_s + bank.passenger_transfer_time_s * passenger_load * .35
    return {"RTT (s)": round(rtt, 1), "Interval (s)": round(interval, 1), "AWT (s)": round(awt, 1), "ATTD (s)": round(attd, 1), "5HC (%)": round(hc_pct, 2), "5HC (pax)": round(hc_pax, 0), "Car Loading Used (%)": 80}

def pass_fail(bank: LiftBankInput, result: Dict[str, float | str]) -> str:
    bm = benchmark_for(bank)
    return "PASS" if float(result["AWT (s)"]) <= float(bm["awt_acceptable"]) and float(result["5HC (%)"]) >= float(bm["hc_min"]) and float(result["ATTD (s)"]) <= float(bm["attd_max"]) else "FAIL"

def performance_comment(bank: LiftBankInput, result: Dict[str, float | str]) -> str:
    bm = benchmark_for(bank); comments = []
    if float(result["AWT (s)"]) > float(bm["awt_acceptable"]): comments.append("AWT exceeds benchmark.")
    if float(result["5HC (%)"]) < float(bm["hc_min"]): comments.append("5-minute handling capacity is below benchmark.")
    if float(result["ATTD (s)"]) > float(bm["attd_max"]): comments.append("Average time to destination exceeds benchmark.")
    if bank.door_clear_width_mm < 1000: comments.append("Door clear width may restrict passenger flow.")
    if bank.floors_served > 35 and bank.zoning_strategy == "Single Zone": comments.append("Consider zoning or split banks for this number of floors.")
    return "Acceptable for preliminary review." if not comments else " ".join(comments)

def practical_system_by_building(bank: LiftBankInput) -> str:
    floors, pop, btype = bank.floors_served, bank.population_served, bank.building_type
    if btype == "Office":
        if floors <= 20 and pop <= 700: return "Conventional"
        if floors <= 35 and pop <= 1200: return "Hybrid"
        return "DCS"
    if btype == "Residential":
        if floors <= 25 and pop <= 800: return "Conventional"
        if floors <= 45 and pop <= 1500: return "Hybrid"
        return "DCS"
    if btype == "Hotel": return "Hybrid" if floors <= 20 and pop <= 700 else "DCS"
    if btype == "Hospital": return "DCS"
    if btype == "Mixed": return "Hybrid" if floors <= 25 and pop <= 900 else "DCS"
    return "Hybrid"

def solve_recommendation(bank: LiftBankInput, scenario_name: str) -> Dict[str, str | int | float]:
    recommended_system = practical_system_by_building(bank)
    current = run_traffic(bank, recommended_system, scenario_name)
    if pass_fail(bank, current) == "PASS":
        return {"Lift Bank": bank.bank_name, "Scenario": scenario_name, "Result": "PASS", "Recommendation": f"Use {recommended_system}. Existing {bank.number_of_lifts} lifts, {bank.car_capacity_persons} persons, {bank.rated_speed_mps} m/s are acceptable."}
    lift_options = range(bank.number_of_lifts, min(bank.number_of_lifts + 8, 16) + 1)
    capacity_options = sorted(set([bank.car_capacity_persons, 13, 16, 20, 21, 24, 26, 33, 40]))
    speed_options = sorted(set([bank.rated_speed_mps, 1.75, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]))
    candidates = []
    for lifts in lift_options:
        for cap in capacity_options:
            if cap < bank.car_capacity_persons: continue
            for speed in speed_options:
                if speed < bank.rated_speed_mps: continue
                for control in ["Conventional", "Hybrid", "DCS"]:
                    for zoning in ZONING_OPTIONS:
                        tb = clone_bank(bank, number_of_lifts=lifts, car_capacity_persons=cap, rated_speed_mps=speed, zoning_strategy=zoning)
                        result = run_traffic(tb, control, scenario_name)
                        if pass_fail(tb, result) == "PASS":
                            score = (lifts-bank.number_of_lifts)*100 + (cap-bank.car_capacity_persons)*8 + (speed-bank.rated_speed_mps)*15 + (0 if control==recommended_system else 12) + (0 if zoning==bank.zoning_strategy else 20)
                            candidates.append((score, lifts, cap, speed, control, zoning))
    if not candidates:
        return {"Lift Bank": bank.bank_name, "Scenario": scenario_name, "Result": "FAIL", "Recommendation": "Traffic not solved within practical search range. Use separate zoning/sectoring, split low/high-rise banks, or request specialist VT traffic study."}
    _, lifts, cap, speed, control, zoning = sorted(candidates, key=lambda x: x[0])[0]
    return {"Lift Bank": bank.bank_name, "Scenario": scenario_name, "Result": "FAIL", "Recommendation": f"Use {control}: {lifts} lifts, {cap} persons, {speed} m/s, zoning: {zoning}."}

def build_analysis_rows(banks: List[LiftBankInput]) -> pd.DataFrame:
    rows=[]
    for bank in banks:
        for scenario in TRAFFIC_PROFILES:
            if not scenario_is_applicable(bank.building_type, scenario): continue
            for control in ["Conventional", "Hybrid", "DCS"]:
                result = run_traffic(bank, control, scenario); bm=benchmark_for(bank)
                rows.append({"Lift Bank": bank.bank_name, "Building Type": bank.building_type, "Grade": bank.building_grade, "Scenario": scenario, "Control": control, **result, "AWT Benchmark (s)": bm["awt_acceptable"], "5HC Min Benchmark (%)": bm["hc_min"], "ATTD Max Benchmark (s)": bm["attd_max"], "Result": pass_fail(bank, result), "Comment": performance_comment(bank, result)})
    return pd.DataFrame(rows)

def build_recommendation_rows(banks: List[LiftBankInput]) -> pd.DataFrame:
    return pd.DataFrame([solve_recommendation(bank, scenario) for bank in banks for scenario in TRAFFIC_PROFILES if scenario_is_applicable(bank.building_type, scenario)])

def build_benchmark_rows(banks: List[LiftBankInput]) -> pd.DataFrame:
    rows=[]
    for bank in banks:
        bm=benchmark_for(bank)
        rows.append({"Lift Bank": bank.bank_name, "Building Grade": bank.building_grade, "AWT Excellent (s)": bm["awt_excellent"], "AWT Acceptable (s)": bm["awt_acceptable"], "5HC Min (%)": bm["hc_min"], "5HC Target (%)": bm["hc_target"], "ATTD Ideal (s)": bm["attd_ideal"], "ATTD Max (s)": bm["attd_max"], "Car Loading Used": "80% of rated capacity", "Benchmark Basis": bm["source"]})
    return pd.DataFrame(rows)

def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["bank_name"]).copy()
    for col, default, opts in [("building_type","Office",BUILDING_TYPES),("building_grade","Mainstream / Speculative Office",BUILDING_GRADES),("door_type","Center Opening",DOOR_TYPES),("zoning_strategy","Single Zone",ZONING_OPTIONS)]:
        df[col] = df[col].fillna(default); df[col] = df[col].where(df[col].isin(opts), default)
    int_cols = ["floors_served","population_served","population_per_floor","number_of_lifts","car_capacity_persons","door_clear_width_mm","main_terminal_floor","sky_lobby_floor","fireman_lift_car_width_mm","fireman_lift_car_depth_mm","fireman_lift_door_clear_mm"]
    float_cols = ["total_travel_height_m","rated_speed_mps","floor_height_m","door_time_s","passenger_transfer_time_s","acceleration_mps2","jerk_mps3","passenger_lift_pit_depth_m","passenger_lift_overhead_m","service_lift_pit_depth_m","service_lift_overhead_m","fireman_lift_pit_depth_m","fireman_lift_overhead_m"]
    for col in int_cols: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    df["amenity_floor_numbers"] = df["amenity_floor_numbers"].fillna("").astype(str)
    return df

def dataframe_to_pdf_table(df: pd.DataFrame, max_rows: int = 45):
    styles=getSampleStyleSheet(); cell=ParagraphStyle("Cell", parent=styles["Normal"], fontSize=6.5, leading=8); head=ParagraphStyle("Header", parent=styles["Normal"], fontSize=6.5, leading=8, textColor=colors.white, fontName="Helvetica-Bold")
    clean=df.head(max_rows).copy().fillna("")
    data=[[Paragraph(str(c), head) for c in clean.columns]] + [[Paragraph(str(v), cell) for v in row] for row in clean.astype(str).values.tolist()]
    tbl=Table(data, colWidths=[780/max(1,len(clean.columns))]*len(clean.columns))
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1E3A8A")),("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#CBD5E1")),("PADDING",(0,0),(-1,-1),2),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    return tbl

def save_uploaded_image(image_bytes, image_name):
    if not image_bytes or not image_name: return None
    suffix=os.path.splitext(image_name)[1] or ".png"; tmp=tempfile.NamedTemporaryFile(delete=False, suffix=suffix); tmp.write(image_bytes); tmp.flush(); tmp.close(); return tmp.name

def create_pdf(project_name, project_address, prepared_by, logo_bytes, logo_name, photo_bytes, photo_name, input_df, analysis_df, rec_df, bm_df) -> bytes:
    buffer=io.BytesIO(); doc=SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=25, leftMargin=25, topMargin=30, bottomMargin=30)
    styles=getSampleStyleSheet(); title=ParagraphStyle("Title", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#0F172A"), alignment=1); sub=ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#475569"), alignment=1); sec=ParagraphStyle("Sec", parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=5)
    elements=[]; logo=save_uploaded_image(logo_bytes, logo_name); photo=save_uploaded_image(photo_bytes, photo_name)
    if logo: elements += [Image(logo, width=110, height=55), Spacer(1,10)]
    elements += [Paragraph("Vertical Transportation Engineering Review", title), Spacer(1,8), Paragraph(f"<b>Project:</b> {project_name}", sub), Paragraph(f"<b>Address:</b> {project_address or '-'}", sub), Paragraph(f"<b>Prepared By:</b> {prepared_by}", sub), Spacer(1,14)]
    if photo: elements += [Image(photo, width=360, height=190), Spacer(1,14)]
    elements += [Paragraph("Benchmark basis: CIBSE Guide D / ISO 8100-32 style target metrics used for preliminary comparison.", sub), Paragraph("Note: final traffic analysis must be verified by elevator specialist/manufacturer.", sub), Spacer(1,18), Paragraph("Tower & Lift Inputs", sec), dataframe_to_pdf_table(input_df), Paragraph("Benchmark Targets", sec), dataframe_to_pdf_table(bm_df), Paragraph("Result Recommendations", sec), dataframe_to_pdf_table(rec_df), Paragraph("Detailed Analysis", sec), dataframe_to_pdf_table(analysis_df)]
    doc.build(elements); buffer.seek(0); return buffer.getvalue()

def init_state():
    defaults={"page":1,"project_name":"Radiant Tower","project_address":"","prepared_by":"ATGC Engineering","logo_bytes":None,"logo_name":None,"project_photo_bytes":None,"project_photo_name":None,"input_df":DEFAULT_BANKS.copy()}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def go_to(page:int): st.session_state.page=page; st.rerun()

def page_header(step:int): st.caption({1:"Step 1 of 3 — Project Information",2:"Step 2 of 3 — Tower & Lift Engineering Inputs",3:"Step 3 of 3 — Benchmarks, Results & Recommendations"}[step])

init_state()
if st.session_state.page == 1:
    page_header(1); st.title("🏗️ Project Information")
    project_name=st.text_input("Project Name", value=st.session_state.project_name)
    project_address=st.text_area("Project Address", value=st.session_state.project_address)
    prepared_by=st.text_input("Prepared By", value=st.session_state.prepared_by)
    c1,c2=st.columns(2)
    with c1:
        logo_file=st.file_uploader("Upload Company Logo", type=["png","jpg","jpeg"])
        if logo_file: st.image(logo_file, caption="Company Logo Preview", width=180)
    with c2:
        project_photo=st.file_uploader("Upload Project Photo", type=["png","jpg","jpeg"])
        if project_photo: st.image(project_photo, caption="Project Photo Preview", width=260)
    if st.button("Next →", type="primary"):
        st.session_state.project_name=project_name; st.session_state.project_address=project_address; st.session_state.prepared_by=prepared_by
        if logo_file: st.session_state.logo_bytes=logo_file.getvalue(); st.session_state.logo_name=logo_file.name
        if project_photo: st.session_state.project_photo_bytes=project_photo.getvalue(); st.session_state.project_photo_name=project_photo.name
        go_to(2)
elif st.session_state.page == 2:
    page_header(2); st.title("🛗 Tower & Lift Engineering Inputs")
    st.write("Include architectural, population, hardware, door, control/zoning, and pit/overhead data.")
    column_config={"building_type":st.column_config.SelectboxColumn("building_type", options=BUILDING_TYPES, required=True),"building_grade":st.column_config.SelectboxColumn("building_grade", options=BUILDING_GRADES, required=True),"door_type":st.column_config.SelectboxColumn("door_type", options=DOOR_TYPES, required=True),"zoning_strategy":st.column_config.SelectboxColumn("zoning_strategy", options=ZONING_OPTIONS, required=True)}
    edited_df=st.data_editor(st.session_state.input_df, num_rows="dynamic", use_container_width=True, hide_index=True, column_config=column_config)
    c1,c2=st.columns(2)
    with c1:
        if st.button("← Back"): st.session_state.input_df=edited_df; go_to(1)
    with c2:
        if st.button("Generate Results →", type="primary"): st.session_state.input_df=clean_input_df(edited_df); go_to(3)
elif st.session_state.page == 3:
    page_header(3); st.title("📊 Benchmarks, Results & Recommendations")
    input_df=clean_input_df(st.session_state.input_df); banks=[LiftBankInput(**row) for row in input_df.to_dict(orient="records")]
    analysis_df=build_analysis_rows(banks); rec_df=build_recommendation_rows(banks); bm_df=build_benchmark_rows(banks)
    st.subheader("Project"); st.write(f"**Project Name:** {st.session_state.project_name}"); st.write(f"**Address:** {st.session_state.project_address or '-'}"); st.write(f"**Prepared By:** {st.session_state.prepared_by}")
    if st.session_state.logo_bytes: st.image(st.session_state.logo_bytes, caption="Company Logo", width=150)
    if st.session_state.project_photo_bytes: st.image(st.session_state.project_photo_bytes, caption="Project Photo", width=300)
    st.subheader("Benchmark Targets"); st.dataframe(bm_df, use_container_width=True, hide_index=True)
    st.subheader("Result Recommendations"); st.dataframe(rec_df, use_container_width=True, hide_index=True)
    st.subheader("Detailed Benchmark Analysis"); st.dataframe(analysis_df, use_container_width=True, hide_index=True)
    m1,m2,m3=st.columns(3); m1.metric("Total Checks", len(analysis_df)); m2.metric("PASS", int((analysis_df["Result"]=="PASS").sum()) if not analysis_df.empty else 0); m3.metric("FAIL", int((analysis_df["Result"]=="FAIL").sum()) if not analysis_df.empty else 0)
    excel_buffer=io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        pd.DataFrame([{"Project Name":st.session_state.project_name,"Address":st.session_state.project_address,"Prepared By":st.session_state.prepared_by}]).to_excel(writer, sheet_name="Project Info", index=False)
        input_df.to_excel(writer, sheet_name="Inputs", index=False); bm_df.to_excel(writer, sheet_name="Benchmark Targets", index=False); rec_df.to_excel(writer, sheet_name="Recommendations", index=False); analysis_df.to_excel(writer, sheet_name="Detailed Analysis", index=False)
    excel_buffer.seek(0)
    pdf_bytes=create_pdf(st.session_state.project_name, st.session_state.project_address, st.session_state.prepared_by, st.session_state.logo_bytes, st.session_state.logo_name, st.session_state.project_photo_bytes, st.session_state.project_photo_name, input_df, analysis_df, rec_df, bm_df)
    st.subheader("Downloads"); d1,d2,d3=st.columns(3)
    with d1: st.download_button("Download Excel", data=excel_buffer.getvalue(), file_name="vt_engineering_review.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with d2: st.download_button("Download PDF", data=pdf_bytes, file_name="vt_engineering_review.pdf", mime="application/pdf", use_container_width=True)
    with d3: st.download_button("Download CSV", data=analysis_df.to_csv(index=False).encode("utf-8-sig"), file_name="vt_detailed_analysis.csv", mime="text/csv", use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        if st.button("← Back to Tower Inputs"): go_to(2)
    with c2:
        if st.button("Start New Project"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    st.warning("Preliminary benchmark comparison only. Final VT traffic analysis, fire/life-safety compliance and shaft dimensions must be confirmed by the elevator specialist/manufacturer.")
