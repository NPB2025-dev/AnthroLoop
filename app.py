import gradio as gr
from openai import OpenAI
import base64
import pandas as pd
from datetime import datetime
import re
import os

# 1. OpenRouter API Gateway Configuration
OPENROUTER_API_KEY = "API key"  # Paste your key here for testing

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

VALID_STREAMS = ["Aluminum/Metal", "Plastic", "Paper", "Unindentified/Contamination"]

# Helper: Infer Building_ID for legacy CSV rows (e.g., SAVANNAH vs Lingnan LKK)
def infer_building_id(unit_str):
    unit_upper = str(unit_str).upper()
    if "SAVANNAH" in unit_upper:
        return "SAVANNAH_3A"
    elif "LINGNAN" in unit_upper or "LKK" in unit_upper:
        return "LINGNAN_LKK"
    elif "TOWER 1" in unit_upper:
        return "TOWER_1"
    return "ESTATE_MAIN"

# Helper: Normalize stream labels for clean merging with the CoC Manifest
def normalize_stream(stream_str):
    s = str(stream_str).strip()
    for valid in VALID_STREAMS:
        if valid.lower() in s.lower():
            return valid
    return "Unidentified/Contamination"

# Helper: Standardize DataFrame schema so legacy CSVs never break new columns
def ensure_schema(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "Timestamp", "Building_ID", "Premises / Unit ID", "Vehicle_ID",
            "Stream (BEAM Plus MW)", "Item Description", "Weight (kg)",
            "Treatment Status", "Is_Recyclable"
        ])

    if "Premises / Unit ID" not in df.columns:
        df["Premises / Unit ID"] = "Unknown Unit"

    if "Building_ID" not in df.columns:
        df["Building_ID"] = df["Premises / Unit ID"].apply(infer_building_id)
    else:
        df["Building_ID"] = df["Building_ID"].fillna(df["Premises / Unit ID"].apply(infer_building_id))

    if "Vehicle_ID" not in df.columns:
        df["Vehicle_ID"] = "AL-01"
    else:
        df["Vehicle_ID"] = df["Vehicle_ID"].fillna("AL-01")

    if "Stream (BEAM Plus MW)" not in df.columns:
        df["Stream (BEAM Plus MW)"] = "Unidentified/Contamination"
    df["Stream (BEAM Plus MW)"] = df["Stream (BEAM Plus MW)"].apply(normalize_stream)

    if "Item Description" not in df.columns:
        df["Item Description"] = "Unclassified Object"

    if "Weight (kg)" not in df.columns:
        df["Weight (kg)"] = 0.0
    df["Weight (kg)"] = pd.to_numeric(df["Weight (kg)"], errors="coerce").fillna(0.0)

    # Support legacy CSVs that used either "Hub Batch ID" or "Treatment Status"
    if "Treatment Status" not in df.columns:
        if "Hub Batch ID" in df.columns:
            df["Treatment Status"] = df["Hub Batch ID"]
        else:
            df["Treatment Status"] = "Pending Routing"
    if "Hub Batch ID" in df.columns:
        df = df.drop(columns=["Hub Batch ID"])

    df["Is_Recyclable"] = df["Stream (BEAM Plus MW)"].isin(VALID_STREAMS)
    return df

# 2. Database & Telemetry Logic (Schema-Safe Append)
def save_to_csv(filepath, data_dict):
    new_df = pd.DataFrame([data_dict])
    if os.path.isfile(filepath):
        try:
            existing_df = pd.read_csv(filepath)
            existing_df = ensure_schema(existing_df)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df

    combined_df = ensure_schema(combined_df)
    combined_df.to_csv(filepath, index=False)

def analyze_trash(image_filepath, building_id, unit_id, vehicle_id, target_csv):
    if image_filepath is None:
        return "Please place an item in the sensor bay first!"
    if not target_csv.endswith(".csv"):
        target_csv += ".csv"

    with open(image_filepath, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2-vl-72b-instruct",
            max_tokens=90,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an optical recycling vehicle sensor complying with Hong Kong BEAM Plus NB v2.0 standards.\n"
                        "Classify the item into exact lines:\n"
                        "Material: <Aluminum/Metal | Plastic | Paper | Glass | Unidentified/Contamination>\n"
                        "Item: <Name>\n"
                        "Est_Weight_kg: <Typical empty weight as float, e.g., 0.015 for can, 0.025 for 500ml PET bottle>\n"
                        "Action: <One short behavioral prep instruction under 15 words>"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze the item in the collection bay."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )

        raw_text = response.choices[0].message.content

        mat_match = re.search(r"Material:\s*(.*)", raw_text)
        item_match = re.search(r"Item:\s*(.*)", raw_text)
        wt_match = re.search(r"Est_Weight_kg:\s*([\d\.]+)", raw_text)
        act_match = re.search(r"Action:\s*(.*)", raw_text)

        raw_material = mat_match.group(1).strip() if mat_match else "Unidentified/Contamination"
        material = normalize_stream(raw_material)
        item_name = item_match.group(1).strip() if item_match else "Unclassified Object"
        weight_kg = float(wt_match.group(1).strip()) if wt_match else 0.020
        action = act_match.group(1).strip() if act_match else raw_text

        clean_building_id = building_id.strip() if building_id and building_id.strip() else infer_building_id(unit_id)
        clean_vehicle_id = vehicle_id.strip() if vehicle_id and vehicle_id.strip() else "AL-01"
        is_recyclable = material in VALID_STREAMS

        data_dict = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Building_ID": clean_building_id,
            "Premises / Unit ID": unit_id.strip(),
            "Vehicle_ID": clean_vehicle_id,
            "Stream (BEAM Plus MW)": material,
            "Item Description": item_name,
            "Weight (kg)": weight_kg,
            "Treatment Status": "Pending Routing" if is_recyclable else "Rejected (Contamination)",
            "Is_Recyclable": is_recyclable
        }

        save_to_csv(target_csv, data_dict)

        return (
            f"✅ Identified: {item_name} ({material})\n"
            f"🏢 Building: {clean_building_id} | Unit: {unit_id}\n"
            f"⚖️ Logged Weight: {weight_kg:.3f} kg\n"
            f"💾 Saved to: {target_csv}\n"
            f"👉 Next Step: {action}"
        )

    except Exception as e:
        return f"Error connecting to AI: {e}"

def generate_beam_report(target_csv, building_filter, total_units_input, coc_data):
    if not target_csv.endswith(".csv"):
        target_csv += ".csv"

    if not os.path.exists(target_csv):
        return pd.DataFrame(), "0.000", "0.0%", "0.0%", "Error: No operational data found.", None

    df = pd.read_csv(target_csv)
    if df.empty:
        return df, "0.000", "0.0%", "0.0%", "No operational data logged yet.", None

    df = ensure_schema(df)

    # 1. Filter by Building / Estate so multi-property logs aren't mixed in one audit
    filter_str = str(building_filter).strip() if building_filter else "ALL"
    if filter_str.upper() not in ["ALL", "ALL BUILDINGS", ""]:
        mask = (
            df["Building_ID"].astype(str).str.contains(filter_str, case=False, na=False) |
            df["Premises / Unit ID"].astype(str).str.contains(filter_str, case=False, na=False)
        )
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame(), "0.000", "0.0%", "0.0%", f"No telemetry records found matching building filter: '{filter_str}'.", None

    # 2. Safely parse manual unit denominator for KPI
    try:
        assumed_total_units = int(total_units_input)
        if assumed_total_units <= 0:
            assumed_total_units = 50
    except Exception:
        assumed_total_units = 50

    # 3. Standardize CoC Manifest DataFrame and join with Telemetry Log on Stream
    coc_cols = ["Stream (BEAM Plus MW)", "Consolidation Batch ID", "Downstream Collector", "Ticket Ref", "Audit Status"]
    if isinstance(coc_data, pd.DataFrame):
        coc_df = coc_data.copy()
        coc_df.columns = coc_cols[:len(coc_df.columns)]
    else:
        coc_df = pd.DataFrame(coc_data, columns=coc_cols)

    coc_df["Stream (BEAM Plus MW)"] = coc_df["Stream (BEAM Plus MW)"].apply(normalize_stream)
    coc_df = coc_df.drop_duplicates(subset=["Stream (BEAM Plus MW)"], keep="first")

    # Merge telemetry rows with downstream collector manifest
    merged_df = pd.merge(df, coc_df, on="Stream (BEAM Plus MW)", how="left")

    # 4. Reconcile Treatment Status & Audit Columns based on Downstream Verification
    def resolve_treatment(row):
        if not row["Is_Recyclable"]:
            return "Rejected (Contamination)"
        status = str(row.get("Audit Status", "")).strip().lower()
        batch = str(row.get("Consolidation Batch ID", "")).strip()
        if status == "verified" and batch and batch != "nan":
            return "Compacted & Routed"
        return "Pending Routing"

    merged_df["Treatment Status"] = merged_df.apply(resolve_treatment, axis=1)
    merged_df["Consolidation Batch ID"] = merged_df.apply(
        lambda r: r["Consolidation Batch ID"] if r["Is_Recyclable"] and pd.notna(r["Consolidation Batch ID"]) else "N/A", axis=1
    )
    merged_df["Downstream Collector"] = merged_df.apply(
        lambda r: r["Downstream Collector"] if r["Is_Recyclable"] and pd.notna(r["Downstream Collector"]) else "N/A (Unrouted/Contamination)", axis=1
    )
    merged_df["Ticket Ref"] = merged_df.apply(
        lambda r: r["Ticket Ref"] if r["Is_Recyclable"] and pd.notna(r["Ticket Ref"]) else "N/A", axis=1
    )
    merged_df["Audit Status"] = merged_df.apply(
        lambda r: r["Audit Status"] if r["Is_Recyclable"] and pd.notna(r["Audit Status"]) else "Excluded", axis=1
    )
    merged_df["Total_Building_Units"] = assumed_total_units
    merged_df["BEAM_Credit_Target"] = merged_df.apply(
        lambda r: "MW 12a & IA Occupant Engagement" if r["Treatment Status"] == "Compacted & Routed" else "Excluded / Pending", axis=1
    )

    # Order columns cleanly for the full BEAM Plus Evidence Package CSV
    export_cols = [
        "Timestamp",
        "Building_ID",
        "Premises / Unit ID",
        "Total_Building_Units",
        "Vehicle_ID",
        "Stream (BEAM Plus MW)",
        "Item Description",
        "Weight (kg)",
        "Is_Recyclable",
        "Treatment Status",
        "Consolidation Batch ID",
        "Downstream Collector",
        "Ticket Ref",
        "Audit Status",
        "BEAM_Credit_Target"
    ]
    audit_export_df = merged_df[export_cols]

    # Columns displayed in the UI Live Collection Database table
    ui_display_cols = [
        "Timestamp", "Building_ID", "Premises / Unit ID", "Stream (BEAM Plus MW)",
        "Item Description", "Weight (kg)", "Treatment Status", "Is_Recyclable",
        "Consolidation Batch ID", "Ticket Ref"
    ]
    ui_df = audit_export_df[ui_display_cols]

    # 5. Calculate Verified Audit KPIs
    total_intake_kg = audit_export_df["Weight (kg)"].sum()
    verified_mask = audit_export_df["Is_Recyclable"] & (audit_export_df["Treatment Status"] == "Compacted & Routed")
    verified_diverted_kg = audit_export_df.loc[verified_mask, "Weight (kg)"].sum()

    recyclable_kg = audit_export_df.loc[audit_export_df["Is_Recyclable"], "Weight (kg)"].sum()
    sorting_yield = (recyclable_kg / total_intake_kg * 100) if total_intake_kg > 0 else 0.0

    unique_units = audit_export_df.loc[audit_export_df["Is_Recyclable"], "Premises / Unit ID"].nunique()
    participation_rate = min((unique_units / assumed_total_units) * 100, 100.0)

    # 6. Export the complete, joined audit trail CSV (fixes the 67-byte summary issue)
    export_filename = f"BEAM_Evidence_Package_{datetime.now().strftime('%Y%m%d')}.csv"
    audit_export_df.to_csv(export_filename, index=False)

    # Prepare stream-by-stream summary with CoC references for the LLM Executive Summary
    stream_summary = (
        audit_export_df[verified_mask]
        .groupby(["Stream (BEAM Plus MW)", "Consolidation Batch ID", "Downstream Collector", "Ticket Ref"])["Weight (kg)"]
        .sum()
        .reset_index()
    )
    summary_str = stream_summary.to_string(index=False) if not stream_summary.empty else "No verified routed streams."

    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2-vl-72b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an impartial ESG Auditor generating a BEAM Plus NB v2.0 / EB v2.0 & HKEX compliance report. "
                        "Strictly adhere to the following output format. DO NOT use conversational filler.\n\n"
                        "### 🏢 BEAM Plus Monthly Audit Report\n"
                        f"**Date of Audit:** {datetime.now().strftime('%Y-%m-%d')} | **Building Scope:** {filter_str} | **Total Verified Diverted Weight:** {verified_diverted_kg:.3f} kg\n\n"
                        "#### 1. MW 12a Enhanced Diversion Metrics\n"
                        "<Provide a bulleted list of the verified weights diverted per stream, including Batch ID and Ticket Ref for each stream.>\n\n"
                        "#### 2. Chain of Custody & Contamination\n"
                        "<State the Effective Sorting Yield and explicitly name the EPD-registered Downstream Collectors, Batch IDs, and Ticket Refs proving authorized landfill diversion.>\n\n"
                        "#### 3. Occupant Engagement (Innovations Credit)\n"
                        "<State the unit-level participation rate, unique participating units vs total registered units, and how automated floor-to-door tracking supports IA credits.>"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Verified Stream Summary:\n{summary_str}\n\n"
                        f"Total Verified Diverted Weight: {verified_diverted_kg:.3f} kg (Total Intake: {total_intake_kg:.3f} kg)\n"
                        f"Effective Sorting Yield: {sorting_yield:.1f}%\n"
                        f"Active Household Participation: {participation_rate:.1f}% ({unique_units} unique active units out of {assumed_total_units} registered units)."
                    )
                }
            ]
        )
        ai_report = response.choices[0].message.content
        return (
            ui_df,
            f"{verified_diverted_kg:.3f}",
            f"{participation_rate:.1f}%",
            f"{sorting_yield:.1f}%",
            ai_report,
            export_filename
        )

    except Exception as e:
        return (
            ui_df,
            f"{verified_diverted_kg:.3f}",
            f"{participation_rate:.1f}%",
            f"{sorting_yield:.1f}%",
            f"Error generating AI summary (CSV still exported successfully): {e}",
            export_filename
        )

# --- Internal Ops Tools ---
def load_raw_data(target_csv):
    if not target_csv.endswith(".csv"):
        target_csv += ".csv"
    if os.path.exists(target_csv):
        return ensure_schema(pd.read_csv(target_csv))
    return pd.DataFrame()

def clear_database(target_csv):
    if not target_csv.endswith(".csv"):
        target_csv += ".csv"
    if os.path.exists(target_csv):
        os.remove(target_csv)
    return pd.DataFrame(), f"⚠️ Database '{target_csv}' has been wiped clean for the next demo."


# 3. Three-Tab User Interface Architecture
with gr.Blocks(title="AnthroLoop Smart Bin", theme=gr.themes.Default(neutral_hue="slate")) as demo:
    gr.Markdown("# ♻️ AnthroLoop: Autonomous Floor-to-Door Recycling Vehicle")

    # Global State
    active_database = gr.State("experimental_log.csv")

    # ---------------------------------------------------------
    # TAB 1: Resident Interface
    # ---------------------------------------------------------
    with gr.Tab("1. Resident Interface (Vehicle Bay)"):
        gr.Markdown("Select building, unit, and target database, then present recyclable item to the optical intake.")
        with gr.Row():
            with gr.Column():
                building_selector = gr.Dropdown(
                    choices=["SAVANNAH_3A", "LINGNAN_LKK", "TOWER_1"],
                    value="SAVANNAH_3A",
                    allow_custom_value=True,
                    label="Building / Estate ID"
                )
                unit_selector = gr.Textbox(
                    value="HK NT TKO 3 CHI SHIN ST SAVANNAH 3A 11B",
                    label="Premises / Unit ID"
                )
                vehicle_selector = gr.Textbox(
                    value="AL-01",
                    label="Autonomous Vehicle / Scale ID"
                )
                csv_selector = gr.Dropdown(
                    choices=["experimental_log.csv", "competition_demo.csv"],
                    value="experimental_log.csv",
                    allow_custom_value=True,
                    label="Target Database File"
                )
                camera = gr.Image(sources=["webcam", "upload"], type="filepath", label="Optical Sensor Bay")
                submit_btn = gr.Button("Deposit Item", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Behavioral Nudge & Telemetry Confirmation", lines=7)

        csv_selector.change(lambda x: x, inputs=csv_selector, outputs=active_database)
        submit_btn.click(
            fn=analyze_trash,
            inputs=[camera, building_selector, unit_selector, vehicle_selector, active_database],
            outputs=output
        )

    # ---------------------------------------------------------
    # TAB 2: Building Manager (Client Dashboard)
    # ---------------------------------------------------------
    with gr.Tab("2. Building Manager (ESG & BEAM Plus)"):
        gr.Markdown("## 📊 Automated ESG Telemetry & BEAM Plus Audit (MW Category)")

        with gr.Row():
            building_filter_input = gr.Dropdown(
                choices=["ALL", "SAVANNAH_3A", "LINGNAN_LKK", "TOWER_1"],
                value="ALL",
                allow_custom_value=True,
                label="Building / Estate Scope Filter (Select 'ALL' or specific property)"
            )
            total_units_input = gr.Number(
                value=2,
                label="Total Registered Units in Building (Manual Input for KPI)",
                precision=0
            )

        with gr.Row():
            total_weight_kpi = gr.Textbox(label="Total Verified Weight Diverted (kg)", interactive=False)
            sorting_yield_kpi = gr.Textbox(label="Effective Sorting Yield (%)", interactive=False)
            participation_kpi = gr.Textbox(label="Active Household Participation", interactive=False)

        gr.Markdown("### 🔗 Chain of Custody: Downstream Hub Reconciliation")
        coc_table = gr.Dataframe(
            value=[
                ["Aluminum/Metal", "HUB-TKO-202610-04M", "Mil Mill / EPD Metal Collector", "REC-99201A", "Verified"],
                ["Plastic", "HUB-TKO-202610-04P", "New Life Plastics (EPD-Registered)", "REC-99202B", "Verified"],
                ["Paper", "HUB-TKO-202610-04W", "Mil Mill (EPD-Registered)", "REC-99203C", "Verified"],
                ["Glass", "HUB-TKO-202610-04G", "Bagzio (EPD Glass Contractor)", "REC-99204D", "Verified"]
            ],
            headers=["Stream (BEAM Plus MW)", "Consolidation Batch ID", "Downstream Collector", "Ticket Ref", "Audit Status"],
            interactive=True,
            row_count=(4, "dynamic"),
            label="Editable Downstream Manifest (Stream-Specific Collector Reconciliation)"
        )

        gr.Markdown("### Live Collection Database")
        data_table = gr.Dataframe(label="Reconciled Vehicle Telemetry Log", wrap=True)

        gr.Markdown("### Formal ESG Executive Summary")
        with gr.Row():
            generate_btn = gr.Button("Pull Data & Generate BEAM Plus Audit", variant="primary")
            export_file = gr.File(label="Download BEAM Plus Evidence Package (CSV)")

        report_output = gr.Markdown(label="Generated Report")

        generate_btn.click(
            fn=generate_beam_report,
            inputs=[active_database, building_filter_input, total_units_input, coc_table],
            outputs=[data_table, total_weight_kpi, participation_kpi, sorting_yield_kpi, report_output, export_file]
        )

    # ---------------------------------------------------------
    # TAB 3: Internal Ops (Developer Backend)
    # ---------------------------------------------------------
    with gr.Tab("3. Internal Ops (Developer Backend)"):
        gr.Markdown("## ⚙️ Fleet Management & Database Controls")
        gr.Markdown("View raw telemetry, test connections, and wipe databases between competition pitches.")

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Raw Database Viewer")
            wipe_btn = gr.Button("🗑️ Wipe Active Database (Reset for Demo)", variant="stop")

        system_msg = gr.Textbox(label="System Notification", interactive=False)
        raw_db_viewer = gr.Dataframe(label="Raw CSV File Output")

        gr.Markdown("### 📡 Hardware Telemetry (Simulated)")
        gr.Dataframe(
            value=[["AL-01", "Online", "92%", "34°C", "0.02ms"]],
            headers=["Vehicle ID", "Vision Subsystem", "Battery Level", "Bay Temp", "API Latency"],
            interactive=False
        )

        refresh_btn.click(fn=load_raw_data, inputs=[active_database], outputs=[raw_db_viewer])
        wipe_btn.click(fn=clear_database, inputs=[active_database], outputs=[raw_db_viewer, system_msg])

if __name__ == "__main__":
    demo.launch()
