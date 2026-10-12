import gradio as gr
from openai import OpenAI
import base64
import pandas as pd
from datetime import datetime
import re

# 1. OpenRouter API Gateway Configuration
# IMPORTANT: Paste your newly generated key here to test, but change it back to "REMOVED_FOR_GITHUB" before saving/pushing!
OPENROUTER_API_KEY = "REMOVED_FOR_GITHUB"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 2. Structured Telemetry Database (BEAM Plus NB v2.0 compliant)
disposal_log = []

def analyze_trash(image_filepath, unit_id):
    if image_filepath is None:
        return "Please place an item in the sensor bay first!"

    with open(image_filepath, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

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
                        "Material: <Aluminum/Metal | Plastic | Paper | Glass | Other>\n"
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
        
        # Parse fields using regex for clean table formatting
        mat_match = re.search(r"Material:\s*(.*)", raw_text)
        item_match = re.search(r"Item:\s*(.*)", raw_text)
        wt_match = re.search(r"Est_Weight_kg:\s*([\d\.]+)", raw_text)
        act_match = re.search(r"Action:\s*(.*)", raw_text)

        material = mat_match.group(1).strip() if mat_match else "Other"
        item_name = item_match.group(1).strip() if item_match else "Unclassified Object"
        weight_kg = float(wt_match.group(1).strip()) if wt_match else 0.020
        action = act_match.group(1).strip() if act_match else raw_text

        # Append structured telemetry
        disposal_log.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Premises / Unit ID": unit_id,
            "Stream (BEAM Plus MW)": material,
            "Item Description": item_name,
            "Weight (kg)": weight_kg,
            "Treatment Status": "Compacted & Routed"
        })

        return f"✅ Identified: {item_name} ({material})\n⚖️ Logged Weight: {weight_kg} kg\n👉 Next Step: {action}"
    
    except Exception as e:
        return f"Error connecting to AI: {e}"

def generate_beam_report():
    if len(disposal_log) == 0:
        return pd.DataFrame(), "No operational data logged. Complete collections on Tab 1 first."
    
    df = pd.DataFrame(disposal_log)
    total_kg = df["Weight (kg)"].sum()
    stream_counts = df["Stream (BEAM Plus MW)"].value_counts().to_dict()
    unit_counts = df["Premises / Unit ID"].value_counts().to_dict()
    
    summary_metrics = (
        f"Total Weight Diverted: {total_kg:.3f} kg\n"
        f"Material Stream Breakdown: {stream_counts}\n"
        f"Participating Premises: {unit_counts}\n"
        f"Raw Collections:\n{df.to_string(index=False)}"
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2-vl-72b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sustainability consultant certified in BEAM Plus New Buildings v2.0 (Hong Kong).\n"
                        "Using the provided operational recycling telemetry, draft an official BEAM Plus Materials & Waste (MW) Audit Report.\n"
                        "Cover:\n"
                        "1. MW P1 Prerequisite compliance (source segregation and audit trail).\n"
                        "2. MW 12a credit contribution (diversion metrics across streams).\n"
                        "3. Premises-level accountability preparing for Hong Kong MSW charging.\n"
                        "Keep it rigorous, professional, and under 160 words."
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate the BEAM Plus MW audit based on these vehicle logs:\n{summary_metrics}"
                }
            ]
        )
        return df, response.choices[0].message.content
    except Exception as e:
        return df, f"Error generating report: {e}"

# 3. Two-Tab User Interface
with gr.Blocks(title="AnthroLoop Smart Bin") as demo:
    gr.Markdown("# ♻️ AnthroLoop: Autonomous Floor-to-Door Recycling Vehicle")
    
    # TAB 1: Resident Touchpoint (On the vehicle body)
    with gr.Tab("Resident Interface (Vehicle Bay)"):
        gr.Markdown("Select unit/premises, then present recyclable item to the optical intake.")
        with gr.Row():
            with gr.Column():
                unit_selector = gr.Textbox(
                    value="Tower 1 - Unit 14B",
                    label="Premises / User ID (MSW PAYT Traceability)",
                    placeholder="Enter location or unit ID manually..."
                )
                camera = gr.Image(sources=["webcam"], type="filepath", label="Optical Sensor Bay")
                submit_btn = gr.Button("Deposit Item", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Behavioral Nudge & Telemetry Confirmation", lines=6)
                
        submit_btn.click(fn=analyze_trash, inputs=[camera, unit_selector], outputs=output)
        
    # TAB 2: Facility Manager / ESG Audit Dashboard
    with gr.Tab("BEAM Plus & HKEX ESG Dashboard"):
        gr.Markdown("Auditable Operational Waste Log for BEAM Plus NB v2.0 (MW Category) & HKEX KPIs")
        
        data_table = gr.Dataframe(label="Vehicle Telemetry Log (Weight, Stream, Unit)", wrap=True)
        report_output = gr.Textbox(label="BEAM Plus NB v2.0 Operational Waste Audit", lines=8)
            
        generate_btn = gr.Button("Generate BEAM Plus Audit Report", variant="secondary")
        generate_btn.click(fn=generate_beam_report, inputs=[], outputs=[data_table, report_output])

if __name__ == "__main__":
    demo.launch()


