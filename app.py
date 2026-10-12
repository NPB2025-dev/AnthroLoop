import gradio as gr
from openai import OpenAI
import base64
import pandas as pd
from datetime import datetime

# 1. OpenRouter API Gateway Configuration
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"  # Replace with your actual OpenRouter API key

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 2. The Local Database (Logs every item scanned)
disposal_log = []

def analyze_trash(image_filepath):
    if image_filepath is None:
        return "Please take a picture of the item first!"

    with open(image_filepath, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2-vl-72b-instruct",
            max_tokens=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an optical recycling sensor in Hong Kong. "
                        "Step 1: Classify the primary material into EXACTLY ONE of these categories: "
                        "[Aluminum/Metal, Plastic, Paper, Other]. "
                        "Step 2: Identify the specific object. "
                        "Step 3: Provide a 1-sentence preparation nudge. "
                        "Format your response EXACTLY like this:\n"
                        "Material: <Category>\n"
                        "Item: <Name>\n"
                        "Action: <Nudge>"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze the primary object held up to the camera."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        
        result = response.choices[0].message.content
        
        disposal_log.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Item Detected": result
        })
        
        return result
    
    except Exception as e:
        return f"Error connecting to AI: {e}"


def generate_esg_report():
    if len(disposal_log) == 0:
        return pd.DataFrame(), "No data collected yet. Scan some items first."
    
    # Convert log to a structured data table
    df = pd.DataFrame(disposal_log)
    raw_data = df.to_string()
    
    # Send the data table to GenAI to write the ESG report
    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2-vl-72b-instruct",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an ESG compliance officer for a Hong Kong property developer. "
                        "Write a brief, professional ESG executive summary based on the provided waste collection data. "
                        "Explicitly mention HKEX compliance, waste reduction tracking, and contamination mitigation. "
                        "Keep it under 150 words."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Here is the raw data from our AnthroLoop smart bin today:\n{raw_data}\n\nWrite the ESG summary report."
                }
            ]
        )
        return df, response.choices[0].message.content
    except Exception as e:
        return df, f"Error generating report: {e}"

# 3. Two-Part User Interface Architecture
with gr.Blocks(title="AnthroLoop Smart Bin") as demo:
    gr.Markdown("# ♻️ AnthroLoop: Smart Autonomous Recycling Vehicle")
    
    # TAB 1: What the resident sees on the physical bin
    with gr.Tab("Resident Interface (The Bin)"):
        gr.Markdown("Drop an item into the collection bay for real-time behavioral guidance.")
        with gr.Row():
            camera = gr.Image(sources=["webcam"], type="filepath", label="Optical Sensor Bay")
            output = gr.Textbox(label="Behavioral Nudge", lines=3)
            
        submit_btn = gr.Button("Analyze Trash", variant="primary")
        submit_btn.click(fn=analyze_trash, inputs=camera, outputs=output)
        
    # TAB 2: What the property manager sees in their office
    with gr.Tab("Property Manager ESG Dashboard"):
        gr.Markdown("Automated B2B Reporting for HKEX Compliance and Waste Tax Tracking")
        
        with gr.Row():
            data_table = gr.Dataframe(label="Raw Disposal Log (Live Database)")
            report_output = gr.Textbox(label="GenAI ESG Executive Summary", lines=10)
            
        generate_btn = gr.Button("Generate Daily ESG Report", variant="secondary")
        generate_btn.click(fn=generate_esg_report, inputs=[], outputs=[data_table, report_output])

# 4. Server Initialization
if __name__ == "__main__":
    demo.launch()
