import gradio as gr
import subprocess
import time
import requests
import cv2
import numpy as np
import os
import threading
import shutil

# State
active_backend = None
backend_process = None
avatar_list = []

def start_avatarify_backend(mode, enhance):
    global active_backend, backend_process, avatar_list
    if backend_process is not None:
        backend_process.terminate()
        backend_process.wait()
    
    python_exe = os.path.join(os.path.dirname(__file__), 'env', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = 'python'
        
    script_path = os.path.join(os.path.dirname(__file__), 'backend_avatarify.py')
    backend_process = subprocess.Popen([python_exe, script_path, '--port', '8001'])
    active_backend = 'avatarify'
    
    # Wait for server to start
    for _ in range(10):
        try:
            time.sleep(1)
            resp = requests.post('http://127.0.0.1:8001/init', json={"mode": mode, "enhance": enhance}, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                avatar_list = data.get("avatars", [])
                return "Avatarify Backend Started!"
        except Exception as e:
            pass
    return "Failed to start Avatarify Backend."

def start_fluxrt_backend(use_int8, use_reference, lora_weights):
    global active_backend, backend_process
    if backend_process is not None:
        backend_process.terminate()
        backend_process.wait()
        
    python_exe = os.path.join(os.path.dirname(__file__), 'fluxRT', 'env_fluxrt', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = 'python'
        
    script_path = os.path.join(os.path.dirname(__file__), 'fluxRT', 'backend_fluxrt.py')
    backend_process = subprocess.Popen([python_exe, script_path, '--port', '8002'])
    active_backend = 'fluxrt'
    
    # Wait for server to start
    for _ in range(10):
        try:
            time.sleep(2)
            resp = requests.post('http://127.0.0.1:8002/init', json={
                "use_int8": use_int8,
                "use_reference": use_reference,
                "lora_weights": lora_weights if lora_weights != "None" else None
            }, timeout=2)
            if resp.status_code == 200:
                return "FluxRT Backend Started!"
        except Exception as e:
            pass
    return "Failed to start FluxRT Backend."

def stop_backend():
    global active_backend, backend_process
    if backend_process is not None:
        backend_process.terminate()
        backend_process.wait()
        backend_process = None
    active_backend = None
    return "Backend Stopped."

def process_frame(frame):
    global active_backend
    if frame is None:
        return None
    if active_backend is None:
        return frame

    try:
        # Convert RGB to BGR for encoding
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, encoded_img = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        port = 8001 if active_backend == 'avatarify' else 8002
        url = f'http://127.0.0.1:{port}/process'
        
        response = requests.post(url, data=encoded_img.tobytes(), headers={'Content-Type': 'application/octet-stream'}, timeout=1)
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        pass
    
    return frame

# Avatarify Controls
def afy_control(action, value=None):
    if active_backend == 'avatarify':
        try:
            requests.post('http://127.0.0.1:8001/control', json={"action": action, "value": value})
        except:
            pass

# FluxRT Controls
def fluxrt_set_prompt(prompt):
    if active_backend == 'fluxrt':
        try:
            requests.post('http://127.0.0.1:8002/set_prompt', json={"prompt": prompt})
        except:
            pass

def download_flux_model(model_choice, progress=gr.Progress()):
    if model_choice == "Select a Model...":
        return "Please select a model."
        
    progress(0, desc="Starting download...")
    from huggingface_hub import snapshot_download, hf_hub_download
    flux_dir = os.path.join(os.path.dirname(__file__), 'fluxRT')
    
    try:
        if model_choice == "FLUX.2-klein-4B":
            repo_id = "black-forest-labs/FLUX.2-klein-4B"
            local_dir = os.path.join(flux_dir, "FLUX.2-klein-4B")
            progress(0.1, desc=f"Downloading {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, token=False)
            
        elif model_choice == "FLUX.2-klein-4B-int8":
            repo_id = "aydin99/FLUX.2-klein-4B-int8"
            local_dir = os.path.join(flux_dir, "FLUX.2-klein-4B-int8")
            progress(0.1, desc=f"Downloading {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, token=False)
            
        elif model_choice == "RIFE-safetensors":
            repo_id = "TensorForger/RIFE-safetensors"
            local_dir = os.path.join(flux_dir, "RIFE-safetensors")
            progress(0.1, desc=f"Downloading {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, token=False)
            
        elif model_choice.endswith(".gguf"):
            repo_id = "unsloth/FLUX.2-klein-4B-GGUF"
            local_dir = os.path.join(flux_dir, "models")
            os.makedirs(local_dir, exist_ok=True)
            progress(0.1, desc=f"Downloading {model_choice}...")
            hf_hub_download(repo_id=repo_id, filename=model_choice, local_dir=local_dir, token=False)
            
        else:
            return "Invalid model choice."
            
        progress(1.0, desc="Download complete!")
        return f"Successfully downloaded {model_choice}."
    except Exception as e:
        return f"Failed to download {model_choice}: {str(e)}"

# UI Design
css = """
body { background-color: #0b0f19; color: #ffffff; font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1200px !important; background: #111827; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
button.primary { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; border: none !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; box-shadow: 0 4px 14px 0 rgba(37,99,235,0.39) !important; transition: transform 0.2s ease !important; }
button.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important; }
button { border-radius: 8px !important; background: #1f2937 !important; border: 1px solid #374151 !important; color: #e5e7eb !important; transition: all 0.2s ease !important; }
button:hover { background: #374151 !important; }
.tabs { border-bottom: 1px solid #374151 !important; }
.tabitem { padding: 20px 0; }
textarea, input, select { background: #1f2937 !important; border: 1px solid #374151 !important; border-radius: 8px !important; color: white !important; }
"""

theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="AI Studio") as demo:
    gr.Markdown("# 🎨 AI Video Studio", elem_id="header")
    
    with gr.Row():
        with gr.Column(scale=2):
            webcam_input = gr.Image(sources=["webcam"], label="Input Feed")
        with gr.Column(scale=2):
            webcam_output = gr.Image(label="Processed Output")
    
    with gr.Tabs():
        with gr.TabItem("Avatarify"):
            with gr.Row():
                btn_fomm = gr.Button("Start Avatarify (FOMM)", variant="primary")
                btn_fomm_enh = gr.Button("Start Avatarify (FOMM) + Enhanced", variant="primary")
                btn_face = gr.Button("Start Face Animation", variant="primary")
                btn_face_enh = gr.Button("Start Face Animation + Enhanced", variant="primary")
            
            with gr.Row():
                btn_calib = gr.Button("Calibrate Face (X)")
                btn_track = gr.Checkbox(label="Auto Track")
                btn_z_in = gr.Button("Zoom In (W)")
                btn_z_out = gr.Button("Zoom Out (S)")
            
            with gr.Row():
                ava_drop = gr.Dropdown(choices=[], label="Select Avatar")
                btn_ava_refresh = gr.Button("Refresh Avatars")
            
            status_out_afy = gr.Textbox(label="Status", interactive=False)
            
            btn_fomm.click(lambda: start_avatarify_backend('fomm', False), outputs=status_out_afy)
            btn_fomm_enh.click(lambda: start_avatarify_backend('fomm', True), outputs=status_out_afy)
            btn_face.click(lambda: start_avatarify_backend('face-animation', False), outputs=status_out_afy)
            btn_face_enh.click(lambda: start_avatarify_backend('face-animation', True), outputs=status_out_afy)
            
            btn_calib.click(lambda: afy_control('calibrate'))
            btn_track.change(lambda x: afy_control('auto_track', x), inputs=btn_track)
            btn_z_in.click(lambda: afy_control('zoom_in'))
            btn_z_out.click(lambda: afy_control('zoom_out'))
            
            def update_avatar_dropdown():
                if active_backend == 'avatarify':
                    return gr.update(choices=avatar_list)
                return gr.update()
            
            btn_ava_refresh.click(update_avatar_dropdown, outputs=ava_drop)
            ava_drop.change(lambda x: afy_control('change_avatar', avatar_list.index(x) if x in avatar_list else 0), inputs=ava_drop)
            
        with gr.TabItem("FluxRT"):
            with gr.Row():
                flux_mode = gr.Dropdown(["Standard", "Reference"], label="FluxRT Mode", value="Standard")
                flux_int8 = gr.Checkbox(label="Enable INT8 Quantization", value=False)
                flux_lora = gr.Dropdown(["None"], label="LoRA Weights", value="None")
            
            with gr.Row():
                flux_prompt = gr.Textbox(label="Prompt", value="Turn this image into cyberpunk night, red and blue neon lamps, bokeh")
                
            btn_flux_start = gr.Button("Start FluxRT", variant="primary")
            status_out_flux = gr.Textbox(label="Status", interactive=False)
            
            btn_flux_start.click(lambda mode, int8, lora: start_fluxrt_backend(int8, mode == "Reference", lora), inputs=[flux_mode, flux_int8, flux_lora], outputs=status_out_flux)
            flux_prompt.change(fluxrt_set_prompt, inputs=flux_prompt)
            
            gr.Markdown("### Downloads")
            with gr.Row():
                gguf_models = [
                    "flux-2-klein-4b-Q8_0.gguf",
                    "flux-2-klein-4b-Q6_K.gguf",
                    "flux-2-klein-4b-Q5_K_S.gguf",
                    "flux-2-klein-4b-Q5_K_M.gguf",
                    "flux-2-klein-4b-Q4_K_S.gguf",
                    "flux-2-klein-4b-Q4_K_M.gguf",
                    "flux-2-klein-4b-Q4_0.gguf",
                    "flux-2-klein-4b-Q4_1.gguf",
                    "flux-2-klein-4b-Q3_K_S.gguf",
                    "flux-2-klein-4b-Q3_K_M.gguf",
                    "flux-2-klein-4b-Q2_K.gguf",
                    "flux-2-klein-4b-F16.gguf",
                    "flux-2-klein-4b-BF16.gguf"
                ]
                model_choices = ["Select a Model...", "FLUX.2-klein-4B", "FLUX.2-klein-4B-int8", "RIFE-safetensors"] + gguf_models
                dl_model = gr.Dropdown(model_choices, label="Download Model", value="Select a Model...")
            dl_status = gr.Textbox(label="Download Status", interactive=False)
            dl_model.change(download_flux_model, inputs=dl_model, outputs=dl_status)
            
            vram_table = """
            ### GGUF Models VRAM Requirements
            | Model File | Approx. VRAM Requirement | Notes |
            |---|---|---|
            | **flux-2-klein-4b-Q8_0.gguf** | ~3.5 GB | Heavily quantized (Q8_0), lowest memory footprint, slower inference but minimal VRAM use. |
            | **flux-2-klein-4b-Q6_K.gguf** | ~4.5 GB | Slightly higher precision than Q8, still efficient for low-VRAM GPUs. |
            | **flux-2-klein-4b-Q5_K_x** | ~5-5.5 GB | Balanced quantization; good trade-off between quality and memory. |
            | **flux-2-klein-4b-Q4_x** | ~6-6.5 GB | Mid-range quantization; moderate VRAM usage, better output fidelity. |
            | **flux-2-klein-4b-Q3_K_x** | ~7-8 GB | Higher precision, requires more VRAM but improves generation quality. |
            | **flux-2-klein-4b-Q2_K.gguf** | ~9-10 GB | Near full precision; suitable for GPUs with ≥12 GB VRAM. |
            | **flux-2-klein-4b-F16.gguf** | ~14-16 GB | Full FP16 precision; high quality, heavy VRAM load. |
            | **flux-2-klein-4b-BF16.gguf** | ~16-18 GB | Slightly heavier than FP16, best fidelity, intended for high-end GPUs. |
            """
            gr.Markdown(vram_table)
    
    with gr.Row():
        btn_stop = gr.Button("Stop All Backends", variant="secondary")
        btn_stop.click(stop_backend, outputs=status_out_afy)

    webcam_input.stream(process_frame, inputs=webcam_input, outputs=webcam_output, stream_every=0.04)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme, css=css)
