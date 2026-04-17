import streamlit as st
import subprocess
import requests
import json
import os
import datetime

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS_FILE = "prompts.json"
SETTINGS_FILE = "settings.json"
LMS_API_BASE = "http://localhost:1234/v1"   # LM Studio default

DEFAULT_PROMPTS = [
    "Summarise this document in 3 bullet points.",
    "List the key action items from this document.",
    "What are the main risks or concerns mentioned?",
    "Extract all dates and deadlines mentioned.",
    "Write an executive summary of this document.",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE) as f:
            return json.load(f)
    return DEFAULT_PROMPTS.copy()

def save_prompts(prompts):
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts, f, indent=2)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {"include_full_text": True, "model": "google/gemma-4-e4b"}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

def check_server():
    try:
        # Check if the local API is responding
        response = requests.get(f"{LMS_API_BASE}/models", timeout=2)
        if response.status_code == 200:
            return True, "Server is responding to API requests."
        return False, f"Server returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to LM Studio API. Is the server started?"
    except Exception as e:
        return False, str(e)
# def check_server():
#     try:
#         result = subprocess.run(
#             ["lms", "server", "start"],
#             capture_output=True, text=True, timeout=8,
#             shell=True  # needed on Windows
#         )
#         output = result.stdout + result.stderr
#         running = result.returncode == 0 and "running" in output.lower()
#         return running, output.strip() or "(no output)"
#     except FileNotFoundError:
#         return False, "`lms` command not found. Is LM Studio installed and on PATH?"
#     except subprocess.TimeoutExpired:
#         return False, "Command timed out."
#     except Exception as e:
#         return False, str(e)

def extract_pdf_text(uploaded_file):
    try:
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return text, None
    except ImportError:
        return None, "pypdf not installed. Run: pip install pypdf"
    except Exception as e:
        return None, str(e)

def stream_from_llm(prompt, pdf_text, model, include_full_text):
    """Yields text chunks from the LLM via SSE streaming."""
    # system = "You are a Document Cleaning and Structural Expert."   ### PAM EDIT - keep it simple with just a user prompt
    if include_full_text and pdf_text:
        # user_msg = f"Here is the document content:\n\n{pdf_text}\n\n---\n\n{prompt}"
        user_msg = f"{prompt}\n\n### RAW TEXT TO PROCESS\n---\n{pdf_text}\n---"
    elif pdf_text:
        excerpt = pdf_text[:3000]
        # user_msg = f"Here is an excerpt from the document:\n\n{excerpt}\n\n---\n\n{prompt}"
        user_msg = f"{prompt}\n\n### RAW TEXT EXCERPTTO PROCESS\n---\n{pdf_text}\n---"
    else:
        user_msg = prompt



    payload = {
        "model": model,
        "messages": [
            # {"role": "system", "content": system},    ### PAM EDIT - keep it simple with just a user prompt
            {"role": "user",   "content": user_msg},
        ],
        "stream": True,
    }
    with requests.post(
        f"{LMS_API_BASE}/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


def save_response_to_file(text, output_dir="responses"):
    """Saves response text to a timestamped .txt file. Returns the path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"response_{ts}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.abspath(path)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local LLM Console",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0d0f14;
    color: #e2e8f0;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.stButton > button {
    background: #1e40af;
    color: #e2e8f0;
    border: none;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2563eb; }

.stTextArea textarea {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: #111827;
    color: #a5f3fc;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
}

.response-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #a5f3fc;
    white-space: pre-wrap;
    line-height: 1.7;
    min-height: 120px;
}

.status-ok   { color: #4ade80; font-weight: 700; }
.status-fail { color: #f87171; font-weight: 700; }

.panel-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.3rem;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stToggle"] label {
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "prompts" not in st.session_state:
    st.session_state.prompts = load_prompts()
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
if "response" not in st.session_state:
    st.session_state.response = ""
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "server_status" not in st.session_state:
    st.session_state.server_status = None   # None = unchecked

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Local LLM Console")
st.markdown("---")

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

# ════════════════════════════ LEFT PANEL ════════════════════════════════════
with left:
    # ── Server Control ──────────────────────────────────────────────────────
    st.markdown('<p class="panel-label">Server Control</p>', unsafe_allow_html=True)
    col_check, col_start = st.columns(2)

    with col_check:
        if st.button("🔍 Check Status", use_container_width=True):
            with st.spinner("Checking..."):
                ok, msg = check_server()
                st.session_state.server_status = (ok, msg)

    with col_start:
        if st.button("🚀 Start Server", use_container_width=True):
            try:
                # We use Popen so the app doesn't wait for the server to close
                # 'shell=True' is usually needed for the 'lms' command on Windows
                subprocess.Popen(
                    ["lms", "server", "start"], 
                    shell=True, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                st.toast("Attempting to start LM Studio server...")
                st.session_state.server_status = (True, "Start command sent!")
            except Exception as e:
                st.error(f"Failed to start: {e}")

    # Display Status Indicator
    if st.session_state.server_status is not None:
        ok, msg = st.session_state.server_status
        label = "● ONLINE" if ok else "● OFFLINE"
        cls = "status-ok" if ok else "status-fail"
        st.markdown(f'<div style="text-align:center; margin-top:10px;"><span class="{cls}">{label}</span></div>', unsafe_allow_html=True)
        with st.expander("Server Logs/Details"):
            st.code(msg)

    st.markdown("---")

    # ── PDF Upload ─────────────────────────────────────────────────────────
    st.markdown('<p class="panel-label">PDF Document</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Select a PDF file", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        # Only process if it's a new file (based on name or size)
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded.name:
            text, err = extract_pdf_text(uploaded)
            if err:
                st.error(err)
            else:
                st.session_state.pdf_text = text
                st.session_state.last_uploaded = uploaded.name
                st.success(f"Loaded **{uploaded.name}**")
    # Remove the 'else: pdf_text = ""' to persist text between reruns

    st.markdown("---")

    # ── Settings ───────────────────────────────────────────────────────────
    st.markdown('<p class="panel-label">Settings</p>', unsafe_allow_html=True)
    settings = st.session_state.settings

    include_full = st.toggle(
        "Send full PDF text (off = first 3 000 chars only)",
        value=settings.get("include_full_text", True),
    )
    settings["include_full_text"] = include_full

    model_name = st.text_input("Model name (as shown in LM Studio)", value=settings.get("model", "google/gemma-4-e4b"))
    settings["model"] = model_name

    if st.button("Save Settings"):
        save_settings(settings)
        st.session_state.settings = settings
        st.success("Settings saved.")

    st.markdown("---")

    # ── Prompts ────────────────────────────────────────────────────────────
    st.markdown('<p class="panel-label">Predefined Prompts</p>', unsafe_allow_html=True)
    prompts = st.session_state.prompts

    # Ensure we always have 5 slots
    while len(prompts) < 5:
        prompts.append("")

    edited = []
    for i, p in enumerate(prompts[:5]):
        val = st.text_area(f"Prompt {i+1}", value=p, height=80, key=f"prompt_{i}")
        edited.append(val)

    col_s, col_r = st.columns(2)
    with col_s:
        if st.button("💾 Save Prompts"):
            st.session_state.prompts = edited
            save_prompts(edited)
            st.success("Prompts saved!")
    with col_r:
        if st.button("↺ Reset to Defaults"):
            st.session_state.prompts = DEFAULT_PROMPTS.copy()
            save_prompts(DEFAULT_PROMPTS)
            st.rerun()

# ════════════════════════════ RIGHT PANEL ═══════════════════════════════════
with right:
    st.markdown('<p class="panel-label">Run a Prompt</p>', unsafe_allow_html=True)

    active_prompts = [p for p in st.session_state.prompts if p.strip()]
    if not active_prompts:
        st.info("Add at least one prompt on the left to get started.")
    else:
        selected = st.selectbox(
            "Choose a prompt",
            options=active_prompts,
            format_func=lambda x: x[:80] + "…" if len(x) > 80 else x,
        )

        custom_override = st.text_area(
            "Or type / edit a prompt here (overrides selection if not empty)",
            height=90,
            placeholder="Leave blank to use the selected prompt above…",
        )

        final_prompt = custom_override.strip() or selected

        st.markdown(f"> **Prompt to send:** {final_prompt[:120]}{'…' if len(final_prompt) > 120 else ''}")

        if st.button("▶ Send to LLM", use_container_width=True):
            if not st.session_state.pdf_text and not custom_override.strip():
                st.warning("No PDF loaded and no custom prompt typed — the LLM will answer without document context.")
            st.session_state.response = ""
            try:
                collected = []
                stream_placeholder = st.empty()

                def token_stream():
                    for chunk in stream_from_llm(
                        final_prompt,
                        st.session_state.pdf_text,
                        model=settings.get("model", "google/gemma-4-e4b"),
                        include_full_text=settings.get("include_full_text", True),
                    ):
                        collected.append(chunk)
                        yield chunk

                stream_placeholder.write_stream(token_stream())
                st.session_state.response = "".join(collected)

            except requests.exceptions.ConnectionError:
                st.session_state.response = "❌ Could not connect to the LLM server.\nMake sure LM Studio is running and the server is started."
                st.error(st.session_state.response)
            except Exception as e:
                st.session_state.response = f"❌ Error: {e}"
                st.error(st.session_state.response)

    st.markdown("---")
    st.markdown('<p class="panel-label">Response</p>', unsafe_allow_html=True)

    if st.session_state.response:
        st.markdown(
            f'<div class="response-box">{st.session_state.response}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

        # ── Download (browser) ──────────────────────────────────────────────
        with c1:
            st.download_button(
                "⬇ Download .htlm",
                data=st.session_state.response,
                file_name="llm_response.html",
                mime="text/plain",
                use_container_width=True,
            )

        # ── Save to local file ──────────────────────────────────────────────
        with c2:
            if st.button("💾 Save to file", use_container_width=True):
                path = save_response_to_file(st.session_state.response)
                st.success(f"Saved → `{path}`")
# # ── Copy to clipboard ───────────────────────────────────────────────
#         with c3:
#             if st.session_state.response:
#                 # json.dumps handles all special characters and wrapping in double quotes
#                 js_content = json.dumps(st.session_state.response)
                
#                 # We use a single-quote f-string for the markdown to avoid breaking the HTML
#                 st.markdown(
#                     f'''
#                     <button onclick='const text = {js_content}; navigator.clipboard.writeText(text).then(()=>{{
#                         this.innerText="✅ Copied!";
#                         setTimeout(()=>this.innerText="📋 Copy to clipboard", 2000);
#                     }})'
#                     style="width:100%; padding:10px; background:#1e40af; color:#e2e8f0; 
#                            border:none; border-radius:6px; font-family:monospace; 
#                            font-size:0.82rem; cursor:pointer;">
#                         📋 Copy to clipboard
#                     </button>
#                     ''',
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.button("📋 Copy", disabled=True, use_container_width=True)

        # ── Copy to clipboard ───────────────────────────────────────────────
        with c3:
            if st.session_state.response:
                # json.dumps handles all special characters and wrapping in double quotes
                js_content = json.dumps(st.session_state.response)
                
                st.markdown(
                    f'''
                    <div style="width: 100%;">
                        <button id="copy-btn" onclick='const text = {js_content}; navigator.clipboard.writeText(text).then(()=>{{
                                const btn = document.getElementById("copy-btn");
                                btn.innerText="✅ Copied!";
                                btn.style.background="#4ade80";
                                setTimeout(()=>{{
                                    btn.innerText="📋 Copy";
                                    btn.style.background="#1e40af";
                                }}, 2000);
                            }})'
                            style="
                                width: 100%;
                                height: 38.4px; /* Matches standard Streamlit button height */
                                background: #1e40af;
                                color: #e2e8f0;
                                border: none;
                                border-radius: 6px;
                                font-family: 'JetBrains Mono', monospace;
                                font-size: 0.82rem;
                                cursor: pointer;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                transition: background 0.2s;
                            "
                            onMouseOver="this.style.background='#2563eb'"
                            onMouseOut="this.style.background='#1e40af'">
                            📋 Copy
                        </button>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                st.button("📋 Copy", disabled=True, use_container_width=True)
        # ── Clear ───────────────────────────────────────────────────────────
        with c4:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.response = ""
                st.rerun()
    else:
        st.markdown('<div class="response-box" style="color:#334155;">Response will stream here…</div>', unsafe_allow_html=True)
