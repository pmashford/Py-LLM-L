# My edits
uv run streamlit run app.py
ctrl + c to stop
# Local LLM Console

A Streamlit UI for sending predefined prompts to your local LLM (LM Studio).

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure LM Studio is running with the server started.

3. Run the app:
   ```
   streamlit run app.py
   ```

## Features

- 📄 PDF upload — extracts text and sends it as context
- 📝 5 editable, saveable predefined prompts
- ⚙️ Toggle: send full PDF or just the first 3,000 chars
- 🟢 Server status check via `lms server start`
- 💬 Response display with download button

## Notes

- The app talks to LM Studio's OpenAI-compatible API at `http://localhost:1234/v1`
- Set the **Model name** in Settings to match whatever model is loaded in LM Studio
- Prompts and settings are saved to `prompts.json` and `settings.json` in the same folder
