# Copy this file to config.py and fill in your Alpaca API credentials.
# config.py is gitignored so your keys are never committed.
#
# Get paper-trading keys at https://app.alpaca.markets/ (Paper Account -> API Keys).

APCA_API_KEY_ID = "<APCA_API_KEY_ID>"
APCA_API_SECRET_KEY = "<APCA_API_SECRET_KEY>"

# Keep this True until you are absolutely sure you want to trade real money.
PAPER_TRADE = True


# --- Optional: AI research agent (`python main.py research`) ---------------- #
# Only needed if you run the research agent. Set the key for the provider you use
# here, OR leave these unset and export the standard environment variable instead
# (the agent uses config.py if set, otherwise falls back to the env var). Ollama
# runs locally and needs no key.
#
#   --provider anthropic (default):  ANTHROPIC_API_KEY   (needs the `ai` extra)
#   --provider openai:               OPENAI_API_KEY      (needs the `openai` extra)
#   --provider ollama (local):       no key; optional OLLAMA_BASE_URL
#
# ANTHROPIC_API_KEY = "<sk-ant-...>"
# OPENAI_API_KEY = "<sk-...>"
# OLLAMA_BASE_URL = "http://localhost:11434"
