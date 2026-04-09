# Polymarket Weather Trading Bot

A production-ready, automated Python 3.11+ trading bot that scalps weather markets on Polymarket. Specifically designed to replicate high-conviction strategies on "highest temperature" daily markets using real-time aviation data (METAR/TAF) and Google Gemini (gemini-2.5-flash) analysis to find probability edges.

## Architecture Highlights
- **Market Discovery**: Automatically scans Polymarket Gamma API for active weather markets and groups them by target city.
- **Weather Fetcher**: Polls free, official aviationweather.gov (METAR/TAF structure) for real-time airport readings.
- **AI Analyzer**: Feeds METAR, current probabilities, and targets into Gemini to extract mathematically robust win confidences and edges.
- **Trading Engine**: Built on the official `py-clob-client` via Polygon (Chain ID 137). Supports Dry Run.
- **Portfolio Manager**: SQLite database tracking open positions per city and limiting total risk (e.g. 15-20% max bankroll).
- **Scheduler**: APScheduler runs every 1-3 hours flawlessly without cron requirements.

## Setup Instructions

### 1. Requirements

- Python 3.11+
- An API Key for Google Gemini (https://aistudio.google.com/app/apikey)
- A Polygon wallet funded with USDC.e (Polymarket token) and a tiny amount of MATIC/POL for gas.

### 2. Polymarket Keys
To get your Polymarket CLOB keys and Funder Address:
- Ensure your wallet has funds on Polygon.
- The `FUNDER_ADDRESS` is the wallet address holding the USDC.
- `POLYMARKET_PRIVATE_KEY` is the private key of the EOAs authorized. If using a proxy wallet created via Polymarket UI, follow Polymarket SDK guidelines to derive your credentials via EIP-712 signatures.

### 3. Installation

1. Select or create your python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `config/settings.py` to add new cities or adjust parameters as needed.

### 4. Configuration
Duplicate `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Ensure `DRY_RUN=True` for your first tests!

### Telegram Alerts (Optional)
Send `@BotFather` a generic `/newbot` command on telegram to get a token, and chat with `userinfobot` to get your Chat ID. Enter them in the `.env`.

### 5. Running the Bot
```bash
python src/main.py
```

## Docker Environment
You can run this perfectly on a VPS (Hetzner) or a Mac Mini.
```bash
docker build -t pm-weather-bot .
docker run -d --env-file .env --name weather_bot pm-weather-bot
```

## Disclaimer
> **Risk Warning**: For educational use. Automated trading can rapidly deplete your funds if misconfigured. You assume all responsibility. Test on `DRY_RUN` heavily.
