import json
import os
import asyncio
import time
import random
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from config.settings import config
from src.utils import logger
from src.calibration import calibration_engine

class AIAnalyzer:
    def __init__(self):
        self.clients = []
        if getattr(config, 'gemini_api_keys_str', None):
            keys = [k.strip() for k in config.gemini_api_keys_str.split(',') if k.strip()]
            for key in keys:
                self.clients.append(genai.Client(api_key=key))
        elif getattr(config, 'gemini_api_key', None):
            self.clients.append(genai.Client(api_key=config.gemini_api_key))
            
        self.current_client_idx = 0
        self.lock = asyncio.Lock() 
        self.global_lock = asyncio.Lock()
        self.last_global_call = 0.0
        self.client_metadata = [{"last_used": 0.0, "use_count": 0} for _ in range(len(self.clients))]
        
        logger.info(f"[AI] Initialized with {len(self.clients)} API keys.")

        # Use gemma-4-31b-it — stable model for financial probability assessment.
        self.fallback_models = ["gemma-4-31b-it"]
            
        self.system_prompt = "Calculate the TRUE probability for the market outcome based on weather arrays."
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "GEMINI.md"), "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except:
            pass
            
        self.generation_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            system_instruction=self.system_prompt,
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ]
        )

    async def analyze_city_batch(self, city: str, markets: list[dict], weather_data: dict, return_all: bool = False) -> list[dict]:
        """Analyzes all markets for one city in a single API call (Batch Mode)."""
        if not self.clients or not markets:
            return []
            
        try:
            # 1. Prepare Weather Data
            metar_raw = json.dumps(weather_data.get("metar", [])[:5])
            taf_raw = json.dumps(weather_data.get("taf", [])[:2])
            ecmwf = weather_data.get("ecmwf_summary", "N/A")
            gfs_hrrr = weather_data.get("gfs_hrrr_summary", "N/A")
            ensemble = weather_data.get("ensemble_summary", "N/A")

            # 2. Build Markets List for Prompt
            markets_context = []
            for i, m in enumerate(markets):
                yes_price = 0.0
                for out in m.get("outcomes", []):
                    if "yes" in out["name"].lower():
                        yes_price = out["current_price"]
                        break
                markets_context.append(f"({i+1}) Market: {m.get('question')} | YES Price: {yes_price} | ID: {m['market_id']}")

            markets_str = "\n".join(markets_context)
            
            content = f"""CITY: {city}
WEATHER DATA:
- METAR: {metar_raw}
- TAF: {taf_raw}
- ECMWF Forecast: {ecmwf}
- GFS/HRRR Forecast: {gfs_hrrr}
- Ensemble Mean: {ensemble}

ACTIVE MARKETS TO ANALYZE (Total: {len(markets)}):
{markets_str}

TASK:
Analyze all {len(markets)} markets above. 
Crucially: treat these as a unified probability distribution for {city}. For example, if you assign high probability to one temperature bucket, the others should be lower to maintain a realistic total distribution.

RESPONSE FORMAT:
Return a JSON array of objects. DO NOT follow the single-object schema from your system instructions. Instead, return a LIST of objects, where each object has this schema:
{{
  "market_id": "STRICTLY COPY FROM INPUT",
  "true_probability": float (0.0 - 1.0),
  "confidence": integer (0-100),
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "reasoning": "...",
  "recommended_action": "BUY_YES" | "BUY_NO" | "SKIP",
  "correction_applied": boolean
}}
"""

            max_keys = len(self.clients)
            success = False
            response = None
            
            # 3. Request logic with retries
            for retry in range(3):
                if success: break
                if retry > 0:
                    await asyncio.sleep(4.0)
                
                for attempt in range(max_keys):
                    async with self.lock:
                        idx = self.current_client_idx
                        self.current_client_idx = (self.current_client_idx + 1) % max_keys
                        client = self.clients[idx]
                        meta = self.client_metadata[idx]
                        
                        # Throttle
                        base_delay = 4.1
                        now = time.time()
                        if now - meta["last_used"] < base_delay:
                            await asyncio.sleep(base_delay - (now - meta["last_used"]))
                        meta["last_used"] = time.time()

                    async with self.global_lock:
                        now_g = time.time()
                        if now_g - self.last_global_call < 0.5:
                            await asyncio.sleep(0.5 - (now_g - self.last_global_call))
                        self.last_global_call = time.time()

                    try:
                        response = await asyncio.wait_for(
                            client.aio.models.generate_content(
                                model=self.fallback_models[0],
                                contents=content,
                                config=self.generation_config
                            ),
                            timeout=300  # 5 min per-call hard cap
                        )
                        if response and response.text:
                            success = True
                            meta["use_count"] += 1
                            break
                    except asyncio.TimeoutError:
                        logger.warning(f"[AI] Key {idx} timed out after 300s. Trying next key.")
                        meta["last_used"] = time.time() + 20.0
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "500" in err_str or "503" in err_str:
                            logger.warning(f"[AI] Key {idx} Batch API Error: {err_str[:40]}. Trying next.")
                            meta["last_used"] = time.time() + 10.0
                        else:
                            logger.error(f"[AI] Key {idx} Fatal ({type(e).__name__}): {e}")
                            break

            if not success or not response:
                return []

            # 4. Parse Batch Results
            try:
                raw_results = json.loads(response.text)
                if isinstance(raw_results, dict) and "predictions" in raw_results:
                    raw_results = raw_results["predictions"] # Handle some prompt variants
                if not isinstance(raw_results, list):
                    logger.error(f"AI Batch error: Expected list, got {type(raw_results)}")
                    return []
            except:
                logger.error(f"JSON Error in Batch response: {response.text[:200]}")
                return []

            if not return_all:
                logger.info(f"[{city}] AI Batch analyzed {len(raw_results)} markets.")
            
            # 5. Process and Rank Signals
            final_signals = []
            calib_factor = calibration_engine.calculate_calibration_factor(city)
            ev_threshold = config.ev_threshold.get(city, config.ev_threshold.get("default", 0.08))

            # Helper for mapping AI result back to market dict
            # AI is asked to return objects that match the market IDs or order.
            for item in raw_results:
                m_id = item.get("market_id")
                # Find matching market config
                m_config = next((m for m in markets if m["market_id"] == m_id), None)
                if not m_config: continue

                raw_prob = item.get("true_probability", 0.0)
                calibrated_prob_yes = min(0.99, max(0.01, raw_prob * calib_factor))
                
                if return_all:
                    # In re-analysis mode, return both YES and NO probabilities unconditionally
                    for out in m_config.get("outcomes", []):
                        out_name = out["name"]
                        p = calibrated_prob_yes if out_name.lower() == "yes" else (1.0 - calibrated_prob_yes)
                        final_signals.append({
                            "market_id": m_config["market_id"],
                            "question": m_config.get("question", "Unknown"),
                            "token_id": out["token_id"],
                            "outcome_name": out_name,
                            "outcome_slug": out_name,
                            "predicted_prob": p,
                            "city": city
                        })
                else:
                    # Original logic for finding BUY signals during Discovery
                    rec = item.get("recommended_action", "SKIP")
                    sentiment = item.get("sentiment", "NEUTRAL")
                    confidence = item.get("confidence", 0)
                    target_outcome_name = "Yes" if "YES" in rec.upper() else "No"

                    matched_out = next((o for o in m_config["outcomes"] if o["name"].lower() == target_outcome_name.lower()), None)
                    if not matched_out: continue

                    market_price = matched_out.get("current_price", 0.0)
                    p = calibrated_prob_yes if target_outcome_name == "Yes" else (1.0 - calibrated_prob_yes)
                    
                    edge_decimal = p - market_price
                    ev = (p * (1 - market_price)) - ((1 - p) * market_price)
                    
                    odds = (1 - market_price) / market_price if market_price > 0 else 0
                    full_kelly = (edge_decimal / odds) if odds > 0 else 0
                    fractional_kelly = full_kelly * config.kelly_fraction if full_kelly > 0 else 0.0

                    if ev > ev_threshold and fractional_kelly > 0 and confidence >= 82 and "BUY" in rec.upper():
                        final_signals.append({
                            "market_id": m_config["market_id"],
                            "question": m_config.get("question", "Unknown"),
                            "token_id": matched_out["token_id"],
                            "outcome_name": matched_out["name"],
                            "outcome_slug": target_outcome_name,
                            "market_price": market_price,
                            "true_probability": p,
                            "predicted_prob": p,
                            "ev": ev,
                            "edge": edge_decimal * 100,
                            "kelly": fractional_kelly,
                            "confidence": confidence,
                            "sentiment": sentiment,
                            "city": city
                        })

            return final_signals
            
        except Exception as e:
            logger.error(f"Global AI Batch Error for {city}: {e}")
            return []

ai_analyzer = AIAnalyzer()
