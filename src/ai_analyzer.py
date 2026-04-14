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
        self.client_metadata = [{"last_used": 0.0, "use_count": 0} for _ in range(len(self.clients))]
        
        logger.info(f"[AI] Initialized with {len(self.clients)} API keys.")

        # Use ONLY gemma-4-31b-it as requested by user.
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

    async def analyze_market(self, market_info: Dict[str, Any], weather_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simple, strictly sequential analysis to prevent 429/500 errors."""
        if not self.clients:
            logger.warning("Gemini API key(s) missing, skipping analysis.")
            return None
            
        try:
            # 1. Prepare Data
            market_price_yes = 0.0
            for out in market_info["outcomes"]:
                if "yes" in out["name"].lower():
                    market_price_yes = out["current_price"]
                    break
            
            metar_raw = json.dumps(weather_data.get("metar", [])[:5])
            taf_raw = json.dumps(weather_data.get("taf", [])[:2])
            
            ecmwf_summary = weather_data.get("ecmwf_summary", "N/A")
            gfs_hrrr_summary = weather_data.get("gfs_hrrr_summary", "N/A")
            ensemble_summary = weather_data.get("ensemble_summary", "N/A")

            content = f"""Market question: {market_info.get('question', '')}
Current Price for YES: {market_price_yes}
METAR: {metar_raw}
TAF: {taf_raw}
ECMWF: {ecmwf_summary}
GFS/HRRR: {gfs_hrrr_summary}
Ensemble: {ensemble_summary}"""

            max_keys = len(self.clients)
            success = False
            response = None
            
            # 2. Sequential Key Selection (ONE AT A TIME)
            for model_name in self.fallback_models:
                if success: break
                
                for attempt in range(max_keys):
                    # Rotate key and check cooldown
                    async with self.lock:
                        idx = self.current_client_idx
                        self.current_client_idx = (self.current_client_idx + 1) % max_keys
                        
                        client = self.clients[idx]
                        meta = self.client_metadata[idx]
                        
                        # Throttle
                        base_delay = 4.1 # Target 15 RPM per key (60/4.1 ~ 14.6)
                        now = time.time()
                        elapsed = now - meta["last_used"]
                        if elapsed < base_delay:
                            await asyncio.sleep(base_delay - elapsed)
                        
                        # UPDATED: Use a dummy last used to claim the slot
                        meta["last_used"] = time.time()
                    
                    # --- GLOBAL STAGGER (Safety across all keys) ---
                    async with self.global_lock:
                        now_g = time.time()
                        # Minimum 0.5s between ANY two API calls across the entire bot
                        wait_global = 0.5 - (now_g - self.last_global_call)
                        if wait_global > 0:
                            await asyncio.sleep(wait_global)
                        self.last_global_call = time.time()
                    
                    try:
                        response = await client.aio.models.generate_content(
                            model=model_name,
                            contents=content,
                            config=self.generation_config
                        )
                        if response and response.text:
                            success = True
                            meta["use_count"] += 1
                            break
                    except Exception as api_err:
                        err_msg = str(api_err)
                        if "429" in err_msg or "500" in err_msg or "quota" in err_msg.lower():
                            logger.warning(f"[AI] Key {idx} Error: {err_msg[:60]}. Skipping to next key.")
                            meta["last_used"] = time.time() + 10.0 # Small lockout
                        else:
                            logger.error(f"[AI] Key {idx} Fatal: {err_msg[:60]}")
                            break # Try next model if applicable
            
            if not success or not response:
                return None
            
            # 3. Parse and Calculate
            try:
                analysis_data = json.loads(response.text)
            except json.JSONDecodeError:
                logger.error(f"JSON Error: {response.text}")
                return None

            logger.info(f"AI JSON Response: {json.dumps(analysis_data, ensure_ascii=False)}")
            logger.info(f"AI Reasoning: {analysis_data.get('reasoning', 'N/A')}")

            rec = analysis_data.get("recommended_action", "SKIP")
            sentiment = analysis_data.get("sentiment", "NEUTRAL")
            confidence = analysis_data.get("confidence", 0)
            target_outcome_name = "Yes" if rec == "BUY_YES" else "No"
            
            matched_out = None
            for out in market_info["outcomes"]:
                if out["name"].lower() == target_outcome_name.lower():
                    matched_out = out
                    break
            
            if matched_out:
                market_price = matched_out["current_price"]
                city = market_info.get("city", "default")
                calib_factor = calibration_engine.calculate_calibration_factor(city)
                
                raw_prob_yes = analysis_data.get("true_probability", 0.0)
                calibrated_prob_yes = min(0.99, max(0.01, raw_prob_yes * calib_factor))
                
                p = calibrated_prob_yes if target_outcome_name == "Yes" else (1.0 - calibrated_prob_yes)
                ev = (p * (1 - market_price)) - ((1 - p) * market_price)
                
                edge_decimal = p - market_price
                odds = (1 - market_price) / market_price if market_price > 0 else 0
                full_kelly = (edge_decimal / odds) if odds > 0 else 0
                fractional_kelly = full_kelly * config.kelly_fraction if full_kelly > 0 else 0.0
                
                ev_threshold = config.ev_threshold.get(city, config.ev_threshold.get("default", 0.08))
                
                if ev > ev_threshold and fractional_kelly > 0 and confidence >= 82 and "BUY" in rec:
                    return {
                        "market_id": market_info["market_id"],
                        "question": market_info.get("question", "Unknown"),
                        "token_id": matched_out["token_id"],
                        "outcome_name": matched_out["name"],
                        "outcome_slug": target_outcome_name,
                        "market_price": market_price,
                        "true_probability": p,
                        "ev": ev,
                        "edge": edge_decimal * 100,
                        "kelly": fractional_kelly,
                        "confidence": confidence,
                        "sentiment": sentiment,
                        "city": city
                    }
            return None
            
        except Exception as e:
            logger.error(f"Global AI Error: {e}")
            return None

ai_analyzer = AIAnalyzer()
