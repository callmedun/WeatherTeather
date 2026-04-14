import json
import os
import asyncio
import time
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
            # Legacy fallback
            self.clients.append(genai.Client(api_key=config.gemini_api_key))
            
        self.current_client_idx = 0
        self.consecutive_failures = 0 # Circuit breaker counter
        self.client_metadata = [{"last_used": 0.0, "use_count": 0} for _ in range(len(self.clients))]
        # Primary is gemini_model, fallback to flash-8b as it's often more available
        self.fallback_models = ["gemini-flash-lite-latest", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
            
        # Provide fallback if GEMINI.md isn't located
        self.system_prompt = "Calculate the TRUE probability for the market outcome based on weather arrays."
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "GEMINI.md"), "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except:
            pass
            
        # Configure model config using the modern google-genai structured GenerateContentConfig
        # We disable safety blocks just in case trading terms trip false positives
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
        if not self.clients:
            logger.warning("Gemini API key(s) missing, skipping analysis.")
            return None
            
        try:
            # Get YES market price specifically or use first available if labeled differently
            market_price_yes = 0.0
            for out in market_info["outcomes"]:
                if "yes" in out["name"].lower():
                    market_price_yes = out["current_price"]
                    break
            
            # Format raw strings for prompt
            metar_raw = json.dumps(weather_data.get("metar", [])[:5])
            taf_raw = json.dumps(weather_data.get("taf", [])[:2])
            
            ecmwf_summary = weather_data.get("ecmwf_summary", "N/A - Pending integration")
            gfs_hrrr_summary = weather_data.get("gfs_hrrr_summary", "N/A - Pending integration")
            ensemble_summary = weather_data.get("ensemble_summary", "N/A - Pending integration")

            content = f"""Market question: {market_info.get('question', '')}

Current Market Price for YES: {market_price_yes}

=== OFFICIAL AVIATION DATA (PRIMARY SOURCE) ===
METAR (current observed): {metar_raw}
TAF (official forecast): {taf_raw}

=== MULTI-MODEL ENSEMBLE FORECASTS (FOR CORRECTION ONLY) ===
Open-Meteo ECMWF IFS summary: {ecmwf_summary}
Open-Meteo GFS + HRRR summary: {gfs_hrrr_summary}
Ensemble consensus (51+ members): {ensemble_summary}

Task: Follow the System Prompt from GEMINI.md exactly. Calculate True Probability using the weighted formula. Apply correction only where justified. Output ONLY the JSON."""

            max_keys = len(self.clients)
            success = False
            response = None
            
            # Circuit breaker check: if we had 5 consecutive total failures previously, fail fast
            if self.consecutive_failures >= 5:
                # logger.warning("[AI] Circuit breaker ACTIVE. Skipping market analysis.")
                return None

            for model_name in self.fallback_models:
                if success: break
                
                for attempt in range(max_keys):
                    # Rotate client index for every market to balance load (Round Robin)
                    idx = self.current_client_idx
                    self.current_client_idx = (self.current_client_idx + 1) % max_keys
                    
                    client = self.clients[idx]
                    meta = self.client_metadata[idx]
                    
                    # Throttle: Ensure at least 4.1s between uses of THIS specific key
                    now = time.time()
                    elapsed = now - meta["last_used"]
                    if elapsed < 4.1:
                        wait_needed = 4.1 - elapsed
                        # logger.debug(f"[AI] Key {idx} throttling ({wait_needed:.1f}s delay)...")
                        await asyncio.sleep(wait_needed)

                    try:
                        # Execute generation using modern Async client
                        response = await client.aio.models.generate_content(
                            model=model_name,
                            contents=content,
                            config=self.generation_config
                        )
                        success = True
                        self.consecutive_failures = 0 # Reset on any success
                        meta["last_used"] = time.time()
                        meta["use_count"] += 1
                        
                        # Warning if near 1500 daily limit (approximate)
                        if meta["use_count"] > 1400:
                            logger.warning(f"⚠️ [AI] Key {idx} reaching daily quote limit ({meta['use_count']}/1500)")
                        
                        break 
                    except Exception as api_err:
                        err_msg = str(api_err)
                        if "429" in err_msg or "503" in err_msg or "quota" in err_msg.lower():
                            wait_time = (attempt + 1) * 2
                            logger.warning(f"[AI] Error ({model_name}) on key {idx}: {err_msg[:60]}. Retrying next...")
                            await asyncio.sleep(0.5) # Quick skip to next key
                        elif "404" in err_msg or "not found" in err_msg.lower():
                            logger.warning(f"[AI] Model {model_name} NOT FOUND (404). Skipping to next model.")
                            break # Go to next model in fallback_models
                        else:
                            logger.error(f"[AI] Unrecoverable Gemini API error: {api_err}")
                            return None
            
            if not success:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 5:
                    logger.error("!!! CIRCUIT BREAKER TRIGGERED !!! AI Service is unstable. Stopping analysis.")
                return None
            
            # Strict JSON parsing according to the new GEMINI.md schema
            try:
                analysis_data = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback / SKIP if parsing fails
                logger.error(f"Failed to decode GEMINI JSON. Falling back to SKIP. Output: {response.text}")
                return None

            # Log the full JSON and Reasoning as requested
            logger.info(f"AI JSON Response: {json.dumps(analysis_data, ensure_ascii=False)}")
            logger.info(f"AI Reasoning: {analysis_data.get('reasoning', 'No reasoning provided')}")

            edge_raw = analysis_data.get("edge", 0.0)
            confidence = analysis_data.get("confidence", 0)
            rec = analysis_data.get("recommended_action", "SKIP")
            
            target_outcome_name = "Yes" if rec == "BUY_YES" else "No"
            
            # Map to correct token_id and price
            matched_out = None
            for out in market_info["outcomes"]:
                if out["name"].lower() == target_outcome_name.lower():
                    matched_out = out
                    break
            
            if matched_out and matched_out["current_price"] >= 0.02:
                market_price = matched_out["current_price"]
                
                # Math Level 3: Fetch Self-Calibration Factor
                city = market_info.get("city", "default")
                calib_factor = calibration_engine.calculate_calibration_factor(city)
                
                # Apply Calibration Factor to Raw Probability
                raw_prob_yes = analysis_data.get("true_probability", 0.0)
                calibrated_prob_yes = min(0.99, max(0.01, raw_prob_yes * calib_factor))
                
                # Math Level 2: Calculate Expected Value (EV)
                if target_outcome_name == "Yes":
                    p = calibrated_prob_yes
                else:
                    p = 1.0 - calibrated_prob_yes
                
                # EV per $1 invested
                ev = (p * (1 - market_price)) - ((1 - p) * market_price)
                
                # Math Level 2: Fractional Kelly
                edge_decimal = p - market_price
                odds = (1 - market_price) / market_price if market_price > 0 else 0
                full_kelly = (edge_decimal / odds) if odds > 0 else 0
                fractional_kelly = full_kelly * config.kelly_fraction if full_kelly > 0 else 0.0
                
                # Get dynamic threshold
                ev_threshold = config.ev_threshold.get(city, config.ev_threshold.get("default", 0.08))
                
                # New Entry Rules: EV > Config API Threshold, Positive Fractional Kelly, Confidence >= 82
                if ev > ev_threshold and fractional_kelly > 0 and confidence >= 82 and "BUY" in rec:
                    return {
                        "market_id": market_info["market_id"],
                        "question": market_info.get("question", ""),
                        "token_id": matched_out["token_id"],
                        "icao_code": market_info.get("icao_code", ""),
                        "current_price": market_price,
                        "outcome_name": matched_out["name"],
                        "outcome_slug": target_outcome_name,
                        "edge": edge_decimal * 100, # Converting back to pct for UI compatibility
                        "ev": ev,
                        "kelly_frac": fractional_kelly,
                        "calibration_factor": calib_factor,
                        "true_probability": calibrated_prob_yes,
                        "confidence": confidence,
                        "recommendation": rec
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error during AI multi-source analysis for {market_info.get('event_title')}: {e}")
            return None
            logger.error(f"Error during AI analysis for {market_info.get('event_title')}: {e}")
            return None

ai_analyzer = AIAnalyzer()
