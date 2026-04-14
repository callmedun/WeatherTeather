import json
import os
import httpx
import re
from datetime import datetime
from typing import Dict, Any, List
from src.portfolio_manager import portfolio_manager
from src.utils import logger
from config.settings import config

class SelfCalibration:
    def __init__(self):
        self.filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical_predictions.jsonl")
        self.gamma_api = "https://gamma-api.polymarket.com"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', encoding='utf-8') as f:
                pass
                
    def save_prediction(self, market_id: str, outcome_token_id: str, city: str, icao: str, question: str, 
                        predicted_prob: float, bought_outcome: str, price_at_buy: float, ev: float = 0.0, size_usd: float = 0.0):
        try:
            record = {
                "market_id": market_id,
                "token_id": outcome_token_id,
                "city": city,
                "icao": icao,
                "question": question,
                "predicted_prob": predicted_prob,
                "ev": ev,
                "bought_outcome": bought_outcome,
                "price_at_buy": price_at_buy,
                "size_usd": size_usd,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "open",
                "actual_outcome": None
            }
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Recorded prediction for {city} (Market: {market_id})")
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")

    async def check_resolutions(self):
        records = []
        try:
            if not os.path.exists(self.filename):
                return
            with open(self.filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read calibration data: {e}")
            return

        updated_any = False
        async with httpx.AsyncClient() as client:
            for rec in records:
                if rec.get("status") == "open":
                    try:
                        url = f"{self.gamma_api}/markets?condition_id={rec['market_id']}"
                        r = await client.get(url, timeout=10.0)
                        if r.status_code == 200:
                            markets = r.json()
                            if markets and len(markets) > 0:
                                market = markets[0]
                                if market.get("closed") or not market.get("active"):
                                    # Identify outcome resolving
                                    tokens = market.get("clobTokenIds", "[]")
                                    if isinstance(tokens, str):
                                        tokens = json.loads(tokens)
                                    prices = market.get("outcomePrices", "[]")
                                    if isinstance(prices, str):
                                        prices = json.loads(prices)
                                    
                                    try:
                                        idx = tokens.index(rec['token_id'])
                                        price_final = float(prices[idx])
                                        if price_final >= 0.99:
                                            rec['actual_outcome'] = True
                                        else:
                                            # Validate if another outcome resolved to 1
                                            has_winner = any(float(p) > 0.99 for p in prices)
                                            if has_winner:
                                                rec['actual_outcome'] = False
                                            else:
                                                # Nullified or unresolved edge case
                                                rec['actual_outcome'] = False
                                                
                                        rec['status'] = "closed"
                                        updated_any = True
                                        logger.info(f"Resolved {rec['city']} market -> actual: {rec['actual_outcome']} (predicted {rec['predicted_prob']:3f})")
                                    except ValueError:
                                        pass
                    except Exception as e:
                        logger.error(f"Error checking gamma API for {rec['market_id']}: {e}")

        if updated_any:
            # Overwrite file with updated records
            with open(self.filename, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    def mark_trade_closed(self, token_id: str, actual_outcome=None, exit_price=None, realized_pnl=None):
        if not os.path.exists(self.filename):
            return
        records = []
        updated = False
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("token_id") == token_id and r.get("status") == "open":
                    r["status"] = "closed"
                    if actual_outcome is not None:
                        r["actual_outcome"] = actual_outcome
                    if exit_price is not None:
                        r["exit_price"] = exit_price
                    if realized_pnl is not None:
                        r["realized_pnl"] = realized_pnl
                    updated = True
                records.append(r)
        if updated:
            with open(self.filename, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    def calculate_calibration_factor(self, city: str) -> float:
        """
        Calculates the historical accuracy factor for a given city to adjust AI confidence natively.
        Requires at least config.calibration_min_trades closed records.
        Returns a factor scaled bounded between 0.85 and 1.15.
        """
        try:
            if not os.path.exists(self.filename):
                return 1.0
            closed_records = []
            with open(self.filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("status") == "closed" and r.get("city") == city and r.get("actual_outcome") is not None:
                            closed_records.append(r)
                            
            if len(closed_records) < config.calibration_min_trades:
                return 1.0
                
            # Compute actual hit rate for this city
            hits = sum(1 for r in closed_records if r["actual_outcome"] is True)
            actual_hit_rate = hits / len(closed_records)
            
            # Compute average predicted probability 
            avg_predicted_prob = sum(r["predicted_prob"] for r in closed_records) / len(closed_records)
            
            if avg_predicted_prob == 0:
                return 1.0
                
            factor = actual_hit_rate / avg_predicted_prob
            # Clamp limits strictly to avoid explosive leverage loops
            return max(0.85, min(1.15, factor))
            
        except Exception as e:
            logger.error(f"Error calculating calibration factor: {e}")
            return 1.0

    async def get_open_trades_async(self, clob_client=None) -> str:
        """Returns formatted string of open trades with live PnL."""
        if not os.path.exists(self.filename):
            return "📁 Нет активных сделок."
        
        lines = ["📂 ОТКРЫТЫЕ СДЕЛКИ:\n"]
        count = 0
        total_unrealized_pnl = 0.0
        
        # Read all records
        records = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("status") == "open":
                    records.append(r)
                    
        if not records:
            return "📁 Нет активных сделок."
            
        for r in records:
            count += 1
            size = r.get('size_usd', 0)
            entry_price = r.get('price_at_buy', 1.0)
            shares = size / entry_price if entry_price > 0 else 0
            
            # Fetch live cached price from Portfolio Manager using strictly PyClob methods
            current_price = portfolio_manager.get_current_price(clob_client, r['token_id'])
            
            if current_price is not None:
                current_value = shares * current_price
                entry_value = shares * entry_price
                pnl = current_value - entry_value
                pnl_str = f"{pnl:+.2f}$"
                
                predicted_prob = r.get('predicted_prob')
                if predicted_prob is not None:
                    new_edge = (predicted_prob - current_price) * 100
                    edge_str = f"| Edge: {new_edge:+.1f}%"
                else:
                    edge_str = ""
                    
                market_str = f"(рынок: {current_price:.3f} {edge_str})"
            else:
                pnl_str = "+0.00$"
                market_str = "(ошибка получения цены)"
                edge_str = ""
                
            # Extract Date and Temperature from question (fixed regex)
            q_text = r.get("question", "")
            date_match = re.search(r'on\s+([A-Za-z]+\s+\d+)', q_text)
            temp_match = re.search(r'be\s+(.*?)(?:\s+or\s+|\?$|$)', q_text)
            
            date_str = date_match.group(1) if date_match else "N/A"
            temp_str = temp_match.group(1).strip() if temp_match else "N/A"
            
            lines.append(f"• {r['city']} | {date_str} [{temp_str}] | {r['bought_outcome']}\n"
                         f"  ↳ Вход: {shares:.1f} shares @ {entry_price:.3f}\n"
                         f"  ↳ PnL: {pnl_str} {market_str}\n")
            if current_price is not None:
                total_unrealized_pnl += pnl

        lines.append(f"\n──────────────────\n📊 ОБЩИЙ РАСЧЕТНЫЙ PNL: {total_unrealized_pnl:+.2f}$")
        return "\n".join(lines)
        
    def get_open_trades(self) -> str:
        # Legacy fallback
        return "📁 Пожалуйста, используйте async метод."
        
    def get_closed_trades(self) -> str:
        """Returns formatted string of latest closed trades."""
        if not os.path.exists(self.filename):
            return "📜 История пуста."
            
        lines = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("status") == "closed":
                    outcome_icon = "✅" if r.get("actual_outcome") is True else "❌"
                    # If it was an early exit, it might not have actual_outcome set but has realized_pnl
                    if r.get("realized_pnl") is not None:
                        pnl = r["realized_pnl"]
                    else:
                        size = r.get('size_usd', 0)
                        price = r.get('price_at_buy', 1.0)
                        pnl = ((size / price) - size) if r.get("actual_outcome") is True else -size
                    
                    # Extract Date and Temperature (re-use logic)
                    q_text = r.get("question", "")
                    date_match = re.search(r'on\s+([A-Za-z]+\s+\d+)', q_text)
                    temp_match = re.search(r'be\s+(.*?)(?:\s+or\s+|\?$|$)', q_text)
                    date_str = date_match.group(1) if date_match else "N/A"
                    temp_str = temp_match.group(1).strip() if temp_match else "N/A"

                    lines.append(f"{outcome_icon} {r['city']} | {date_str} [{temp_str}] | {r['bought_outcome']} | PnL: {pnl:+.2f}$\n"
                                 f"  ↳ Вход: ${r.get('size_usd',0):.2f} @ {r.get('price_at_buy',0):.3f} | AI: {r.get('predicted_prob',0):.2f}")
        
        if not lines:
            return "📜 История пуста."
        return "📜 ПОСЛЕДНИЕ ЗАКРЫТЫЕ СДЕЛКИ (до 15):\n\n" + "\n\n".join(lines[-15:])

    def get_risk_summary(self) -> str:
        """Returns a string summary of current risk thresholds."""
        tp_edge = portfolio_manager.get_risk_setting("tp_edge", 10.0)
        strong_tp_pnl = portfolio_manager.get_risk_setting("strong_tp_pnl", 30.0)
        sl_edge = portfolio_manager.get_risk_setting("sl_edge", -12.0)
        sl_pnl = portfolio_manager.get_risk_setting("sl_pnl", -15.0)
        time_exit = portfolio_manager.get_risk_setting("time_exit_h", 6.0)

        return (
            "⚙️ ТЕКУЩИЕ НАСТРОЙКИ РИСКА:\n\n"
            f"🎯 Take Profit (Edge): <= {tp_edge}%\n"
            f"🚀 Strong TP (PnL): >= {strong_tp_pnl}%\n"
            f"📉 Stop Loss (Edge): <= {sl_edge}%\n"
            f"🚫 Stop Loss (PnL): <= {sl_pnl}%\n"
            f"⏳ Time Exit: < {time_exit} hours\n\n"
            "Используйте команды:\n"
            "/tp [value] - изменить порог TP Edge\n"
            "/stp [value] - изменить порог Strong TP PnL\n"
            "/sle [value] - изменить порог SL Edge\n"
            "/slp [value] - изменить порог SL PnL\n"
            "/time [value] - изменить время выхода (часы)"
        )

    def get_portfolio_stats(self, clob_client=None) -> str:
        """Returns general portfolio math based on historical records."""
        if not os.path.exists(self.filename):
            return "📊 Нет данных для статистики."
            
        total_trades = 0
        wins = 0
        losses = 0
        pnl = 0.0
        ev_sum = 0.0
        
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                total_trades += 1
                ev_sum += r.get("ev", 0.0)
                
                if r.get("status") == "closed":
                    trade_pnl = 0.0
                    if r.get("realized_pnl") is not None:
                        trade_pnl = r["realized_pnl"]
                    else:
                        size = r.get('size_usd', 0)
                        price = r.get('price_at_buy', 1.0)
                        if r.get("actual_outcome") is True:
                            trade_pnl = ((size / price) - size)
                        else:
                            trade_pnl = -size
                    
                    pnl += trade_pnl
                    if trade_pnl > 0:
                        wins += 1
                    elif trade_pnl < 0:
                        losses += 1
                    else:
                        # Breakeven - we can decide to count as loss or win, 
                        # but usually neutral trades are excluded or counted as losses.
                        # We'll leave it out of both counters to not skew WR.
                        pass
                        
        resolved = wins + losses
        win_rate = (wins / resolved * 100) if resolved > 0 else 0
        avg_ev = (ev_sum / total_trades) if total_trades > 0 else 0
        
        balance_str = "N/A (DRY_RUN)"
        if clob_client is not None and not config.dry_run:
            try:
                bal = clob_client.get_balance_allowance(asset_type="COLLATERAL")
                balance_str = f"${float(bal['balance']):.2f}" if isinstance(bal, dict) else str(bal)
            except Exception:
                pass

        if resolved == 0:
            return "📊 Статистика Баланса:\nПока слишком мало данных для PnL/Winrate (все сделки открыты)."

        return (
            f"📊 РАДАР СТАТИСТИКИ:\n\n"
            f"💰 Общий Баланс USDC: {balance_str}\n\n"
            f"📈 PNL Closed: {pnl:+.2f}$\n"
            f"🎯 Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)\n"
            f"🎲 Всего сделок: {total_trades}\n"
            f"🧠 Средний EV входа: {avg_ev:+.3f}\n"
        )

# Singleton
calibration_engine = SelfCalibration()
