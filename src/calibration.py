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
                
    def save_prediction(self, *args, **kwargs):
        """
        Saves a prediction record. 
        Can be called with a single dictionary 'sig' or with full positional arguments.
        """
        try:
            if len(args) == 1 and isinstance(args[0], dict):
                sig = args[0]
                record = {
                    "market_id": sig.get("market_id"),
                    "token_id": sig.get("token_id"),
                    "city": sig.get("city"),
                    "icao": sig.get("icao_code", sig.get("icao", "")),
                    "question": sig.get("question", ""),
                    "predicted_prob": sig.get("predicted_prob", 0.0),
                    "ev": sig.get("ev", 0.0),
                    "bought_outcome": sig.get("outcome_slug", sig.get("bought_outcome", "")),
                    "price_at_buy": sig.get("eff_price", sig.get("price_at_buy", 0.0)),
                    "size_usd": sig.get("final_cost", sig.get("size_usd", 0.0)),
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": sig.get("status", "open"),
                    "actual_outcome": sig.get("actual_outcome", None),
                    "outcomes": sig.get("outcomes", [])
                }
            else:
                # Fallback to positional (keeping original order for compatibility)
                # market_id, outcome_token_id, city, icao, question, predicted_prob, bought_outcome, price_at_buy, ev, size_usd
                record = {
                    "market_id": args[0] if len(args) > 0 else kwargs.get("market_id"),
                    "token_id": args[1] if len(args) > 1 else kwargs.get("outcome_token_id"),
                    "city": args[2] if len(args) > 2 else kwargs.get("city"),
                    "icao": args[3] if len(args) > 3 else kwargs.get("icao"),
                    "question": args[4] if len(args) > 4 else kwargs.get("question"),
                    "predicted_prob": args[5] if len(args) > 5 else kwargs.get("predicted_prob"),
                    "bought_outcome": args[6] if len(args) > 6 else kwargs.get("bought_outcome"),
                    "price_at_buy": args[7] if len(args) > 7 else kwargs.get("price_at_buy"),
                    "ev": args[8] if len(args) > 8 else kwargs.get("ev", 0.0),
                    "size_usd": args[9] if len(args) > 9 else kwargs.get("size_usd", 0.0),
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "open",
                    "actual_outcome": None
                }
            
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Recorded prediction update for {record['city']} (Market: {record['market_id']})")
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

    def update_prediction_prob(self, token_id: str, new_prob: float):
        if not os.path.exists(self.filename) or new_prob is None:
            return
        records = []
        updated = False
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("token_id") == token_id and r.get("status") == "open":
                    r["predicted_prob"] = new_prob
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
        
        # Read all records, deduplicating by token_id (keep highest size in case of corruption)
        records_dict = {}
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("status") == "open":
                    t_id = r.get("token_id")
                    if t_id not in records_dict or r.get("size_usd", 0) > records_dict[t_id].get("size_usd", 0):
                        records_dict[t_id] = r
                        
        records = list(records_dict.values())
                    
        if not records:
            return "📁 Нет активных сделок."
            
        # --- NEW: Batch Sync Prices once for the menu ---
        unique_tokens = list(set([r['token_id'] for r in records]))
        portfolio_manager.get_current_prices(clob_client, unique_tokens)
            
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
                    ai_prob_pct = predicted_prob * 100
                    edge_str = f"| Edge: {new_edge:+.1f}% | AI: {ai_prob_pct:.1f}%"
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
                         f"  ↳ Вход: {shares:.2f} shares @ {entry_price:.3f}\n"
                         f"  ↳ PnL: {pnl_str} {market_str}\n")
            if current_price is not None:
                total_unrealized_pnl += pnl

        lines.append(f"\n──────────────────\n📊 ОБЩИЙ РАСЧЕТНЫЙ PNL: {total_unrealized_pnl:+.2f}$")
        return "\n".join(lines)

    def get_open_trades(self) -> str:
        # Legacy fallback
        return "📁 Пожалуйста, используйте async метод."

    def get_closed_trades(self) -> str:
        """
        Returns the last 15 closed trades with REAL execution data.
        Entry price and size come from SQLite TradePosition (actual execution).
        Question text, PnL, and outcome come from the JSONL prediction file.
        """
        # 1. Load JSONL for text fields and realized_pnl
        jsonl_map: dict = {}
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        r = json.loads(line)
                        tid = r.get("token_id")
                        if not tid: continue
                        existing = jsonl_map.get(tid)
                        # Prefer records that have realized_pnl set
                        if existing is None or r.get("realized_pnl") is not None:
                            jsonl_map[tid] = r
                    except Exception:
                        pass

        # 2. Get real execution data from SQLite
        from src.portfolio_manager import TradePosition
        session = portfolio_manager.Session()
        try:
            db_trades = {
                t.token_id: t
                for t in session.query(TradePosition).filter(
                    TradePosition.status != "OPEN"
                ).all()
            }
        except Exception as e:
            logger.error(f"Error reading SQLite for closed trades: {e}")
            db_trades = {}
        finally:
            session.close()

        # 3. Merge: use all known token_ids from both sources
        all_tokens = list(set(list(jsonl_map.keys()) + list(db_trades.keys())))
        if not all_tokens:
            return "📜 История пуста."

        merged = []
        for tid in all_tokens:
            jrec   = jsonl_map.get(tid, {})
            db_obj = db_trades.get(tid)

            # Skip if still open everywhere
            if jrec.get("status") == "open" and db_obj is None:
                continue
            if jrec.get("status") == "open" and db_obj is not None and db_obj.status == "OPEN":
                continue
            if not jrec and db_obj is None:
                continue

            # Execution data: SQLite first (real execution), JSONL as fallback
            if db_obj and db_obj.entry_price and db_obj.entry_price > 0:
                price_at_buy = db_obj.entry_price
                size_usd     = db_obj.size_usd or 0.0
            else:
                price_at_buy = jrec.get("price_at_buy", 0.0) or 0.0
                size_usd     = jrec.get("size_usd", 0.0) or 0.0

            shares = size_usd / price_at_buy if price_at_buy > 0 else 0.0

            realized_pnl = jrec.get("realized_pnl")
            pnl = float(realized_pnl) if realized_pnl is not None else 0.0
            outcome_icon = "✅" if pnl > 0 else ("❌" if pnl < 0 else "⚪")

            q_text         = jrec.get("question", "")
            city           = jrec.get("city") or (db_obj.city if db_obj else "?")
            bought_outcome = jrec.get("bought_outcome") or (db_obj.outcome_name if db_obj else "?")

            date_match = re.search(r'on\s+([A-Za-z]+\s+\d+)', q_text)
            temp_match = re.search(r'be\s+(.*?)(?:\s+or\s+|\?$|$)', q_text)
            date_str   = date_match.group(1) if date_match else "?"
            temp_str   = temp_match.group(1).strip() if temp_match else q_text[:30]

            merged.append({
                "line": (
                    f"{outcome_icon} {city} | {date_str} [{temp_str}] | {bought_outcome} | PnL: {pnl:+.2f}$\n"
                    f"  ⤷ Вход: {size_usd:.2f}$ ({shares:.2f} sh) @ {price_at_buy:.3f}"
                ),
                "sort_key": jrec.get("timestamp", ""),
            })

        if not merged:
            return "📜 История пуста."

        merged.sort(key=lambda x: x["sort_key"])
        lines = [m["line"] for m in merged[-15:]]
        return "📜 ПОСЛЕДНИЕ ЗАКРЫТЫЕ СДЕЛКИ (до 15):\n\n" + "\n\n".join(lines)


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
        """Returns general portfolio stats based on historical records."""
        if not os.path.exists(self.filename):
            return "📊 Нет данных для статистики."

        # Deduplicate by token_id so repeated scan records don't inflate counts.
        # Last record per token_id represents the final state.
        token_records: dict = {}
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    if r.get("token_id"):
                        token_records[r["token_id"]] = r
                except Exception:
                    pass

        total_trades = len(token_records)
        wins = 0
        losses = 0
        pnl = 0.0
        ev_sum = 0.0

        for r in token_records.values():
            ev_sum += r.get("ev", 0.0)

            if r.get("status") == "closed":
                price = r.get('price_at_buy', 0)
                size = r.get('size_usd', 0)

                if r.get("realized_pnl") is not None:
                    trade_pnl = float(r["realized_pnl"])
                elif price and price > 0:
                    trade_pnl = ((size / price) - size) if r.get("actual_outcome") is True else -size
                else:
                    trade_pnl = 0.0

                pnl += trade_pnl
                if trade_pnl > 0:
                    wins += 1
                elif trade_pnl < 0:
                    losses += 1

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
            f"🎲 Всего сделок (unique): {total_trades}\n"
            f"🧠 Средний EV входа: {avg_ev:+.3f}\n"
        )

    def get_balance_summary(self, clob_client=None) -> str:
        """Returns account balance summary for Live and Dry Run modes."""
        base_dry_run_balance = 1000.0
        
        from src.portfolio_manager import portfolio_manager
        session = portfolio_manager.Session()
        
        try:
            from src.portfolio_manager import TradePosition
            
            total_realized_pnl = 0.0
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            r = json.loads(line)
                            if r.get("status") == "closed":
                                if r.get("realized_pnl") is not None:
                                    total_realized_pnl += float(r["realized_pnl"])
                                else:
                                    size = r.get('size_usd', 0)
                                    price = r.get('price_at_buy', 1.0)
                                    pnl = ((size / price) - size) if r.get("actual_outcome") is True else -float(size)
                                    total_realized_pnl += pnl
                        except: pass
            
            open_trades = session.query(TradePosition).filter_by(status="OPEN").all()
            total_exposure = sum([t.size_usd for t in open_trades])
            
            if config.dry_run:
                current_balance = base_dry_run_balance + total_realized_pnl
                free_cash = current_balance - total_exposure
                return (
                    f"💰 Виртуальный Баланс (DRY RUN):\n\n"
                    f"💳 Всего на счету: {current_balance:.2f} $\n"
                    f"🧊 Заморожено в сделках: {total_exposure:.2f} $\n"
                    f"💵 Свободный кэш: {free_cash:.2f} $\n\n"
                    f"📈 Общий PnL (закр.): {total_realized_pnl:+.2f} $"
                )
            else:
                balance_str = "Ошибка получения"
                free_cash_str = "Ошибка получения"
                if clob_client is not None:
                    try:
                        bal = clob_client.get_balance_allowance(asset_type="COLLATERAL")
                        if isinstance(bal, dict) and 'balance' in bal:
                            free_cash_str = f"{float(bal['balance']):.2f} $"
                            current_balance = float(bal['balance']) + total_exposure
                            balance_str = f"{current_balance:.2f} $"
                    except: pass
                
                return (
                    f"💰 Реальный Баланс (LIVE):\n\n"
                    f"💳 Оценочный эквити: ~{balance_str}\n"
                    f"🧊 В открытых позициях: {total_exposure:.2f} $\n"
                    f"💵 Доступно (USDC): {free_cash_str}\n\n"
                    f"📈 Зафиксированный PnL бота: {total_realized_pnl:+.2f} $"
                )
        except Exception as e:
            logger.error(f"Error calculating balance summary: {e}")
            return "Ошибка при расчете баланса."
        finally:
            session.close()

# Singleton
calibration_engine = SelfCalibration()
