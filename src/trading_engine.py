from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs

from config.settings import config
from src.utils import logger, send_telegram_message
from src.portfolio_manager import portfolio_manager
from src.calibration import calibration_engine
import asyncio

class TradingEngine:
    def __init__(self):
        if not config.polymarket_private_key or not config.funder_address:
            logger.warning("Polymarket credentials not set. Trading Engine disabled.")
            self.client = None
            return
            
        host = "https://clob.polymarket.com" if config.chain_id == 137 else "https://clob.amoy.polymarket.com"
        
        try:
            self.client = ClobClient(
                host=host,
                key=config.polymarket_private_key,
                chain_id=config.chain_id,
                funder=config.funder_address,
                signature_type=1 # Use EOA by default or 2 normally for PM proxy
            )
            # Create/derive API credentials if needed
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
            logger.success("Successfully authenticated PyClob client.")
        except Exception as e:
            logger.error(f"Failed to initialize PyClob API: {e}")
            self.client = None

    def _get_effective_fill(self, token_id: str, target_usd: float) -> tuple[float, float, float]:
        """Calculates Weighted Average Price (WAP) for a given USD size based on order book depth."""
        if not self.client:
            return 0.0, 0.0, 0.0
            
        try:
            book = self.client.get_order_book(token_id)
            asks = getattr(book, "asks", [])
            if not asks:
                return 0.0, 0.0, 0.0
                
            total_cost = 0.0
            total_shares = 0.0
            
            # Asks are usually sorted by price ascending
            logger.info(f"Analyzing depth for ${target_usd:.2f}...")
            for ask in asks:
                p = float(getattr(ask, 'price', 0.0))
                s = float(getattr(ask, 'size', 0.0))
                
                remaining_usd = target_usd - total_cost
                if remaining_usd <= 0:
                    break
                    
                level_max_cost = s * p
                if level_max_cost <= remaining_usd:
                    total_cost += level_max_cost
                    total_shares += s
                    logger.debug(f"  - Filled {s:.1f} shares @ {p:.3f}")
                else:
                    shares_needed = remaining_usd / p
                    total_cost += remaining_usd
                    total_shares += shares_needed
                    logger.debug(f"  - Filled {shares_needed:.1f}/{s:.1f} shares @ {p:.3f}")
                    break
            
            if total_shares == 0:
                return 0.0, 0.0, 0.0
                
            avg_price = total_cost / total_shares
            logger.info(f"Target: ${target_usd:.2f} | Execution: ${total_cost:.2f} | Avg Price: {avg_price:.3f}")
            return avg_price, total_shares, total_cost
        except Exception as e:
            logger.warning(f"Depth check failed for {token_id}: {e}")
            return 0.0, 0.0, 0.0

    async def execute_trade(self, analysis: dict):
        city = analysis.get("city", "Unknown")
        target_token = analysis.get("token_id")
        best_ask_price = analysis.get("market_price") # This is top-of-book from discovery
        true_prob = analysis.get("true_probability", 0.0)
        sentiment = analysis.get("sentiment", "NEUTRAL")
        
        logger.info(f"Preparing trade for {city} | Outcome: {analysis.get('outcome_slug')} | Top Ask: {best_ask_price}")

        # 1. Calculate Initial Kelly Sizing
        kelly_frac = analysis.get("kelly", 0.0)
        target_payout = 100.0 * kelly_frac
        initial_shares = target_payout / (1 - best_ask_price) if best_ask_price < 1 else 0
        intended_size = initial_shares * best_ask_price
        
        if intended_size < config.min_trade_usd:
            logger.info(f"Skipping trade: Kelly size too small (${intended_size:.2f})")
            return

        # 2. SMART DEPTH CHECK: Calculate effective price for our size
        if self.client:
            eff_price, eff_shares, filled_usd = self._get_effective_fill(target_token, intended_size)
        else:
            # Fallback if no client (should not happen in main flow)
            eff_price = best_ask_price * 1.005 
            eff_shares = intended_size / eff_price
            filled_usd = intended_size

        if eff_price == 0:
            logger.warning(f"Trade aborted: No liquidity found in order book for {city}")
            return

        # 3. SLIPPAGE FILTER: Re-calculate EV with effective price
        ev_threshold = config.ev_threshold.get(city, config.ev_threshold.get("default", 0.08))
        new_ev = (true_prob * (1 - eff_price)) - ((1 - true_prob) * eff_price)
        slippage = ((eff_price - best_ask_price) / best_ask_price) * 100 if best_ask_price > 0 else 0

        if new_ev < ev_threshold:
            # If slippage ruins the trade, try to reduce size to find a sweet spot
            logger.warning(f"Slippage too high ({slippage:.1f}%). New EV {new_ev:.3f} < {ev_threshold}. Attempting to scale down size...")
            # Try to buy only what's available at the top levels (limit to 50% of intended size)
            intended_size = intended_size * 0.5
            eff_price, eff_shares, filled_usd = self._get_effective_fill(target_token, intended_size)
            new_ev = (true_prob * (1 - eff_price)) - ((1 - true_prob) * eff_price)
            
            if new_ev < ev_threshold or filled_usd < config.min_trade_usd:
                logger.error(f"Trade aborted: Depth too thin for {city}. Even at ${filled_usd:.2f}, EV {new_ev:.3f} is below threshold.")
                return
            logger.info(f"Scaledown successful. Reduced size to ${intended_size:.2f} @ {eff_price:.3f}")

        # 4. Finalize execution params
        shares = int(max(config.min_shares, round(eff_shares)))
        final_cost = round(float(shares * eff_price), config.max_decimals_amount)
        profit_percent = round(((shares - final_cost) / final_cost) * 100, 2) if final_cost > 0 else 0
        
        # Check exposure limits
        if not portfolio_manager.can_trade_city(city, final_cost):
            logger.info("Skipping trade due to exposure limits.")
            return

        if config.dry_run:
            logger.info(f"[DRY RUN] Execute: {final_cost:.2f} USD ({shares} shares) | Price: {eff_price:.3f} | Slippage: {slippage:.1f}%")
            portfolio_manager.record_trade(analysis["market_id"], target_token, city, analysis.get("outcome_name"), eff_price, final_cost, sentiment)
            calibration_engine.save_prediction(analysis["market_id"], target_token, city, analysis.get("icao_code", ""), analysis.get("question", ""), true_prob, analysis.get("outcome_slug", ""), eff_price, new_ev, final_cost)
            
            await send_telegram_message(
                f"<b>[DRY RUN] Trade Executed</b>\n"
                f"<b>City:</b> {city}\n"
                f"<b>Question:</b> {analysis.get('question')}\n"
                f"<b>Outcome:</b> {analysis.get('outcome_name')}\n"
                f"<b>Shares Bought:</b> {shares}\n"
                f"<b>Price (Avg):</b> {eff_price:.3f} (Slip: {slippage:.1f}%)\n"
                f"<b>Invested:</b> ${final_cost}\n"
                f"<b>Kelly Fractional Size:</b> {kelly_frac:.3f}\n"
                f"<b>Potential Profit:</b> {profit_percent}%\n"
                f"<b>new EV:</b> {new_ev:.3f}\n"
                f"<b>AI Confidence:</b> {analysis.get('confidence')}/100"
            )
            return

        try:
            # Polymarket use LIMIT orders. We set the price to our calculated WAP or a bit higher to ensure fill
            order_args = OrderArgs(price=round(eff_price + 0.001, 3), size=final_cost, side="BUY", token_id=target_token)
            
            logger.info(f"Submitting LIVE order: {shares} shares @ {eff_price:.3f}")
            resp = self.client.create_and_post_order(order_args)
            
            if resp and resp.get("success"):
                logger.success(f"Trade successful! Order ID: {resp.get('orderID')}")
                portfolio_manager.record_trade(analysis["market_id"], target_token, city, analysis.get("outcome_name"), eff_price, final_cost, sentiment)
                calibration_engine.save_prediction(analysis["market_id"], target_token, city, analysis.get("icao_code", ""), analysis.get("question", ""), true_prob, analysis.get("outcome_slug", ""), eff_price, new_ev, final_cost)
                
                await send_telegram_message(
                    f"🟢 <b>LIVE Trade Executed</b>\n"
                    f"<b>City:</b> {city}\n"
                    f"<b>Question:</b> {analysis.get('question')}\n"
                    f"<b>Outcome:</b> {analysis.get('outcome_name')}\n"
                    f"<b>Shares Bought:</b> {shares}\n"
                    f"<b>Price (Avg):</b> {eff_price:.3f} (Slip: {slippage:.1f}%)\n"
                    f"<b>Invested:</b> ${final_cost}\n"
                    f"<b>Kelly Fractional Size:</b> {kelly_frac:.3f}\n"
                    f"<b>Potential Profit:</b> {profit_percent}%\n"
                    f"<b>new EV:</b> {new_ev:.3f}\n"
                    f"<b>AI Confidence:</b> {analysis.get('confidence')}/100"
                )
            else:
                logger.error(f"Failed to post order: {resp}")
        except Exception as e:
            logger.error(f"Trading exception: {e}")

trading_engine = TradingEngine()
