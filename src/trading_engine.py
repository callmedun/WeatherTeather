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

    async def execute_trade(self, analysis: dict):
        city = analysis.get("city", "Unknown")
        target_token = analysis.get("token_id")
        price = analysis.get("market_price")
        sentiment = analysis.get("sentiment", "NEUTRAL")
        
        logger.info(f"Preparing trade for {city} | Sentiment: {sentiment} | Outcome: {analysis.get('outcome_slug')} | Expected Price: {price}")

        # Match Fractional Kelly sizing from AI Analysis
        kelly_frac = analysis.get("kelly", 0.0)
        
        # Target payout scales linearly with fractional Kelly. 
        target_payout = 100.0 * kelly_frac
        
        # Calculate ideal constraints
        shares_to_buy = target_payout / (1 - price) if price < 1 else 0
        cost = shares_to_buy * price
        
        # 1. Reject microscopic trades to avoid gas overhead
        if cost < config.min_trade_usd:
            logger.info(f"Skipping trade: Too small after Kelly (${cost:.2f} < ${config.min_trade_usd})")
            return
            
        # 2. Polymarket ClobClient API strictly enforces NO fractional shares and decimals clamping
        shares_to_buy = int(max(config.min_shares, round(shares_to_buy)))
        cost = round(float(shares_to_buy * price), config.max_decimals_amount)
        
        # Re-check after rounding just in case of weird pricing rounding below threshold
        if cost < config.min_trade_usd:
            logger.info(f"Skipping trade: Too small after Kelly rounding (${cost:.2f} < ${config.min_trade_usd})")
            return
            
        intended_size = cost
        shares = shares_to_buy
        profit_percent = round(((shares - intended_size) / intended_size) * 100, 2) if intended_size > 0 else 0
        
        # Check exposure limits
        if not portfolio_manager.can_trade_city(city, intended_size):
            logger.info("Skipping trade due to exposure limits.")
            return

        if config.dry_run:
            logger.info(f"[DRY RUN] Would execute BUY of {intended_size:.2f} USD ({shares} shares) on token {target_token} at ~{price}")
            # Still record the paper trade
            portfolio_manager.record_trade(
                market_id=analysis["market_id"],
                token_id=target_token,
                city=city,
                outcome=analysis.get("outcome_name"),
                price=price,
                size=intended_size,
                sentiment=sentiment
            )
            calibration_engine.save_prediction(
                market_id=analysis["market_id"],
                outcome_token_id=target_token,
                city=city,
                icao=analysis.get("icao_code", ""),
                question=analysis.get("question", ""),
                predicted_prob=analysis.get("true_probability", 0.0), # Stored dynamically from AI payload
                bought_outcome=analysis.get("outcome_slug", ""),
                price_at_buy=price,
                size_usd=intended_size,
                ev=analysis.get("ev", 0.0)
            )
            await send_telegram_message(
                f"<b>[DRY RUN] Trade Executed</b>\n"
                f"<b>City:</b> {city}\n"
                f"<b>Question:</b> {analysis.get('question')}\n"
                f"<b>Outcome:</b> {analysis.get('outcome_name')}\n"
                f"<b>Shares Bought:</b> {shares}\n"
                f"<b>Invested:</b> ${intended_size}\n"
                f"<b>Kelly Fractional Size:</b> {kelly_frac:.3f}\n"
                f"<b>Potential Profit:</b> {profit_percent}%\n"
                f"<b>AI Confidence:</b> {analysis.get('confidence')}/100"
            )
            return

        if self.client is None:
            logger.error("Client is not initialized. Cannot execute real trade.")
            return
            
        try:
            # Per official docs: For BUY, the exact USD amount is passed. For SELL, shares are passed.
            order_args = OrderArgs(
                price=price,
                size=intended_size,
                side="BUY",
                token_id=target_token
            )
            
            logger.info(f"Submitting LIVE order: {shares} shares @ {price}")
            resp = self.client.create_and_post_order(order_args)
            
            if resp and resp.get("success"):
                logger.success(f"Trade successful! Order ID: {resp.get('orderID')}")
                portfolio_manager.record_trade(
                    market_id=analysis["market_id"],
                    token_id=target_token,
                    city=city,
                    outcome=analysis.get("outcome_name"),
                    price=price,
                    size=intended_size,
                    sentiment=sentiment
                )
                calibration_engine.save_prediction(
                    market_id=analysis["market_id"],
                    outcome_token_id=target_token,
                    city=city,
                    icao=analysis.get("icao_code", ""),
                    question=analysis.get("question", ""),
                    predicted_prob=analysis.get("true_probability", 0.0),
                    bought_outcome=analysis.get("outcome_slug", ""),
                    price_at_buy=price,
                    size_usd=intended_size,
                    ev=analysis.get("ev", 0.0)
                )
                await send_telegram_message(
                    f"🟢 <b>LIVE Trade Executed</b>\n"
                    f"<b>City:</b> {city}\n"
                    f"<b>Question:</b> {analysis.get('question')}\n"
                    f"<b>Outcome:</b> {analysis.get('outcome_name')}\n"
                    f"<b>Shares Bought:</b> {shares}\n"
                    f"<b>Invested:</b> ${intended_size}\n"
                    f"<b>Kelly Fractional Size:</b> {kelly_frac:.3f}\n"
                    f"<b>Potential Profit:</b> {profit_percent}%\n"
                    f"<b>AI Confidence:</b> {analysis.get('confidence')}/100"
                )
            else:
                logger.error(f"Failed to post order: {resp}")
        except Exception as e:
            logger.error(f"Trading exception: {e}")

trading_engine = TradingEngine()
