import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from config.settings import config
from src.calibration import calibration_engine
from src.trading_engine import trading_engine
from src.utils import logger

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📂 Открытые сделки", callback_data="open_trades")],
        [InlineKeyboardButton("📜 Закрытые сделки", callback_data="closed_trades")],
        [InlineKeyboardButton("📊 Баланс и статистика", callback_data="stats")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Главное меню Полимаркет Бота:", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    try:
        if query.data == "open_trades":
            text = await calibration_engine.get_open_trades_async(trading_engine.client)
        elif query.data == "closed_trades":
            text = calibration_engine.get_closed_trades()
        elif query.data == "stats":
            client = trading_engine.client
            text = calibration_engine.get_portfolio_stats(client)
        else:
            text = "Неизвестная команда."

        # Limit text length to avoid Telegram 4096 chars error
        if len(text) > 4000:
            text = text[:4000] + "\n... [Обрезано из-за лимитов Telegram]"

        keyboard = [
            [InlineKeyboardButton("📂 Открытые сделки", callback_data="open_trades")],
            [InlineKeyboardButton("📜 Закрытые сделки", callback_data="closed_trades")],
            [InlineKeyboardButton("📊 Баланс и статистика", callback_data="stats")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(text=text, reply_markup=reply_markup)
    except Exception as e:
        if "Message is not modified" not in str(e):
            logger.error(f"Telegram menu error: {e}")

application = None

async def start_telegram_bot():
    global application
    if not getattr(config, 'telegram_menu_enabled', False) or not getattr(config, 'telegram_bot_token', ''):
        logger.warning("Telegram Interactive Menu disabled. (Token missing or config off)")
        return
        
    try:
        # We must build with loop intercept if running
        application = ApplicationBuilder().token(config.telegram_bot_token).build()
        application.add_handler(CommandHandler("start", menu_command))
        application.add_handler(CommandHandler("menu", menu_command))
        application.add_handler(CallbackQueryHandler(button_callback))
        
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("Interactive Telegram Menu started successfully.")
    except Exception as e:
        logger.error(f"Failed to start telegram menu bot: {e}")
