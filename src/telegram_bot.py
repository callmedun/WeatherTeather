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
        [InlineKeyboardButton("📊 Статистика", callback_data="stats"), InlineKeyboardButton("⚙️ Риски", callback_data="risk_settings")]
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
        elif query.data == "risk_settings":
            text = calibration_engine.get_risk_summary()
        else:
            text = "Неизвестная команда."

        # Limit text length to avoid Telegram 4096 chars error
        if len(text) > 4000:
            text = text[:4000] + "\n... [Обрезано из-за лимитов Telegram]"

        keyboard = [
            [InlineKeyboardButton("📂 Открытые сделки", callback_data="open_trades")],
            [InlineKeyboardButton("📜 Закрытые сделки", callback_data="closed_trades")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats"), InlineKeyboardButton("⚙️ Риски", callback_data="risk_settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(text=text, reply_markup=reply_markup)
    except Exception as e:
        if "Message is not modified" not in str(e):
            logger.error(f"Telegram menu error: {e}")

async def set_risk_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите значение. Пример: /tp 5")
        return
    
    cmd = update.message.text.split()[0].replace("/", "")
    try:
        val = float(context.args[0])
        from src.portfolio_manager import portfolio_manager
        
        mapping = {
            "tp": ("tp_edge", "Take Profit Edge"),
            "stp": ("strong_tp_pnl", "Strong TP PnL"),
            "sle": ("sl_edge", "Stop Loss Edge"),
            "slp": ("sl_pnl", "Stop Loss PnL"),
            "time": ("time_exit_h", "Time Exit (hours)")
        }
        
        if cmd in mapping:
            key, name = mapping[cmd]
            portfolio_manager.set_risk_setting(key, val)
            await update.message.reply_text(f"✅ Настройка {name} обновлена: {val}")
            logger.info(f"User updated risk setting {key} to {val}")
        else:
            await update.message.reply_text("Неизвестная команда настройки.")
    except ValueError:
        await update.message.reply_text("Ошибка: значение должно быть числом.")
    except Exception as e:
        logger.error(f"Error updating risk via TG: {e}")
        await update.message.reply_text("Произошла ошибка при обновлении настройки.")

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
        
        # Risk settings commands
        risk_cmds = ["tp", "stp", "sle", "slp", "time"]
        for c in risk_cmds:
            application.add_handler(CommandHandler(c, set_risk_threshold))
            
        application.add_handler(CallbackQueryHandler(button_callback))
        
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("Interactive Telegram Menu started successfully.")
    except Exception as e:
        logger.error(f"Failed to start telegram menu bot: {e}")
