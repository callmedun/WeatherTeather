import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from config.settings import config
from src.calibration import calibration_engine
from src.trading_engine import trading_engine
from src.utils import logger

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        ["📂 Открытые сделки", "📜 Закрытые сделки"],
        ["📊 Статистика", "⚙️ Риски"],
        ["▶️ Старт Бот", "⏸ Пауза Бот"],
        ["🛑 Закрыть все сделки"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Главное меню Полимаркет Бота:", reply_markup=reply_markup)

async def handle_menu_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_cmd = update.message.text
    
    try:
        if text_cmd == "📂 Открытые сделки":
            text = await calibration_engine.get_open_trades_async(trading_engine.client)
            await update.message.reply_text(text)
        elif text_cmd == "📜 Закрытые сделки":
            text = calibration_engine.get_closed_trades()
            await update.message.reply_text(text)
        elif text_cmd == "📊 Статистика":
            client = trading_engine.client
            text = calibration_engine.get_portfolio_stats(client)
            await update.message.reply_text(text)
        elif text_cmd == "⚙️ Риски":
            text = calibration_engine.get_risk_summary()
            await update.message.reply_text(text)
        elif text_cmd == "▶️ Старт Бот":
            was_paused = config.is_paused
            config.is_paused = False
            
            if was_paused:
                await update.message.reply_text("▶️ Бот запущен. Запускаю внеочередной цикл сканирования рынков...")
                logger.info("Bot execution resumed via Telegram. Triggering immediate scan_and_trade.")
                import asyncio
                from src.scheduler import bot_scheduler
                asyncio.create_task(bot_scheduler.scan_and_trade())
            else:
                await update.message.reply_text("▶️ Бот уже работает.")
        elif text_cmd == "⏸ Пауза Бот":
            config.is_paused = True
            await update.message.reply_text("⏸ Бот поставлен на паузу. Новые сделки не открываются (монитор продолжает работать).")
            logger.info("Bot execution paused via Telegram.")
        elif text_cmd == "🛑 Закрыть все сделки":
            # Show inline confirmation
            keyboard = [
                [
                    InlineKeyboardButton("🔥 ДА, ЗАКРЫТЬ ВСЕ", callback_data="confirm_liquidate"),
                    InlineKeyboardButton("Отмена", callback_data="cancel_liquidate")
                ]
            ]
            markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("⚠️ ОПАСНО!\nВы уверены, что хотите принудительно продать **все** открытые сделки по рыночным ценам (WAP)?", reply_markup=markup, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Telegram menu text error: {e}")
        await update.message.reply_text("Произошла ошибка при выполнении команды.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "confirm_liquidate":
        await query.edit_message_text("Начинаю экстренную ликвидацию всех позиций...")
        from src.portfolio_manager import portfolio_manager
        client = trading_engine.client
        await portfolio_manager.liquidate_all_trades(client)
        await context.bot.send_message(chat_id=query.message.chat_id, text="✅ Все открытые позиции были выставлены на продажу.")
        
    elif query.data == "cancel_liquidate":
        await query.edit_message_text("Отмена ликвидации. Продолжаем штатную работу.")

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
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_text))
        
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("Interactive Telegram Menu started successfully.")
    except Exception as e:
        logger.error(f"Failed to start telegram menu bot: {e}")
