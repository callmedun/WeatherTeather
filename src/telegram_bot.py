from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

from config.settings import config
from src.calibration import calibration_engine
from src.trading_engine import trading_engine
from src.utils import logger


MENU_OPEN = "📂 Открытые сделки"
MENU_CLOSED = "📜 Закрытые сделки"
MENU_BALANCE = "💰 Баланс"
MENU_STATS = "📊 Статистика"
MENU_STATUS = "⚙️ Статус"
MENU_RISK = "⚙️ Риски"
MENU_START = "▶️ Старт Бот"
MENU_PAUSE = "⏸ Пауза Бот"
MENU_LIQUIDATE = "🛑 Закрыть все сделки"
# Phase 6 additions
MENU_PAPER  = "📋 Paper Trader"
MENU_SKILL  = "🧠 Model Skill"
MENU_ERRORS = "🔴 Ошибки"
MENU_RESET  = "🔄 Reset CB"


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [MENU_OPEN, MENU_CLOSED],
        [MENU_BALANCE, MENU_STATS],
        [MENU_STATUS, MENU_RISK],
        [MENU_PAPER, MENU_SKILL],
        [MENU_ERRORS, MENU_RESET],
        [MENU_START, MENU_PAUSE],
        [MENU_LIQUIDATE],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Главное меню Polymarket-бота v2:", reply_markup=reply_markup)


async def safe_reply(update: Update, text: str):
    if len(text) <= 4000:
        await update.message.reply_text(text)
        return

    parts = []
    current_part = ""
    for line in text.split("\n"):
        if len(current_part) + len(line) + 1 > 4000:
            parts.append(current_part)
            current_part = line + "\n"
        else:
            current_part += line + "\n"

    if current_part:
        parts.append(current_part)

    for part in parts:
        if part.strip():
            await update.message.reply_text(part)


async def handle_menu_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_cmd = update.message.text

    try:
        if text_cmd == MENU_OPEN:
            text = await calibration_engine.get_open_trades_async(trading_engine.client)
            await safe_reply(update, text)
        elif text_cmd == MENU_CLOSED:
            text = calibration_engine.get_closed_trades()
            await safe_reply(update, text)
        elif text_cmd == MENU_BALANCE:
            text = calibration_engine.get_balance_summary(trading_engine.client)
            await safe_reply(update, text)
        elif text_cmd == MENU_STATS:
            text = calibration_engine.get_portfolio_stats(trading_engine.client)
            await safe_reply(update, text)
        elif text_cmd == MENU_STATUS:
            from src.scheduler import bot_scheduler

            text = bot_scheduler.get_system_status()
            await safe_reply(update, text)
        elif text_cmd == MENU_RISK:
            text = calibration_engine.get_risk_summary()
            await safe_reply(update, text)
        elif text_cmd == MENU_START:
            was_paused = config.is_paused
            config.is_paused = False

            if was_paused:
                await update.message.reply_text(
                    "▶️ Бот запущен. Запускаю внеочередной цикл сканирования рынков..."
                )
                logger.info("Bot execution resumed via Telegram. Triggering immediate scan_and_trade.")
                import asyncio
                from src.scheduler import bot_scheduler

                asyncio.create_task(bot_scheduler.scan_and_trade())
            else:
                await update.message.reply_text("▶️ Бот уже работает.")
        elif text_cmd == MENU_PAUSE:
            config.is_paused = True
            await update.message.reply_text(
                "⏸ Бот поставлен на паузу. Новые сделки не открываются, монитор продолжает работать."
            )
            logger.info("Bot execution paused via Telegram.")
        elif text_cmd == MENU_LIQUIDATE:
            keyboard = [[
                InlineKeyboardButton("🔥 ДА, ЗАКРЫТЬ ВСЕ", callback_data="confirm_liquidate"),
                InlineKeyboardButton("Отмена", callback_data="cancel_liquidate"),
            ]]
            markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "⚠️ ОПАСНО!\nВы уверены, что хотите принудительно продать **все** открытые сделки по рыночным ценам (WAP)?",
                reply_markup=markup,
                parse_mode="Markdown",
            )

        elif text_cmd == MENU_PAPER:
            try:
                from src.backtest.paper_trader import paper_trader
                text = paper_trader.get_report(days=30)
            except Exception as e:
                text = f"Paper Trader недоступен: {e}"
            await safe_reply(update, text)

        elif text_cmd == MENU_SKILL:
            try:
                from src.model_skill_tracker import model_skill_tracker
                text = model_skill_tracker.get_skill_report()
            except Exception as e:
                text = f"Skill Tracker недоступен: {e}"
            await safe_reply(update, text)

        elif text_cmd == MENU_ERRORS:
            try:
                from src.health_monitor import health_monitor
                text = health_monitor.get_recent_errors(10)
            except Exception as e:
                text = f"Health Monitor недоступен: {e}"
            await safe_reply(update, text)

        elif text_cmd == MENU_RESET:
            try:
                from src.scheduler import bot_scheduler
                text = bot_scheduler.reset_circuit()
            except Exception as e:
                text = f"Ошибка сброса Circuit Breaker: {e}"
            await safe_reply(update, text)

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
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="✅ Все открытые позиции были выставлены на продажу.",
        )
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
            "time": ("time_exit_h", "Time Exit (hours)"),
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


async def paper_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """[Phase 6] /paper — paper trader full metrics."""
    try:
        from src.backtest.paper_trader import paper_trader
        days = int(context.args[0]) if context.args else 30
        text = paper_trader.get_full_metrics_report(days=days)
    except Exception as e:
        text = f"Paper Trader ошибка: {e}"
    await safe_reply(update, text)


async def skill_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """[Phase 6] /skill [ICAO] — model skill scores."""
    try:
        from src.model_skill_tracker import model_skill_tracker
        icao = context.args[0].upper() if context.args else None
        text = model_skill_tracker.get_skill_report(icao=icao)
    except Exception as e:
        text = f"Skill Tracker ошибка: {e}"
    await safe_reply(update, text)


async def errors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """[Phase 6] /errors — recent error log."""
    try:
        from src.health_monitor import health_monitor
        n = int(context.args[0]) if context.args else 10
        text = health_monitor.get_recent_errors(n)
    except Exception as e:
        text = f"Health Monitor ошибка: {e}"
    await safe_reply(update, text)


async def reset_cb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """[Phase 6] /reset [circuit_name] — reset circuit breaker."""
    try:
        from src.scheduler import bot_scheduler
        name = context.args[0] if context.args else None
        text = bot_scheduler.reset_circuit(name)
    except Exception as e:
        text = f"Reset ошибка: {e}"
    await safe_reply(update, text)


application = None


async def start_telegram_bot():
    global application
    if not getattr(config, "telegram_menu_enabled", False) or not getattr(config, "telegram_bot_token", ""):
        logger.warning("Telegram Interactive Menu disabled. (Token missing or config off)")
        return

    try:
        application = ApplicationBuilder().token(config.telegram_bot_token).build()
        application.add_handler(CommandHandler("start", menu_command))
        application.add_handler(CommandHandler("menu", menu_command))

        for command in ["tp", "stp", "sle", "slp", "time"]:
            application.add_handler(CommandHandler(command, set_risk_threshold))

        # [Phase 6] new commands
        application.add_handler(CommandHandler("paper",  paper_command))
        application.add_handler(CommandHandler("skill",  skill_command))
        application.add_handler(CommandHandler("errors", errors_command))
        application.add_handler(CommandHandler("reset",  reset_cb_command))

        application.add_handler(CallbackQueryHandler(button_callback))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_text))

        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("Interactive Telegram Menu started successfully (Phase 6).")
    except Exception as e:
        logger.error(f"Failed to start telegram menu bot: {e}")
