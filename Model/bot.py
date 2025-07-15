from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
    CommandHandler,
)
from final import recommend
from final import TMDbAPI, get_watch_providers
from final import parse_query_with_gemini
import logging
from final import INTRO_PHRASES, NO_RESULTS_PHRASES, FINAL_RECS_PHRASES
from final import get_recommendations

# Enable logging (helpful to see errors)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Replace with your actual bot token
BOT_TOKEN = "Your Token"

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎬 Welcome to MovieBot!\n"
        "Send me a message like:\n"
        "- Recommend action movies\n"
        "- Show me movies like Inception\n"
        "- Find movies with Brad Pitt\n"
        "- Christopher Nolan movies on Netflix"
    )

# Message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    await update.message.chat.send_action("typing")

    try:
        parsed = parse_query_with_gemini(user_input)
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        await update.message.reply_text("⚠️ Sorry, there was an error understanding your request.")
        return

    if not parsed.get("intent"):
        await update.message.reply_text("⚠️ Sorry, I couldn't understand your request.")
        return

    try:
        recs = get_recommendations(parsed)
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        await update.message.reply_text("⚠️ Sorry, there was an error fetching recommendations.")
        return

    if not recs:
        await update.message.reply_text("⚠️ No recommendations found.")
        return

    response_lines = []
    for idx, r in enumerate(recs, 1):
        providers = ", ".join(r["providers"]) if r["providers"] else "No streaming info"
        line = (
            f"{idx}. 🎬 *{r['title']}*\n"
            f"_Plot:_ {r['overview'][:200]}...\n"
            f"_Available on:_ {providers}\n"
        )
        response_lines.append(line)

    response_text = "\n\n".join(response_lines)

    await update.message.reply_text(response_text, parse_mode="Markdown")

# Main entry point
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started.")
    app.run_polling()

if __name__ == "__main__":
    main()
