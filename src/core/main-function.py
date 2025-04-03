def main() -> None:
    """Main function to run the bot."""
    # Initialize and start the bot
    bot = TelegramBot(TELEGRAM_BOT_TOKEN)
    
    try:
        logger.info("Starting PocketBotX57 Telegram bot")
        bot.start()
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
    finally:
        # Ensure proper cleanup
        asyncio.run(bot.stop())

if __name__ == '__main__':
    main()