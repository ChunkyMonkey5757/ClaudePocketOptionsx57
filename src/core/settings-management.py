async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /settings command - configure settings."""
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    
    # Build settings message
    assets_str = ", ".join(session["assets"])
    timeframes_str = ", ".join(f"{t}m" for t in session["timeframes"])
    
    settings_message = (
        f"⚙️ *Bot Settings* ⚙️\n\n"
        f"*Monitored Assets:* {assets_str}\n"
        f"*Timeframes:* {timeframes_str}\n"
        f"*Confidence Threshold:* {session['confidence_threshold']}%\n"
        f"*Cooldown:* {session['cooldown']}s\n\n"
        f"Select a setting to change:"
    )
    
    # Create settings keyboard
    keyboard = [
        [
            InlineKeyboardButton("Assets", callback_data="setting_assets"),
            InlineKeyboardButton("Timeframes", callback_data="setting_timeframes")
        ],
        [
            InlineKeyboardButton("Confidence", callback_data="setting_confidence"),
            InlineKeyboardButton("Cooldown", callback_data="setting_cooldown")
        ],
        [
            InlineKeyboardButton(
                f"Auto Rotation {'✅' if session.get('auto_rotate', False) else '❌'}",
                callback_data="setting_auto_rotate"
            ),
            InlineKeyboardButton(
                f"Shadow Mode {'✅' if session.get('shadow_mode', False) else '❌'}",
                callback_data="setting_shadow_mode"
            )
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        settings_message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )