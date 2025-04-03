async def settings_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle settings buttons."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    data = query.data
    
    # Toggle boolean settings
    if data == "setting_auto_rotate":
        session["auto_rotate"] = not session.get("auto_rotate", False)
        await self.settings_command(update, context)
        return
    
    elif data == "setting_shadow_mode":
        session["shadow_mode"] = not session.get("shadow_mode", False)
        await self.settings_command(update, context)
        return
    
    # Handle other settings
    if data == "setting_confidence":
        await query.edit_message_text(
            "Enter new confidence threshold (50-95):",
            parse_mode=ParseMode.MARKDOWN
        )
        return SETTING_CONFIDENCE
    
    elif data == "setting_cooldown":
        await query.edit_message_text(
            "Enter new cooldown in seconds (60-300):",
            parse_mode=ParseMode.MARKDOWN
        )
        return SETTING_COOLDOWN
    
    elif data == "setting_timeframes":
        # Create timeframe selection keyboard
        keyboard = []
        row = []
        
        for tf in SUPPORTED_TIMEFRAMES:
            # Check if timeframe is already selected
            is_selected = tf in session.get("timeframes", [1])
            button_text = f"{tf}m {'✅' if is_selected else ''}"
            
            row.append(InlineKeyboardButton(button_text, callback_data=f"timeframe_{tf}"))
            
            # Create new row every 3 buttons
            if len(row) == 3:
                keyboard.append(row)
                row = []
        
        # Add remaining buttons in last row
        if row:
            keyboard.append(row)
        
        # Add done button
        keyboard.append([InlineKeyboardButton("Done", callback_data="timeframe_done")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "Select timeframes to monitor:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        return SETTING_TIMEFRAME
    
    elif data == "setting_assets":
        # Redirect to assets command
        await self.assets_command(update, context)
        return SELECTING_ASSETS

async def confidence_input_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle confidence threshold input."""
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    
    try:
        # Parse confidence value
        confidence = int(update.message.text.strip())
        
        # Validate range
        if confidence < 50 or confidence > 95:
            await update.message.reply_text(
                "⚠️ Invalid value. Confidence must be between 50 and 95.",
                parse_mode=ParseMode.MARKDOWN
            )
            return SETTING_CONFIDENCE
        
        # Update setting
        session["confidence_threshold"] = confidence
        
        await update.message.reply_text(
            f"✅ Confidence threshold updated to {confidence}%.",
            parse_mode=ParseMode.MARKDOWN
        )
        
        return ConversationHandler.END
        
    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid value. Please enter a number.",
            parse_mode=ParseMode.MARKDOWN
        )
        return SETTING_CONFIDENCE

async def cooldown_input_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle cooldown input."""
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    
    try:
        # Parse cooldown value
        cooldown = int(update.message.text.strip())
        
        # Validate range
        if cooldown < 60 or cooldown > 300:
            await update.message.reply_text(
                "⚠️ Invalid value. Cooldown must be between 60 and 300 seconds.",
                parse_mode=ParseMode.MARKDOWN
            )
            return SETTING_COOLDOWN
        
        # Update setting
        session["cooldown"] = cooldown
        
        await update.message.reply_text(
            f"✅ Cooldown updated to {cooldown} seconds.",
            parse_mode=ParseMode.MARKDOWN
        )
        
        return ConversationHandler.END
        
    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid value. Please enter a number.",
            parse_mode=ParseMode.MARKDOWN
        )
        return SETTING_COOLDOWN

async def timeframe_selected_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle timeframe selection."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    data = query.data
    
    if data == "timeframe_done":
        # Make sure at least one timeframe is selected
        if not session.get("timeframes"):
            session["timeframes"] = [1]  # Default to 1m
        
        await query.edit_message_text(
            f"✅ Timeframes updated: {', '.join(f'{t}m' for t in session['timeframes'])}",
            parse_mode=ParseMode.MARKDOWN
        )
        return ConversationHandler.END
    
    # Extract timeframe value
    timeframe = int(data.split("_")[1])
    
    # Toggle timeframe selection
    if "timeframes" not in session:
        session["timeframes"] = []
    
    if timeframe in session["timeframes"]:
        session["timeframes"].remove(timeframe)
    else:
        session["timeframes"].append(timeframe)
        session["timeframes"].sort()
    
    # Rebuild timeframe keyboard
    keyboard = []
    row = []
    
    for tf in SUPPORTED_TIMEFRAMES:
        # Check if timeframe is already selected
        is_selected = tf in session.get("timeframes", [1])
        button_text = f"{tf}m {'✅' if is_selected else ''}"
        
        row.append(InlineKeyboardButton(button_text, callback_data=f"timeframe_{tf}"))
        
        # Create new row every 3 buttons
        if len(row) == 3:
            keyboard.append(row)
            row = []
    
    # Add remaining buttons in last row
    if row:
        keyboard.append(row)
    
    # Add done button
    keyboard.append([InlineKeyboardButton("Done", callback_data="timeframe_done")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "Select timeframes to monitor:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )
    return SETTING_TIMEFRAME