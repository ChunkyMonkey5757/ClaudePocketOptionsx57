async def assets_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /assets command - select assets to monitor."""
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    
    # Try to get available assets
    try:
        available_assets = await self.api_manager.get_available_assets()
        if not available_assets:
            available_assets = SUPPORTED_ASSETS
    except Exception as e:
        logger.error(f"Error fetching available assets: {str(e)}")
        available_assets = SUPPORTED_ASSETS
    
    # Create asset selection keyboard
    keyboard = []
    row = []
    
    for i, asset in enumerate(available_assets):
        # Check if asset is selected
        is_selected = asset in session["assets"]
        button_text = f"{asset} {'✅' if is_selected else ''}"
        
        row.append(InlineKeyboardButton(button_text, callback_data=f"asset_{asset}"))
        
        # Create new row every 3 buttons
        if len(row) == 3 or i == len(available_assets) - 1:
            keyboard.append(row)
            row = []
    
    # Add popular assets button and auto-rotate button
    keyboard.append([
        InlineKeyboardButton("📈 Popular Assets", callback_data="asset_popular")
    ])
    
    # Add done button
    keyboard.append([InlineKeyboardButton("✅ Done", callback_data="assets_done")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Build assets message
    assets_str = ", ".join(session["assets"])
    assets_message = (
        f"🎯 *Select Assets to Monitor* 🎯\n\n"
        f"Current assets: {assets_str}\n\n"
        f"Choose assets to monitor for trading signals:"
    )
    
    # Check if update is from callback or command
    if hasattr(update, 'callback_query') and update.callback_query:
        await update.callback_query.edit_message_text(
            assets_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            assets_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    return SELECTING_ASSETS

async def asset_selected_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle asset selection."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    data = query.data
    
    if data == "asset_popular":
        # Set popular assets
        popular_assets = ["BTC", "ETH", "SOL", "BNB", "XRP"]
        session["assets"] = popular_assets
    else:
        # Extract asset symbol
        asset = data.split("_")[1]
        
        # Toggle asset selection
        if asset in session["assets"]:
            session["assets"].remove(asset)
        else:
            session["assets"].append(asset)
    
    # Ensure at least one asset is selected
    if not session["assets"]:
        session["assets"] = ["BTC"]
    
    # Refresh assets command
    await self.assets_command(update, context)
    
    return SELECTING_ASSETS

async def assets_done_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle assets done selection."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    session = self._get_user_session(user_id)
    
    # Format assets list
    assets_str = ", ".join(session["assets"])
    
    await query.edit_message_text(
        f"✅ Assets updated successfully!\n\nNow monitoring: {assets_str}",
        parse_mode=ParseMode.MARKDOWN
    )
    
    return ConversationHandler.END

async def cancel_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the current conversation."""
    await update.message.reply_text(
        "Operation cancelled. Settings unchanged.",
        parse_mode=ParseMode.MARKDOWN
    )
    
    return ConversationHandler.END