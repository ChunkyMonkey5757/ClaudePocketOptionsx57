"""
Visualization utilities for PocketBotX57.
Provides functions for creating charts and graphs.
"""

import io
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, List, Optional, Union

def create_signal_chart(market_data: Dict[str, Any], signal: Dict[str, Any]) -> io.BytesIO:
    """
    Create a chart visualizing a trading signal.
    
    Args:
        market_data: Market data dictionary
        signal: Signal dictionary
    
    Returns:
        BytesIO object with the chart image
    """
    # Extract OHLC data
    if "ohlc" not in market_data or not market_data["ohlc"]:
        raise ValueError("Market data missing OHLC values")
    
    # Convert to DataFrame
    df = pd.DataFrame(market_data["ohlc"])
    
    # Limit to last 30 candles for readability
    df = df.iloc[-30:]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    # Plot candlesticks
    for i, (_, row) in enumerate(df.iterrows()):
        # Candle color (green for up, red for down)
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Plot candle body
        plt.bar(
            i, 
            row['close'] - row['open'], 
            bottom=min(row['open'], row['close']),
            color=color,
            width=0.8,
            alpha=0.7
        )
        
        # Plot candle wicks
        plt.plot(
            [i, i], 
            [row['low'], row['high']], 
            color=color,
            linewidth=1
        )
    
    # Extract indicators from signal
    indicators = signal.get("indicators", {})
    
    # Add support and resistance lines if available
    if "support_levels" in indicators:
        for level in indicators["support_levels"]:
            plt.axhline(y=level, color='lightgreen', linestyle='--', alpha=0.7)
    
    if "resistance_levels" in indicators:
        for level in indicators["resistance_levels"]:
            plt.axhline(y=level, color='lightcoral', linestyle='--', alpha=0.7)
    
    # Add moving averages if available
    if "sma_20" in indicators:
        plt.plot(
            range(len(df)), 
            indicators["sma_20"][-len(df):], 
            color='cyan', 
            linestyle='-', 
            linewidth=1,
            label='SMA 20'
        )
    
    if "sma_50" in indicators:
        plt.plot(
            range(len(df)), 
            indicators["sma_50"][-len(df):], 
            color='magenta', 
            linestyle='-', 
            linewidth=1,
            label='SMA 50'
        )
    
    # Add RSI if available
    if "rsi" in indicators:
        # Create a small subplot for RSI
        rsi_height = 0.2  # 20% of the plot height
        ax1 = plt.gca()
        ax2 = plt.gcf().add_axes(
            [ax1.get_position().x0, ax1.get_position().y0 - rsi_height, 
             ax1.get_position().width, rsi_height]
        )
        
        # Plot RSI
        rsi_values = indicators["rsi"][-len(df):]
        ax2.plot(range(len(df)), rsi_values, color='yellow', label='RSI')
        
        # Add RSI reference lines
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        
        # Set RSI y-axis limits and label
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI')
        ax2.grid(True, alpha=0.3)
    
    # Add signal direction indicator
    direction = signal.get("direction", "NEUTRAL")
    last_idx = len(df) - 1
    last_price = df['close'].iloc[-1]
    
    if direction == "BUY":
        plt.annotate(
            '↑ BUY', 
            xy=(last_idx, last_price),
            xytext=(last_idx, last_price * 0.97),
            color='lime',
            fontsize=14,
            fontweight='bold',
            arrowprops=dict(facecolor='lime', shrink=0.05),
            horizontalalignment='center'
        )
    elif direction == "SELL":
        plt.annotate(
            '↓ SELL', 
            xy=(last_idx, last_price),
            xytext=(last_idx, last_price * 1.03),
            color='red',
            fontsize=14,
            fontweight='bold',
            arrowprops=dict(facecolor='red', shrink=0.05),
            horizontalalignment='center'
        )
    
    # Add confidence level
    confidence = signal.get("confidence", 0)
    plt.title(
        f"{market_data.get('symbol', 'Unknown')}USD - {direction} Signal ({confidence:.1f}% confidence)",
        fontsize=16
    )
    
    # Add timeframe indicator
    timeframe = signal.get("timeframe", 1)
    plt.figtext(
        0.02, 0.02, 
        f"Timeframe: {timeframe}m", 
        color='white', 
        backgroundcolor='black', 
        alpha=0.7
    )
    
    # Style adjustments
    plt.grid(True, alpha=0.2)
    plt.xticks([])  # Hide x-axis values for cleaner look
    plt.ylabel('Price', fontsize=12)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Save chart to buffer
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return buffer

def create_performance_chart(session: Dict[str, Any]) -> io.BytesIO:
    """
    Create a performance chart based on session data.
    
    Args:
        session: User session dictionary
        
    Returns:
        BytesIO object with the chart image
    """
    # Extract signals with feedback
    signals = [s for s in session.get("signals", []) if "feedback" in s]
    
    if not signals:
        raise ValueError("No feedback data available for chart")
    
    # Sort signals by timestamp
    signals.sort(key=lambda s: s.get("timestamp", ""))
    
    # Initialize data
    dates = []
    cumulative_wins = []
    win_rate = []
    win_count = 0
    total_count = 0
    
    # Process signals
    for signal in signals:
        # Extract date from timestamp
        try:
            date = datetime.datetime.fromisoformat(signal.get("timestamp", ""))
            dates.append(date)
        except ValueError:
            # Skip signals with invalid timestamps
            continue
        
        # Update counts
        total_count += 1
        if signal["feedback"] == "win":
            win_count += 1
        
        # Calculate cumulative metrics
        cumulative_wins.append(win_count)
        win_rate.append((win_count / total_count) * 100 if total_count > 0 else 0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    plt.style.use('dark_background')
    
    # Plot 1: Cumulative wins
    ax1.plot(dates, cumulative_wins, 'g-', linewidth=2, label='Cumulative Wins')
    ax1.set_title('Trading Performance', fontsize=16)
    ax1.set_ylabel('Wins', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Win rate percentage
    ax2.plot(dates, win_rate, 'c-', linewidth=2, label='Win Rate %')
    ax2.set_ylabel('Win Rate %', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='yellow', linestyle='--', alpha=0.5, label='50% Threshold')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    # Add final win rate
    final_win_rate = win_rate[-1] if win_rate else 0
    total_signals = len(signals)
    
    ax1.annotate(
        f"Win Rate: {final_win_rate:.1f}% ({win_count}/{total_signals})",
        xy=(0.02, 0.95),
        xycoords='axes fraction',
        color='white',
        backgroundcolor='black',
        alpha=0.7
    )
    
    # Calculate asset performance
    asset_performance = {}
    for signal in signals:
        symbol = signal.get("symbol", "UNKNOWN")
        if symbol not in asset_performance:
            asset_performance[symbol] = {"wins": 0, "total": 0}
        
        asset_performance[symbol]["total"] += 1
        if signal["feedback"] == "win":
            asset_performance[symbol]["wins"] += 1
    
    # Add asset performance table
    asset_text = "Asset Win Rates:\n"
    for symbol, perf in asset_performance.items():
        if perf["total"] > 0:
            asset_win_rate = (perf["wins"] / perf["total"]) * 100
            asset_text += f"{symbol}: {asset_win_rate:.1f}% ({perf['wins']}/{perf['total']})\n"
    
    plt.figtext(
        0.02, 0.02,
        asset_text,
        fontsize=9,
        color='white',
        backgroundcolor='black',
        alpha=0.7
    )
    
    # Save chart to buffer
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return buffer