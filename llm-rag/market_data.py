"""Market data and Alpaca API integration."""

from typing import Dict, List
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest, StockLatestQuoteRequest


class AlpacaClient:
    """A class to interact with Alpaca API for trading and market data."""
    
    def __init__(self, api_key, secret_key, paper=True):
        """Initialize Alpaca clients."""
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.market_data_client = StockHistoricalDataClient(api_key, secret_key)
        
    def get_user_account_info(self) -> Dict:
        """
        Get user account information from Alpaca API.

        Returns:
            A dictionary containing account details
        """
        try:
            account = self.trading_client.get_account()
            return {
                "account_id": account.id,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "status": account.status
            }
        except Exception as e:
            raise Exception(f"Error fetching account info: {str(e)}")

    def get_user_positions(self) -> List[Dict]:
        """
        Get user portfolio positions from Alpaca API.

        Returns:
            A list of dictionaries containing position details
        """
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "quantity": float(p.qty),
                    "market_value": float(p.market_value),
                    "cost_basis": float(p.cost_basis),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "current_price": float(p.current_price),
                    "lastday_price": float(p.lastday_price),
                    "change_today": float(p.change_today)
                }
                for p in positions
            ]
        except Exception as e:
            raise Exception(f"Error fetching positions: {str(e)}")

    def get_stock_price(self, ticker: str) -> Dict:
        """
        Get current stock price information from Alpaca API.

        Args:
            ticker: The stock ticker symbol

        Returns:
            A dictionary containing price and market data
        """
        ticker = ticker.upper()
        try:
            # get latest trade information
            trade_request = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trade_response = self.market_data_client.get_stock_latest_trade(trade_request)
            trade = trade_response[ticker]

            # get latest quote information (bid/ask)
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote_response = self.market_data_client.get_stock_latest_quote(quote_request)
            quote = quote_response[ticker]

            # get daily OHLCV data
            bars_request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.strptime("2025-03-01", '%Y-%m-%d'),
                limit=1
            )
            bars_response = self.market_data_client.get_stock_bars(bars_request)
            bar = bars_response[ticker][0]

            return {
                "symbol": ticker,
                "price": float(trade.price),
                "time": trade.timestamp.isoformat(),
                "ask_price": float(quote.ask_price),
                "ask_size": float(quote.ask_size),
                "bid_price": float(quote.bid_price),
                "bid_size": float(quote.bid_size),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "volume": float(bar.volume)
            }
        except Exception as e:
            raise Exception(f"Error fetching stock price for {ticker}: {str(e)}")