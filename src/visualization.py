"""Data Visualization stage (CRISP-DM: Evaluation support).

Builds the full suite of required charts for a stock's processed data.
Each chart has a `build_*` function that returns the figure object
(reused live by the Streamlit dashboard) and a `plot_*` function that
builds and saves it under `reports/figures/`. Interactive charts use
Plotly (saved as standalone HTML); distribution/statistical charts use
Matplotlib (saved as PNG).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, safe for scripts and Streamlit

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

from src.config import FIGURES_DIR, MOVING_AVERAGE_WINDOWS
from src.utils import get_logger

logger = get_logger(__name__)


def _fig_path(ticker: str, name: str, ext: str) -> Path:
    return FIGURES_DIR / f"{ticker}_{name}.{ext}"


# ---------------------------------------------------------------------------
# Plotly (interactive) figure builders
# ---------------------------------------------------------------------------
def build_closing_price_trend(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive line chart of the Close price over time."""
    fig = px.line(df, x="Date", y="Close", title=f"{ticker} — Closing Price Trend")
    fig.update_layout(xaxis_title="Date", yaxis_title="Close Price")
    return fig


def build_trading_volume(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive bar chart of daily trading volume."""
    fig = px.bar(df, x="Date", y="Volume", title=f"{ticker} — Trading Volume")
    fig.update_layout(xaxis_title="Date", yaxis_title="Volume")
    return fig


def build_moving_average_comparison(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive overlay of Close price against its moving averages."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
    for window in MOVING_AVERAGE_WINDOWS:
        col = f"MA_{window}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=col))
    fig.update_layout(
        title=f"{ticker} — Moving Average Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return fig


def build_daily_returns(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive line chart of daily percentage returns."""
    fig = px.line(df, x="Date", y="Daily_Return", title=f"{ticker} — Daily Returns (%)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Daily Return (%)")
    return fig


def build_monthly_avg_closing_price(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive bar chart of average closing price per month."""
    monthly = (
        df.assign(Month=pd.to_datetime(df["Date"]).dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)["Close"]
        .mean()
    )
    fig = px.bar(
        monthly, x="Month", y="Close", title=f"{ticker} — Monthly Average Closing Price"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Average Close Price")
    return fig


def build_top_gain_loss_days(df: pd.DataFrame, ticker: str, top_n: int = 10) -> PlotlyFigure:
    """Interactive bar chart of the largest single-day gains and losses."""
    ranked = df.dropna(subset=["Daily_Return"]).sort_values("Daily_Return")
    losses = ranked.head(top_n)
    gains = ranked.tail(top_n)
    combined = pd.concat([losses, gains])

    fig = px.bar(
        combined,
        x="Date",
        y="Daily_Return",
        color=combined["Daily_Return"] > 0,
        color_discrete_map={True: "seagreen", False: "indianred"},
        title=f"{ticker} — Top {top_n} Gain and Loss Days",
        labels={"color": "Gain Day"},
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Daily Return (%)", showlegend=False)
    return fig


def build_candlestick(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive OHLC candlestick chart."""
    fig = go.Figure(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )
    fig.update_layout(
        title=f"{ticker} — Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_bollinger_bands(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive Close price with Bollinger Bands and the 20-day MA."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["BB_Upper"], name="Upper Band",
            line={"color": "rgba(173,216,230,0.7)"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["BB_Lower"], name="Lower Band",
            line={"color": "rgba(173,216,230,0.7)"},
            fill="tonexty", fillcolor="rgba(173,216,230,0.15)",
        )
    )
    if "MA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["MA_20"], name="MA_20",
                       line={"color": "orange", "dash": "dot"})
        )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], name="Close", line={"color": "steelblue"})
    )
    fig.update_layout(
        title=f"{ticker} — Bollinger Bands", xaxis_title="Date", yaxis_title="Price"
    )
    return fig


def build_rsi(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive RSI line with 30/70 overbought/oversold reference lines."""
    fig = px.line(df, x="Date", y="RSI", title=f"{ticker} — Relative Strength Index (RSI)")
    fig.add_hline(y=70, line_dash="dash", line_color="indianred",
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="seagreen",
                  annotation_text="Oversold (30)")
    fig.update_layout(xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100])
    return fig


def build_macd(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive MACD chart: MACD and signal lines plus the histogram."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram",
               marker_color="lightslategray")
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line={"color": "steelblue"})
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal",
                   line={"color": "darkorange"})
    )
    fig.update_layout(title=f"{ticker} — MACD", xaxis_title="Date", yaxis_title="MACD")
    return fig


def build_volatility_over_time(df: pd.DataFrame, ticker: str) -> PlotlyFigure:
    """Interactive line of rolling volatility, revealing volatility clustering."""
    fig = px.line(
        df, x="Date", y="Volatility", title=f"{ticker} — Rolling Volatility Over Time"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Volatility (std of returns)")
    return fig


# ---------------------------------------------------------------------------
# Matplotlib (static) figure builders
# ---------------------------------------------------------------------------
def build_returns_histogram(df: pd.DataFrame, ticker: str) -> Figure:
    """Static histogram of daily returns distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Daily_Return"].dropna(), bins=50, color="steelblue", edgecolor="white")
    ax.set_title(f"{ticker} — Daily Returns Distribution")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def build_returns_boxplot(df: pd.DataFrame, ticker: str) -> Figure:
    """Static box plot of daily returns, useful for spotting outliers."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(df["Daily_Return"].dropna(), vert=True, patch_artist=True)
    ax.set_title(f"{ticker} — Daily Returns Box Plot")
    ax.set_ylabel("Daily Return (%)")
    fig.tight_layout()
    return fig


def build_correlation_heatmap(df: pd.DataFrame, ticker: str) -> Figure:
    """Static heatmap of correlations between numeric features."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{ticker} — Correlation Heatmap")
    fig.tight_layout()
    return fig


def build_autocorrelation(df: pd.DataFrame, ticker: str, lags: int = 40) -> Figure:
    """Static bar chart of daily-return autocorrelation across `lags` lags.

    Uses pandas' built-in `Series.autocorr`, so no extra dependency is
    needed. Near-zero autocorrelation across lags is the empirical
    signature of the (weak-form) efficient-market hypothesis.
    """
    returns = df["Daily_Return"].dropna()
    acf_values = [returns.autocorr(lag=lag) for lag in range(1, lags + 1)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, lags + 1), acf_values, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{ticker} — Daily Returns Autocorrelation")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save wrappers (used by the batch EDA pipeline / main.py)
# ---------------------------------------------------------------------------
def plot_closing_price_trend(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_closing_price_trend(df, ticker)
    path = _fig_path(ticker, "closing_price_trend", "html")
    fig.write_html(path)
    return path


def plot_trading_volume(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_trading_volume(df, ticker)
    path = _fig_path(ticker, "trading_volume", "html")
    fig.write_html(path)
    return path


def plot_moving_average_comparison(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_moving_average_comparison(df, ticker)
    path = _fig_path(ticker, "moving_average_comparison", "html")
    fig.write_html(path)
    return path


def plot_daily_returns(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_daily_returns(df, ticker)
    path = _fig_path(ticker, "daily_returns", "html")
    fig.write_html(path)
    return path


def plot_returns_histogram(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_returns_histogram(df, ticker)
    path = _fig_path(ticker, "returns_histogram", "png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_returns_boxplot(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_returns_boxplot(df, ticker)
    path = _fig_path(ticker, "returns_boxplot", "png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_correlation_heatmap(df, ticker)
    path = _fig_path(ticker, "correlation_heatmap", "png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_monthly_avg_closing_price(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_monthly_avg_closing_price(df, ticker)
    path = _fig_path(ticker, "monthly_avg_closing_price", "html")
    fig.write_html(path)
    return path


def plot_top_gain_loss_days(df: pd.DataFrame, ticker: str, top_n: int = 10) -> Path:
    fig = build_top_gain_loss_days(df, ticker, top_n=top_n)
    path = _fig_path(ticker, "top_gain_loss_days", "html")
    fig.write_html(path)
    return path


def plot_candlestick(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_candlestick(df, ticker)
    path = _fig_path(ticker, "candlestick", "html")
    fig.write_html(path)
    return path


def plot_bollinger_bands(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_bollinger_bands(df, ticker)
    path = _fig_path(ticker, "bollinger_bands", "html")
    fig.write_html(path)
    return path


def plot_rsi(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_rsi(df, ticker)
    path = _fig_path(ticker, "rsi", "html")
    fig.write_html(path)
    return path


def plot_macd(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_macd(df, ticker)
    path = _fig_path(ticker, "macd", "html")
    fig.write_html(path)
    return path


def plot_volatility_over_time(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_volatility_over_time(df, ticker)
    path = _fig_path(ticker, "volatility_over_time", "html")
    fig.write_html(path)
    return path


def plot_autocorrelation(df: pd.DataFrame, ticker: str) -> Path:
    fig = build_autocorrelation(df, ticker)
    path = _fig_path(ticker, "autocorrelation", "png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_figures(df: pd.DataFrame, ticker: str) -> dict[str, Path]:
    """Generate and save the complete set of required charts for `ticker`.

    Returns:
        A mapping of figure name to the saved file path.
    """
    logger.info("Generating figures for %s", ticker)
    figures = {
        "closing_price_trend": plot_closing_price_trend(df, ticker),
        "trading_volume": plot_trading_volume(df, ticker),
        "moving_average_comparison": plot_moving_average_comparison(df, ticker),
        "daily_returns": plot_daily_returns(df, ticker),
        "returns_histogram": plot_returns_histogram(df, ticker),
        "returns_boxplot": plot_returns_boxplot(df, ticker),
        "correlation_heatmap": plot_correlation_heatmap(df, ticker),
        "monthly_avg_closing_price": plot_monthly_avg_closing_price(df, ticker),
        "top_gain_loss_days": plot_top_gain_loss_days(df, ticker),
        "candlestick": plot_candlestick(df, ticker),
        "bollinger_bands": plot_bollinger_bands(df, ticker),
        "rsi": plot_rsi(df, ticker),
        "macd": plot_macd(df, ticker),
        "volatility_over_time": plot_volatility_over_time(df, ticker),
        "autocorrelation": plot_autocorrelation(df, ticker),
    }
    logger.info("Generated %d figures in %s", len(figures), FIGURES_DIR)
    return figures
