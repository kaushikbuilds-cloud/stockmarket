"""Streamlit dashboard for the Stock Market Analysis and Price
Prediction project.

Pages:
    Dashboard         - key metrics, closing price snapshot, CSV export
    Stock Analysis    - interactive OHLCV charts with date-range filtering
    EDA               - descriptive statistics and distribution charts
    Prediction        - future closing price forecast
    Model Performance - model comparison table and diagnostic plots

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.bulk_analysis import is_offline_dataset_available, load_bulk_summary
from src.config import (
    DATE_COLUMN,
    DEFAULT_PERIOD_YEARS,
    DEFAULT_TICKER,
    EXAMPLE_TICKERS,
    PROCESSED_DATA_DIR,
)
from src.data_collection import collect_and_save
from src.eda import (
    correlation_matrix,
    monthly_average_closing_price,
    summary_statistics,
    top_gain_days,
    top_loss_days,
)
from src.evaluate import evaluate_model
from src.feature_engineering import engineer_and_save
from src.predict import predict_future
from src.preprocessing import preprocess_and_save
from src.train_model import train_and_select_best_model
from src.utils import get_logger, load_dataframe, processed_csv_path, validate_ticker
from src.visualization import (
    build_autocorrelation,
    build_bollinger_bands,
    build_candlestick,
    build_closing_price_trend,
    build_correlation_heatmap,
    build_daily_returns,
    build_macd,
    build_moving_average_comparison,
    build_monthly_avg_closing_price,
    build_returns_boxplot,
    build_returns_histogram,
    build_rsi,
    build_top_gain_loss_days,
    build_trading_volume,
    build_volatility_over_time,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="Stock Market Analysis & Price Prediction",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached pipeline steps
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pipeline_data(ticker: str, years: int, _nonce: int, source: str = "yfinance") -> pd.DataFrame:
    """Run collection -> preprocessing -> feature engineering for `ticker`.

    `_nonce` is a cache-busting key: bump it (e.g. via a Refresh button)
    to force a fresh download instead of reusing the cached result.
    `source` selects live yfinance data, the offline bulk dataset, or
    yfinance with an automatic offline fallback (see
    `src.data_collection.collect_and_save`).
    """
    collect_and_save(ticker=ticker, years=years, source=source)
    preprocess_and_save(ticker)
    engineer_and_save(ticker)
    return load_dataframe(
        processed_csv_path(PROCESSED_DATA_DIR, ticker), parse_dates=[DATE_COLUMN]
    )


@st.cache_resource(show_spinner=False)
def load_trained_model(ticker: str, _nonce: int):
    """Train all candidate models for `ticker` and evaluate the best one."""
    training_result = train_and_select_best_model(ticker)
    evaluation = evaluate_model(training_result)
    return training_result, evaluation


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
if "data_nonce" not in st.session_state:
    st.session_state.data_nonce = 0

st.sidebar.title("📈 Controls")

ticker_input = st.sidebar.text_input(
    "Stock Ticker",
    value=DEFAULT_TICKER,
    help="Examples: " + ", ".join(EXAMPLE_TICKERS),
)

years_input = st.sidebar.slider(
    "Years of History", min_value=1, max_value=10, value=DEFAULT_PERIOD_YEARS
)

_source_labels = {
    "yfinance (live)": "yfinance",
    "offline (bulk dataset, ends 2020-04-01)": "offline",
    "auto (yfinance, offline fallback)": "auto",
}
source_label = st.sidebar.selectbox(
    "Data Source",
    list(_source_labels.keys()),
    help="Offline mode reads the local Kaggle bulk dataset instead of "
    "calling the network -- data is frozen at 2020-04-01.",
)
source_choice = _source_labels[source_label]

fetch_clicked = st.sidebar.button("🔄 Fetch / Refresh Data", width='stretch')
if fetch_clicked:
    st.session_state.data_nonce += 1
    load_trained_model.clear()

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Stock Analysis",
        "Technical Indicators",
        "EDA",
        "Prediction",
        "Model Performance",
        "Compare Tickers",
        "Market Overview",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: Yahoo Finance (yfinance), with an offline bulk-dataset "
    "option. Educational project — not financial advice."
)

# ---------------------------------------------------------------------------
# Page: Market Overview (offline bulk dataset -- ticker-independent, so it
# runs before the per-ticker pipeline below and exits early).
# ---------------------------------------------------------------------------
if page == "Market Overview":
    st.title("🗂️ Market Overview — Offline Bulk Dataset")
    st.caption(
        "Cross-sectional view of the local Kaggle 'jacksoncrow/"
        "stock-market-dataset' snapshot (~5,900 stocks / ~2,000 ETFs, "
        "history ending 2020-04-01). Built once via "
        "`python -m src.bulk_analysis --build`, not recomputed live."
    )

    if not is_offline_dataset_available():
        st.warning(
            "Offline bulk dataset not found under `data/stocks/`. "
            "Download it (e.g. via `kagglehub`) into the project's "
            "`data/` folder to use this page."
        )
        st.stop()

    kind = st.radio("Universe", ["stocks", "etfs"], horizontal=True)
    try:
        summary = load_bulk_summary(kind)
    except FileNotFoundError as exc:
        st.warning(str(exc))
        st.stop()

    valid = summary.dropna(subset=["CAGR_Pct"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers Covered", f"{len(summary):,}")
    col2.metric("Median CAGR", f"{valid['CAGR_Pct'].median():.1f}%")
    col3.metric("Median Volatility", f"{summary['Annualized_Volatility_Pct'].median():.1f}%")
    col4.metric("Latest Data Through", str(summary["Last_Date"].max().date()))

    import plotly.express as px

    cagr_lo, cagr_hi = valid["CAGR_Pct"].quantile([0.02, 0.98])
    st.subheader("Distribution of Annualized Returns (CAGR)")
    hist_fig = px.histogram(
        valid[valid["CAGR_Pct"].between(cagr_lo, cagr_hi)],
        x="CAGR_Pct",
        nbins=60,
        title=f"CAGR Distribution across {kind} (middle 96%, outliers clipped for readability)",
    )
    hist_fig.update_layout(xaxis_title="CAGR (%)", yaxis_title="Number of Tickers")
    st.plotly_chart(hist_fig, width='stretch')

    # A handful of penny stocks (price near MIN_VALID_PRICE) swing by
    # huge percentages day to day, which blows up their annualized
    # volatility into the millions of percent and stretches the whole
    # chart's axis. Clip volatility to its 98th percentile too, on top
    # of the existing CAGR clip, so the scatter is actually readable.
    vol_hi = valid["Annualized_Volatility_Pct"].quantile(0.98)
    scatter_data = valid[
        valid["CAGR_Pct"].between(cagr_lo, cagr_hi)
        & (valid["Annualized_Volatility_Pct"] <= vol_hi)
    ]

    st.subheader("Risk vs. Return")
    scatter_fig = px.scatter(
        scatter_data,
        x="Annualized_Volatility_Pct",
        y="CAGR_Pct",
        hover_name="Symbol",
        hover_data=["Security_Name", "Years_Held"],
        opacity=0.4,
        title=f"Volatility vs. CAGR across {kind} (98th-percentile outliers clipped)",
    )
    scatter_fig.update_layout(xaxis_title="Annualized Volatility (%)", yaxis_title="CAGR (%)")
    st.plotly_chart(scatter_fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 15 by CAGR** (min. 3 years held)")
        top_gainers = valid[valid["Years_Held"] >= 3].nlargest(15, "CAGR_Pct")
        st.dataframe(
            top_gainers[["Symbol", "Security_Name", "CAGR_Pct", "Years_Held", "Annualized_Volatility_Pct"]],
            width='stretch',
        )
    with col_b:
        st.markdown("**Bottom 15 by CAGR** (min. 3 years held)")
        top_losers = valid[valid["Years_Held"] >= 3].nsmallest(15, "CAGR_Pct")
        st.dataframe(
            top_losers[["Symbol", "Security_Name", "CAGR_Pct", "Years_Held", "Annualized_Volatility_Pct"]],
            width='stretch',
        )

    st.subheader("Search")
    query = st.text_input("Filter by symbol or company name")
    display_df = summary
    if query:
        q = query.strip().lower()
        display_df = summary[
            summary["Symbol"].str.lower().str.contains(q)
            | summary["Security_Name"].fillna("").str.lower().str.contains(q)
        ]
    st.dataframe(display_df, width='stretch')
    st.download_button(
        "⬇️ Download Full Summary CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"bulk_summary_{kind}.csv",
        mime="text/csv",
        width='stretch',
    )
    st.stop()


# ---------------------------------------------------------------------------
# Load data (shared across all pages)
# ---------------------------------------------------------------------------
try:
    ticker = validate_ticker(ticker_input)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    with st.spinner(f"Loading data for {ticker}..."):
        df = load_pipeline_data(
            ticker, years_input, st.session_state.data_nonce, source=source_choice
        )
except (ValueError, FileNotFoundError) as exc:
    st.error(f"Could not load data for '{ticker}': {exc}")
    st.stop()

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
min_date, max_date = df[DATE_COLUMN].min().date(), df[DATE_COLUMN].max().date()

st.sidebar.subheader("Date Range Filter")
date_range = st.sidebar.date_input(
    "Chart display range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    range_start, range_end = date_range
else:
    range_start, range_end = min_date, max_date

mask = (df[DATE_COLUMN].dt.date >= range_start) & (df[DATE_COLUMN].dt.date <= range_end)
filtered_df = df.loc[mask].reset_index(drop=True)
if filtered_df.empty:
    st.warning("No data in the selected date range. Showing full history instead.")
    filtered_df = df


def download_button(data: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width='stretch',
    )


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.title(f"📈 {ticker} — Stock Market Dashboard")

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    change = latest["Close"] - previous["Close"]
    pct_change = (change / previous["Close"] * 100) if previous["Close"] else 0.0
    year_slice = df[df[DATE_COLUMN] >= df[DATE_COLUMN].max() - pd.Timedelta(days=365)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"{latest['Close']:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    col2.metric("52-Week High", f"{year_slice['Close'].max():.2f}")
    col3.metric("52-Week Low", f"{year_slice['Close'].min():.2f}")
    col4.metric("Latest Volume", f"{int(latest['Volume']):,}")

    st.plotly_chart(
        build_closing_price_trend(filtered_df, ticker), width='stretch'
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(build_trading_volume(filtered_df, ticker), width='stretch')
    with col_b:
        st.plotly_chart(
            build_moving_average_comparison(filtered_df, ticker), width='stretch'
        )

    st.subheader("Processed Data")
    st.dataframe(filtered_df.tail(20), width='stretch')
    download_button(filtered_df, "⬇️ Download Processed CSV", f"{ticker}_processed.csv")


# ---------------------------------------------------------------------------
# Page: Stock Analysis
# ---------------------------------------------------------------------------
elif page == "Stock Analysis":
    st.title(f"🔍 {ticker} — Stock Analysis")

    st.plotly_chart(
        build_closing_price_trend(filtered_df, ticker), width='stretch'
    )
    st.plotly_chart(build_trading_volume(filtered_df, ticker), width='stretch')
    st.plotly_chart(
        build_moving_average_comparison(filtered_df, ticker), width='stretch'
    )
    st.plotly_chart(build_daily_returns(filtered_df, ticker), width='stretch')

    st.subheader("Raw Feature Data")
    st.dataframe(filtered_df, width='stretch')
    download_button(filtered_df, "⬇️ Download Filtered CSV", f"{ticker}_filtered.csv")


# ---------------------------------------------------------------------------
# Page: Technical Indicators
# ---------------------------------------------------------------------------
elif page == "Technical Indicators":
    st.title(f"📉 {ticker} — Technical Indicators")
    st.caption(
        "Candlestick price action with common momentum and volatility "
        "indicators used in technical analysis."
    )

    st.subheader("Candlestick Chart")
    st.plotly_chart(build_candlestick(filtered_df, ticker), width='stretch')

    st.subheader("Bollinger Bands")
    st.plotly_chart(build_bollinger_bands(filtered_df, ticker), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("RSI")
        st.plotly_chart(build_rsi(filtered_df, ticker), width='stretch')
    with col_b:
        st.subheader("MACD")
        st.plotly_chart(build_macd(filtered_df, ticker), width='stretch')


# ---------------------------------------------------------------------------
# Page: EDA
# ---------------------------------------------------------------------------
elif page == "EDA":
    st.title(f"📊 {ticker} — Exploratory Data Analysis")

    st.subheader("Summary Statistics")
    st.dataframe(summary_statistics(filtered_df), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(build_returns_histogram(filtered_df, ticker))
    with col_b:
        st.pyplot(build_returns_boxplot(filtered_df, ticker))

    st.subheader("Correlation Heatmap")
    st.pyplot(build_correlation_heatmap(filtered_df, ticker))

    st.subheader("Monthly Average Closing Price")
    st.plotly_chart(
        build_monthly_avg_closing_price(filtered_df, ticker), width='stretch'
    )

    st.subheader("Top Gain and Loss Days")
    st.plotly_chart(
        build_top_gain_loss_days(filtered_df, ticker), width='stretch'
    )
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Highest Gain Days**")
        st.dataframe(top_gain_days(filtered_df), width='stretch')
    with col_d:
        st.markdown("**Highest Loss Days**")
        st.dataframe(top_loss_days(filtered_df), width='stretch')

    st.subheader("Rolling Volatility Over Time")
    st.caption(
        "Periods of high volatility tend to cluster together — a "
        "well-known stylized fact of financial time series."
    )
    st.plotly_chart(build_volatility_over_time(filtered_df, ticker), width='stretch')

    st.subheader("Daily Returns Autocorrelation")
    st.caption(
        "Near-zero autocorrelation across lags is the empirical "
        "signature of a weak-form efficient market."
    )
    st.pyplot(build_autocorrelation(filtered_df, ticker))

    st.subheader("Correlation Matrix")
    st.dataframe(correlation_matrix(filtered_df), width='stretch')


# ---------------------------------------------------------------------------
# Page: Prediction
# ---------------------------------------------------------------------------
elif page == "Prediction":
    st.title(f"🔮 {ticker} — Price Prediction")
    st.caption(
        "Forecasts use the best model trained on historical features. "
        "Multi-day forecasts iteratively simulate forward and are an "
        "educational approximation, not a guaranteed price target."
    )

    forecast_days = st.slider("Days to Forecast", min_value=1, max_value=30, value=5)
    predict_clicked = st.button("🔮 Predict Future Prices", width='stretch')

    if predict_clicked:
        try:
            with st.spinner("Training model and forecasting..."):
                training_result, _ = load_trained_model(ticker, st.session_state.data_nonce)
                forecast = predict_future(ticker, days=forecast_days)
        except (ValueError, FileNotFoundError) as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            latest_close = df.iloc[-1]["Close"]
            next_close = forecast.iloc[0]["Predicted_Close"]
            delta = next_close - latest_close

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Close", f"{latest_close:.2f}")
            col2.metric(
                "Next Predicted Close", f"{next_close:.2f}", f"{delta:+.2f}"
            )
            col3.metric("Model Used", training_result.best_model_name)

            history_tail = df[[DATE_COLUMN, "Close"]].tail(60).rename(
                columns={"Close": "Price"}
            )
            history_tail["Series"] = "Actual"
            forecast_series = forecast.rename(
                columns={"Predicted_Close": "Price"}
            )
            forecast_series["Series"] = "Forecast"
            combined = pd.concat(
                [history_tail, forecast_series[[DATE_COLUMN, "Price", "Series"]]]
            )

            import plotly.express as px

            fig = px.line(
                combined,
                x=DATE_COLUMN,
                y="Price",
                color="Series",
                title=f"{ticker} — Recent History and {forecast_days}-Day Forecast",
            )
            st.plotly_chart(fig, width='stretch')

            st.subheader("Forecast Table")
            st.dataframe(forecast, width='stretch')
            download_button(forecast, "⬇️ Download Forecast CSV", f"{ticker}_forecast.csv")
    else:
        st.info("Set the forecast horizon and click **Predict Future Prices**.")


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title(f"🧪 {ticker} — Model Performance")

    try:
        with st.spinner("Training and evaluating models..."):
            training_result, evaluation = load_trained_model(
                ticker, st.session_state.data_nonce
            )
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

    metrics_table = evaluation["metrics_table"]
    best_name = training_result.best_model_name

    col1, col2, col3, col4 = st.columns(4)
    best_row = metrics_table.loc[best_name]
    col1.metric("Best Model", best_name)
    col2.metric("R² Score", f"{best_row['R2']:.4f}")
    col3.metric("RMSE", f"{best_row['RMSE']:.2f}")
    col4.metric("MAE", f"{best_row['MAE']:.2f}")

    st.subheader("Model Comparison")
    st.dataframe(
        metrics_table.style.highlight_max(subset=["R2"], color="lightgreen").highlight_min(
            subset=["MAE", "MSE", "RMSE"], color="lightgreen"
        ),
        width='stretch',
    )

    tuned_params = {
        name: params
        for name, params in training_result.best_params.items()
        if params
    }
    if tuned_params:
        st.subheader("Tuned Hyperparameters")
        st.caption(
            "Selected via GridSearchCV over a TimeSeriesSplit "
            "cross-validation (no shuffling)."
        )
        st.json(tuned_params)

    st.subheader("Diagnostic Plots")
    figures = evaluation["figures"]
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(figures["actual_vs_predicted"]), width='stretch')
        st.image(str(figures["prediction_error"]), width='stretch')
    with col_b:
        st.image(str(figures["residual_plot"]), width='stretch')

    download_button(
        metrics_table.reset_index(), "⬇️ Download Metrics CSV", f"{ticker}_metrics.csv"
    )


# ---------------------------------------------------------------------------
# Page: Compare Tickers
# ---------------------------------------------------------------------------
elif page == "Compare Tickers":
    st.title("⚖️ Compare Tickers")
    st.caption(
        "Overlay several stocks (indexed to 100 at the start of the "
        "selected date range) and compare their returns, volatility, "
        "and best model. Each new ticker may trigger a fresh download "
        "and model training on first view."
    )

    default_selection = [t for t in (DEFAULT_TICKER, "MSFT", "GOOG") if t in EXAMPLE_TICKERS]
    selected = st.multiselect(
        "Tickers to compare",
        options=EXAMPLE_TICKERS,
        default=default_selection or EXAMPLE_TICKERS[:2],
        help="Pick two or more tickers from the example list.",
    )

    if len(selected) < 2:
        st.info("Select at least two tickers to compare.")
    else:
        import plotly.express as px

        normalized_frames = []
        summary_rows = []

        for cmp_ticker in selected:
            try:
                with st.spinner(f"Loading {cmp_ticker}..."):
                    cmp_df = load_pipeline_data(
                        cmp_ticker, years_input, st.session_state.data_nonce
                    )
                    cmp_result, _ = load_trained_model(
                        cmp_ticker, st.session_state.data_nonce
                    )
            except (ValueError, FileNotFoundError) as exc:
                st.warning(f"Skipping {cmp_ticker}: {exc}")
                continue

            cmp_df = cmp_df.copy()
            cmp_df[DATE_COLUMN] = pd.to_datetime(cmp_df[DATE_COLUMN])
            cmp_mask = (cmp_df[DATE_COLUMN].dt.date >= range_start) & (
                cmp_df[DATE_COLUMN].dt.date <= range_end
            )
            cmp_slice = cmp_df.loc[cmp_mask].reset_index(drop=True)
            if cmp_slice.empty:
                cmp_slice = cmp_df

            base_price = cmp_slice["Close"].iloc[0]
            cmp_slice = cmp_slice.assign(
                Indexed=cmp_slice["Close"] / base_price * 100, Ticker=cmp_ticker
            )
            normalized_frames.append(cmp_slice[[DATE_COLUMN, "Indexed", "Ticker"]])

            total_return = (cmp_slice["Close"].iloc[-1] / base_price - 1) * 100
            summary_rows.append(
                {
                    "Ticker": cmp_ticker,
                    "Total Return %": round(total_return, 2),
                    "Return Volatility": round(cmp_slice["Daily_Return"].std(), 3),
                    "Best Model": cmp_result.best_model_name,
                    "Best R²": round(cmp_result.r2_scores[cmp_result.best_model_name], 4),
                }
            )

        if normalized_frames:
            combined = pd.concat(normalized_frames, ignore_index=True)
            fig = px.line(
                combined,
                x=DATE_COLUMN,
                y="Indexed",
                color="Ticker",
                title="Normalized Price Comparison (Start = 100)",
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Indexed Price")
            st.plotly_chart(fig, width='stretch')

            st.subheader("Summary")
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, width='stretch')
            download_button(summary_df, "⬇️ Download Comparison CSV", "ticker_comparison.csv")
