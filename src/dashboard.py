import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from stock_model import fetch_data, add_features, train_predict, preprocess_custom_data
from rag_engine import StockRAGEngine
import datetime
from datetime import timedelta

st.set_page_config(page_title="AI Equity Analyst", layout="wide", initial_sidebar_state="expanded")

# --- CSS Overrides (Dark Mode Polish) ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font_size: 24px;
        color: #00CC96;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.title("AI Equity Analyst")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Data Source
data_source = st.sidebar.radio("Data Source", ["Yahoo Finance API", "Upload CSV File"])

df = None
ticker = "Custom Asset"

if data_source == "Yahoo Finance API":
    st.sidebar.markdown("### Asset Selection")
    ticker = st.sidebar.text_input("Ticker Symbol", "NVDA")
    st.sidebar.caption("Tip: For Indian Stocks, append .NS (e.g., RELIANCE.NS, TCS.NS)")
    
    # Model Selection
    st.sidebar.markdown("### Model Architecture")
    interval_options = ["Daily (1d)", "Hourly (1h)", "Minute (1m)"]
    selected_interval_str = st.sidebar.radio("Timeframe", interval_options, index=0)
    interval = selected_interval_str.split(" ")[1][1:-1]

    # Date Selection
    now = datetime.date.today()
    if interval == "1m":
        default_start = now - timedelta(days=6)
        st.sidebar.caption("Minute data limited to last 7 days.")
    else:
        default_start = now - timedelta(days=365*4)

    start_date = st.sidebar.date_input("Start Date", default_start)
    
    # End Date Logic
    use_latest = st.sidebar.checkbox("Use Latest Data (Today)", value=True)
    if use_latest:
        end_date = now
        st.sidebar.caption(f"End Date: {end_date} (Auto)")
    else:
        end_date = st.sidebar.date_input("End Date", now)
else:
    st.sidebar.markdown("### File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Required: 'Close' column)", type=["csv"])
    if uploaded_file:
        try:
            df = preprocess_custom_data(uploaded_file)
            st.sidebar.success(f"Successfully loaded {len(df)} records.")
        except Exception as e:
            st.sidebar.error(f"File Error: {e}")

# Model Selection
model_type = st.sidebar.selectbox("Algorithm", 
    ["Random Forest", "XGBoost", "Logistic Regression", "Stacked Ensemble", 
     "LSTM (Deep Learning)", "GRU (Deep Learning)", "Voting (RF + XGB + LSTM)"])

# Hyper-Tuning Toggle
enable_tuning = st.sidebar.checkbox("Enable Hyper-Tuning", help="Tests multiple configurations for best precision. Slower.")

# Determinism
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1, help="Fixed seed for reproducibility.")

# Model Info Expander
with st.sidebar.expander("Model Documentation"):
    if model_type == "Random Forest":
        st.write("**Random Forest Classifier**")
        st.write("Bagging Ensemble. Robust against noise.")
    elif model_type == "XGBoost":
        st.write("**XGBoost Classifier**")
        st.write("Gradient Boosting. Fast and accurate.")
    elif model_type == "Logistic Regression":
        st.write("**Logistic Regression**")
        st.write("Linear classification. Effective for trends.")
    elif model_type == "Stacked Ensemble":
        st.write("**Stacked Ensemble**")
        st.write("RF + XGBoost, judged by Logistic Regression.")
    elif "LSTM" in model_type and "Voting" not in model_type:
        st.write("**LSTM (Deep Learning)**")
        st.write("Long Short-Term Memory. Learns sequential patterns.")
    elif "GRU" in model_type:
        st.write("**GRU (Deep Learning)**")
        st.write("Gated Recurrent Unit. Efficient for short sequences.")
    else:
        st.write("**Voting Ensemble**")
        st.write("Averages Random Forest, XGBoost, and LSTM. Maximum stability.")

# Initialize RAG Engine in Session State
if 'rag_engine' not in st.session_state:
    st.session_state['rag_engine'] = StockRAGEngine()

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["Quantitative Analysis", "Deep Dive Data", "AI Insights (RAG)"])

# --- TAB 1: Quantitative Analysis ---
with tab1:
    if st.sidebar.button("Run Quantitative Analysis"):
        if df is None and data_source == "Upload CSV File":
            st.error("Please upload a valid CSV file.")
        else:
            with st.spinner(f"Processing data for {ticker}... (Tuning: {enable_tuning})"):
                try:
                    # Fetch Data
                    if data_source == "Yahoo Finance API":
                        df = fetch_data(ticker, start_date, end_date, interval)
                    
                    # Feature Engineering
                    df = add_features(df)
                    
                    # Train & Predict
                    model, precision, preds, probs, test_data, predictors = train_predict(
                        df, model_type=model_type, random_state=random_seed, enable_tuning=enable_tuning
                    )
                
                    # --- BENTO GRID LAYOUT ---
                    st.divider()
                    st.subheader(f"Dashboard: {ticker}")
                    
                    # Tuning Info
                    if enable_tuning:
                        st.caption("Optimized with Grid Search: Tested Depth [3, 5, 10], Estimators [100, 200, 500]")

                    # ROW 1: Key Metrics
                    k1, k2, k3, k4 = st.columns([1, 1, 2, 1])
                    
                    last_close = df["Close"].iloc[-1]
                    prev_close = df["Close"].iloc[-2]
                    daily_change = last_close - prev_close
                    
                    k1.metric("Last Price", f"{last_close:.2f}", f"{daily_change:.2f}")
                    k2.metric("Model Precision", f"{precision:.2%}", "Test Accuracy")
                    
                    current_prob = probs.iloc[-1]
                    if current_prob > 0.60:
                        sig_text, sig_color = "STRONG BUY", "normal"
                    elif current_prob > 0.50:
                        sig_text, sig_color = "WEAK BUY (Wait)", "off"
                    else:
                        sig_text, sig_color = "NEUTRAL / SELL", "inverse"
                        
                    k3.metric("Signal", sig_text, delta=f"{current_prob:.1%} Conf.", help="Strong Buy: >60% Probability\nWeak Buy: >50%\nNeutral/Sell: <50%")
                    k4.metric("Dataset Size", f"{len(df)} Rows", f"{len(test_data)} Tested")
                    
                    # ROW 2: Main Chart + Gauge
                    g1, g2 = st.columns([3, 1])
                    
                    with g1:
                        st.markdown("##### Price Action & Bands")
                        fig_main = go.Figure()
                        fig_main.add_trace(go.Candlestick(
                            x=test_data.index,
                            open=test_data['Open'], high=test_data['High'],
                            low=test_data['Low'], close=test_data['Close'],
                            name='OHLC'
                        ))
                        fig_main.add_trace(go.Scatter(x=test_data.index, y=test_data['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Upper Band'))
                        fig_main.add_trace(go.Scatter(x=test_data.index, y=test_data['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='Lower Band', fill='tonexty'))
                        
                        buys = test_data[preds == 1]
                        fig_main.add_trace(go.Scatter(
                            x=buys.index, y=buys['Low']*0.99, mode='markers', 
                            marker=dict(symbol='triangle-up', size=10, color='#00CC96'), name='Buy Signal'
                        ))
                        
                        fig_main.update_layout(height=400, template="plotly_white", margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
                        st.plotly_chart(fig_main, use_container_width=True)
                    
                    with g2:
                        st.markdown("##### Confidence")
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = current_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Buy Probability %"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#00CC96" if current_prob > 0.6 else "#EF553B"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "white"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 60}}))
                        fig_gauge.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    # Store for other tabs
                    # Session state allows the quantitative data (df, model) to be accessed
                    # by the 'Deep Dive' tab without re-running the analysis.
                    st.session_state['df'] = df
                    st.session_state['model'] = model
                    st.session_state['predictors'] = predictors

                except Exception as e:
                    st.error(f"System Error: {e}")
    else:
        st.info("Configure parameters and click 'Run Quantitative Analysis' to start.")

# --- TAB 2: Deep Dive ---
with tab2:
    if 'df' in st.session_state:
        df = st.session_state['df']
        model = st.session_state['model']
        predictors = st.session_state['predictors']
        
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("##### Feature Importance")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat_df = pd.DataFrame({"Feature": predictors, "Impact": importances}).sort_values(by="Impact", ascending=True).tail(10)
                fig_feat = px.bar(feat_df, x="Impact", y="Feature", orientation='h', color="Impact", color_continuous_scale="Teal")
                fig_feat.update_layout(template="plotly_white", height=300, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.info("Feature Importance not available for Deep Learning / Ensemble models.")
        
        with d2:
            st.markdown("##### Monthly Returns Heatmap")
            from stock_model import calculate_monthly_returns
            heatmap_data = calculate_monthly_returns(df)
            
            if not heatmap_data.empty:
                fig_heat = px.imshow(
                    heatmap_data, 
                    labels=dict(x="Month", y="Year", color="Return %"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale="RdBu", 
                    origin='lower'
                )
                fig_heat.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Insufficient data for Heatmap.")
    else:
        st.warning("Please run the Quantitative Analysis in Tab 1 first.")

# --- TAB 3: AI Insights (RAG) ---
with tab3:
    st.header("AI Market Insights")
    st.markdown("Leverage **RAG (Retrieval-Augmented Generation)** to analyze market news alongside price data.")
    
    rag_col1, rag_col2 = st.columns([1, 2])
    
    with rag_col1:
        st.subheader("1. Ingest News")
        if st.button("Fetch & Vectorize News"):
            with st.spinner(f"Fetching news for {ticker} and updating VectorDB..."):
                engine = st.session_state['rag_engine']
                status = engine.ingest_data(ticker)
                st.success(status)
                
    with rag_col2:
        st.subheader("2. Ask the AI")
        user_query = st.text_input("Ask a question about this stock:", f"What are the recent bullish indicators for {ticker}?")
        
        if st.button("Generate Insight"):
            with st.spinner("Analyzing vector embeddings and generating response with Ollama..."):
                engine = st.session_state['rag_engine']
                response = engine.generate_response(user_query, ticker)
                st.markdown("### AI Analysis")
                st.write(response)
                
    st.info("Note: Ensure 'ollama serve' is running locally for AI Insights to work.")
