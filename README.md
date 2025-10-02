# Bitcoin AI Analyst

## Overview
Bitcoin AI Analyst is a Streamlit-powered dashboard and machine learning pipeline designed to analyze Bitcoin using real-time data, AI-driven insights, and predictive modeling. The system integrates technical indicators, macroeconomic signals, and on-chain metrics with a stacked ensemble approach to forecast Bitcoin’s next-day closing price.

---

## Key Features
- Interactive dashboard with candlestick charts and prediction visualizations  
- Stacked ensemble regression for next-day Bitcoin price forecasting  
- AI agents for Bitcoin news summarization, sentiment analysis, and conversational queries  
- Feature engineering with technical indicators (RSI, MACD, SMA/EMA, Bollinger Bands)  
- Integration of macroeconomic and on-chain metrics  

---

## Technology and Libraries
- **Frontend:** Streamlit, Plotly  
- **Machine Learning:** scikit-learn, XGBoost, RandomForest, Ridge/Lasso/ElasticNet  
- **Data Processing:** pandas, numpy  
- **Agents & NLP:** Agno framework (Gemini, Groq LLaMA, Ollama integration)  
- **Data Sources:** yFinance, on-chain metrics APIs, macroeconomic datasets  

---

## Project Performance
- **R²:** 0.9877  
- **RMSE:** $2,490.13  
- **MAE:** $1,950.63  
- **Directional Accuracy:** 51.38%  

---

## Limitations and Notes
- Directional accuracy is modest (~51%), suggesting limited predictive power for trading decisions.  
- Dependent on availability and reliability of external data sources.  
- Real-time deployment may require optimization for API latency and agent response times.  
- Future improvements include diversification of ensemble learners, explainability with SHAP, and online retraining pipelines.

---
