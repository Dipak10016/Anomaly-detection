Diagnostic Analytics: Root Cause Analysis for Supply Chain Disruptions in FMCG 🚚
Project Overview 🛠️
FMCG (Fast-Moving Consumer Goods) companies frequently face supply chain disruptions due to various factors such as unforeseen demand fluctuations, supplier delays, logistics bottlenecks, and regional economic challenges. Traditional methods focus mainly on post-incident reporting and response, leaving supply chain managers reactive rather than proactive.

Our AI-driven Diagnostic Analytics System dynamically identifies recurring patterns that lead to disruptions, performs real-time root cause analysis, and recommends proactive actions to mitigate risks. This system helps supply chain managers act before disruptions escalate, improving efficiency, reducing losses, and enhancing supply chain resilience.

Additionally, we've integrated NLP-based Text Extraction from trending FMCG news articles to keep supply chain managers updated on industry trends, emerging risks, and news that could affect their operations.

Problem Statement 🧠
FMCG companies face frequent supply chain disruptions due to unforeseen demand fluctuations, supplier delays, logistics bottlenecks, and regional economic factors. Traditional methods focus on post-incident reporting rather than real-time root cause identification. The challenge is to build an AI-driven diagnostic analytics system that dynamically identifies patterns leading to disruptions and pinpoints the root cause, enabling supply chain managers to take proactive action.

Additionally, the system should also integrate NLP-based Text Extraction from trending FMCG news articles to help managers stay informed about any external developments that might impact their supply chain operations.

Project Objectives 🎯
Historical Disruption Analysis:

Analyze historical disruptions to identify recurring patterns in supply chain failures.

Visualize disruption frequency by region and type of disruption using heatmaps. 🌍

Real-Time Risk Detection:

Develop an anomaly detection model that flags potential supply chain risks before they occur.

Alert supply chain managers when key metrics (e.g., delivery delays, weather events) deviate from normal behavior. ⚠️

Root Cause Analysis:

Implement causal inference techniques to determine the primary causes behind disruptions, such as weather, transport issues, or supplier delays. 🔎

Provide actionable insights into the drivers of each disruption to prevent future occurrences.

Interactive Dashboard:

Build a user-friendly, real-time dashboard to visualize supply chain metrics and risks.

Include interactive filters and charts to help managers drill down into specific disruptions or regions. 📊

Self-Learning System:

Continuously improve predictions by learning from incoming data, refining the model for better risk detection. 🧠

Adapt the system to evolving patterns of disruptions, enhancing its effectiveness over time. 🔄

Real-Time Mobile Notifications 📱:

Send real-time notifications to supply chain managers’ mobile devices when critical disruptions or risks are detected.

Enable managers to take immediate action or make informed decisions based on the latest data.

Future Product Predictions 🔮:

Use machine learning to predict future demand fluctuations, potential disruptions related to products, and trends in supply chain performance.

Provide forecasts to help managers optimize inventory levels and plan proactively for potential issues.

NLP Text Extraction from Trending FMCG News 📰:

Implement NLP techniques to extract key information from trending FMCG news articles.

Automatically fetch the latest news from sources, preprocess the content, and extract relevant insights, such as emerging risks, product launches, and industry trends.

Provide managers with timely news updates that could impact their supply chain operations.

Technologies Used 🧑‍💻
Backend: Python (Flask/Django)

Machine Learning:

Scikit-learn (IsolationForest, RandomForestClassifier, StandardScaler, PCA, DBSCAN)

Pandas

NumPy

Time Series:

Statsmodels (Granger Causality, VAR models)

Data Visualization:

Matplotlib

Seaborn

Plotly (Plotly Express, Plotly Graph Objects)

Web Scraping & NLP: BeautifulSoup, SpaCy, NLTK, Scrapy

Model Persistence: Joblib

Real-Time Data Processing: WebSocket, Celery

Network Visualization: NetworkX

Time Management: datetime, timedelta

Methods Used 🔧
Isolation Forest:

Used for anomaly detection. It is an unsupervised learning algorithm ideal for identifying outliers and disruptions in supply chain data. It helps to detect patterns that deviate from the norm, allowing the system to flag potential risks before they materialize.

Random Forest:

A supervised machine learning algorithm used for classification and regression tasks. It builds multiple decision trees and merges them together to get a more accurate and stable prediction, helping identify causes of disruptions based on historical data.

PCA (Principal Component Analysis):

A dimensionality reduction technique used to reduce the complexity of high-dimensional data while retaining its essential features. PCA is used to simplify the dataset for better visualization and model training, helping identify key factors contributing to supply chain issues.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):

A clustering algorithm used to detect patterns or outliers in data, such as unusual disruption events or emerging trends. DBSCAN is particularly useful for identifying areas of the supply chain that may need attention due to abnormal behavior.

Time Series Analysis:

Using VAR (Vector Autoregression) and Granger Causality, time series models help capture the relationship between multiple time-dependent variables, such as sales, inventory, and external factors like weather. These models help predict future supply chain disruptions and identify causal relationships between variables.

NLP (Natural Language Processing):

Text extraction from trending FMCG news articles to identify key insights, such as product launches, market trends, or external disruptions that may impact supply chains. NLP techniques like Named Entity Recognition (NER) and sentiment analysis are used to extract valuable information from unstructured news text.

Performance Metrics 📊
The performance of the model is evaluated based on key metrics, and here are the results:

Accuracy: 0.8778

Precision: 0.8597

Recall: 0.8703

F1 Score: 0.8649

These performance metrics demonstrate the model's ability to identify disruptions and risks accurately, offering a balanced precision and recall.

Team 👥
Team Name: SIES Falcon 🦅

Team Members:

Dipak Ghadge

Atharva Golwalkar

Atharva Kaskar

Prem Rana

Contact 📧
For more information or inquiries, please reach out to the project team:

Dipak Ghadge
dipakghadge2004@gmail.com
