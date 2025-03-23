# Diagnostic Analytics: Root Cause Analysis for Supply Chain Disruptions in FMCG 🚚

## Project Overview 🛠️

FMCG (Fast-Moving Consumer Goods) companies frequently face **supply chain disruptions** due to various factors such as **unforeseen demand fluctuations**, **supplier delays**, **logistics bottlenecks**, and **regional economic challenges**. Traditional methods primarily focus on **post-incident reporting** and **response**, leaving supply chain managers reactive rather than proactive.

Our **AI-driven Diagnostic Analytics System** dynamically identifies recurring patterns leading to disruptions, performs **real-time root cause analysis**, and recommends proactive actions to mitigate risks. This system empowers supply chain managers to act before disruptions escalate, improving efficiency, reducing losses, and enhancing supply chain resilience.

Additionally, we have integrated **NLP-based Text Extraction** from trending FMCG news articles to keep supply chain managers updated on **industry trends**, emerging risks, and news that could affect their operations.

---

## Problem Statement 🧠

FMCG companies face frequent supply chain disruptions due to unforeseen demand fluctuations, supplier delays, logistics bottlenecks, and regional economic factors. Traditional methods focus on **post-incident reporting** rather than real-time **root cause identification**. 

### The Challenge:
Build an **AI-driven diagnostic analytics system** that:
- Dynamically identifies patterns leading to disruptions
- Pinpoints the root cause of disruptions
- Enables supply chain managers to take **proactive action**.

Additionally, the system should integrate **NLP-based Text Extraction** from trending FMCG news articles to help managers stay informed about any **external developments** that might impact their supply chain operations.

---

## Project Objectives 🎯

1. **Historical Disruption Analysis**:
   - Analyze historical disruptions to identify recurring patterns.
   - Visualize disruption frequency by region and type using **heatmaps** 🌍.

2. **Real-Time Risk Detection**:
   - Develop an **anomaly detection model** to flag potential risks before they occur.
   - Alert supply chain managers when key metrics (e.g., **delivery delays**, **weather events**) deviate from normal behavior ⚠️.

3. **Root Cause Analysis**:
   - Implement **causal inference techniques** to identify primary causes of disruptions, such as weather, transport issues, or supplier delays 🔎.
   - Provide actionable insights to prevent future occurrences.

4. **Interactive Dashboard**:
   - Build a **user-friendly, real-time dashboard** to visualize supply chain metrics and risks.
   - Include interactive **filters** and **charts** to help managers drill down into specific disruptions or regions 📊.

5. **Self-Learning System**:
   - Continuously improve predictions by learning from incoming data, refining the model for better **risk detection** 🧠.
   - Adapt the system to evolving disruption patterns, enhancing its effectiveness over time 🔄.

6. **Real-Time Mobile Notifications** 📱:
   - Send **real-time notifications** to mobile devices when critical disruptions or risks are detected.
   - Enable managers to take immediate action based on the latest data.

7. **Future Product Predictions** 🔮:
   - Use **machine learning** to predict future demand fluctuations, potential disruptions, and trends in supply chain performance.
   - Provide **forecasts** to optimize inventory and plan proactively for potential issues.

8. **NLP Text Extraction from Trending FMCG News** 📰:
   - Implement **NLP** techniques to extract key information from trending FMCG news articles.
   - Automatically fetch, preprocess, and extract relevant insights such as **emerging risks**, **product launches**, and **industry trends**.

---
**DEMO VIDEO OF Realtime Mobile Notification System :** 
Video Shows Realtime Notification system to buy certain product stocks which are Popular in Market News.

https://github.com/user-attachments/assets/db364dfc-0371-44b6-9f85-d7a45622a656

## Technologies Used 🧑‍💻

- **Backend**: Python (Flask/Django)
- **Machine Learning**: 
  - Scikit-learn (IsolationForest, RandomForestClassifier, StandardScaler, PCA, DBSCAN)
  - Pandas
  - NumPy
- **Time Series**: Statsmodels (Granger Causality, VAR models)
- **Data Visualization**: 
  - Matplotlib
  - Seaborn
  - Plotly (Plotly Express, Plotly Graph Objects)
- **Web Scraping & NLP**: BeautifulSoup, SpaCy, NLTK, Scrapy
- **Model Persistence**: Joblib
- **Real-Time Data Processing**: WebSocket, Celery
- **Network Visualization**: NetworkX
- **Time Management**: datetime, timedelta

---

## Methods Used 🔧

1. **Isolation Forest**:
   - An unsupervised learning algorithm ideal for **anomaly detection**. It helps detect patterns deviating from the norm, flagging potential risks before they arise.

2. **Random Forest**:
   - A supervised learning algorithm for **classification and regression** tasks. It merges multiple decision trees to provide stable predictions, identifying causes of disruptions from historical data.

3. **PCA (Principal Component Analysis)**:
   - A dimensionality reduction technique simplifying high-dimensional data while retaining essential features, making it easier to visualize and train models.

4. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
   - A clustering algorithm used to detect unusual disruption events and emerging trends, particularly useful for identifying areas requiring attention.

5. **Time Series Analysis**:
   - **VAR (Vector Autoregression)** and **Granger Causality** models help capture the relationships between time-dependent variables like sales, inventory, and external factors, predicting future disruptions.

6. **NLP (Natural Language Processing)**:
   - Extracts key insights from trending FMCG news articles using **Named Entity Recognition (NER)** and **sentiment analysis** to identify emerging risks and industry trends.

---

## Performance Metrics 📊

- **Accuracy**: 0.8778
- **Precision**: 0.8597
- **Recall**: 0.8703
- **F1 Score**: 0.8649

These metrics demonstrate the model’s ability to accurately identify disruptions and risks, offering a balanced precision and recall.

---

## Team 👥

**Team Name**: SIES Falcon 🦅

### Team Members:
- [Dipak Ghadge](https://github.com/Dipak10016)
- [Atharva Golwalkar](https://github.com/atharvagolwalkar)
- [Atharva Kaskar](https://github.com/Atharva-Kaskar-45)
- [Prem Rana](https://github.com/Prem16-pro)

---

## Contact 📧

For more information or inquiries, please reach out to the project team:

- **Dipak Ghadge**  
  Email: [dipakghadge2004@gmail.com](mailto:dipakghadge2004@gmail.com)

---
