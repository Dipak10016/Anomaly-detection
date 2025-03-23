import requests
import pandas as pd
import spacy
import os
from twilio.rest import Client
from fastapi import FastAPI

# ✅ Fetch News Data Using NewsAPI
API_KEY = "86b470122abe4be495fae92177b2941d"
QUERY = "toilet paper OR sanitizer OR shampoo OR soap OR biscuits OR chocolates OR noodles OR cold drinks OR tea OR coffee OR cooking oil OR rice OR detergent OR toothpaste OR toothbrush OR dishwashing liquid OR deodorants OR shaving cream OR face wash OR hand wash OR body lotion OR cleaning supplies OR baby diapers OR wet wipes OR snacks OR fruit juices OR canned foods OR cheese OR yogurt OR ice cream OR bread OR cereal OR jam OR bottled water OR milk powder OR pet food OR mouthwash"
URL = f"https://newsapi.org/v2/everything?q={QUERY}&language=en&apiKey={API_KEY}"
response = requests.get(URL)
news_data = response.json()

# ✅ Extract news articles
articles = []
for article in news_data.get("articles", []):
    articles.append({
        "title": article["title"],
        "description": article["description"],
        "content": article["content"],
        "published_at": article["publishedAt"],
        "source": article["source"]["name"],
        "url": article["url"]
    })
df_news = pd.DataFrame(articles)
df_news.dropna(inplace=True)

# ✅ Extract product mentions
fmcg_products = ["sanitizer", "shampoo", "biscuits", "toothpaste", "soap"]
nlp = spacy.load("en_core_web_sm")

def extract_products(text):
    return list(set([product for product in fmcg_products if product in str(text).lower()]))

df_news["products_mentioned"] = df_news["content"].apply(extract_products)
df_news = df_news[df_news["products_mentioned"].apply(len) > 0]

# ✅ Predict demand
product_demand = {}
for _, row in df_news.iterrows():
    for product in row["products_mentioned"]:
        product_demand[product] = product_demand.get(product, 0) + 1
df_demand = pd.DataFrame(product_demand.items(), columns=["Product", "Positive Mentions"])
df_demand.sort_values(by="Positive Mentions", ascending=False, inplace=True)

# ✅ FastAPI Service for Demand Predictions
app = FastAPI()

@app.get("/demand-prediction")
def get_demand():
    return df_demand.to_dict(orient="records")

# ✅ Twilio SMS Alert Configuration
os.environ["TWILIO_SID"] = "AC27635806cc59a26a326a8a80ce639ab3"
os.environ["TWILIO_AUTH_TOKEN"] = "1b109c2ff1510684f28e17d5d9b66b8d"
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = "+18103461792"
USER_PHONE = "+918010490660"
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_alert(product, count):
    try:
        message = client.messages.create(
            body=f"🚨 Demand Alert: '{product}' demand is rising! {count} positive mentions in the news.",
            from_=TWILIO_PHONE,
            to=USER_PHONE,
        )
        print(f"✅ Alert Sent for {product}: {message.sid}")
    except Exception as e:
        print("❌ Twilio Error:", e)

# ✅ Send SMS alerts dynamically based on extracted demand
if not df_demand.empty:
    for _, row in df_demand.iterrows():
        send_alert(row["Product"], row["Positive Mentions"])
else:
    print("⚠️ No relevant product mentions found in the news.")
