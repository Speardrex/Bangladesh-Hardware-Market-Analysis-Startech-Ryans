import psycopg2
import pandas as pd
import streamlit as st

# 🔴 FALLBACK NEON CONNECTION STRING
# It is highly recommended to use Streamlit Secrets instead of hardcoding this in a public repo!
NEON_URL = "postgresql://neondb_owner:npg_hz9HmZADkX3G@ep-morning-cloud-a1do81xx-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def get_connection():
    """Connects to Neon Cloud (Prefers Secure Secrets over Hardcoded URL)"""
    try:
        # 1. Try secure Streamlit Secrets first
        if hasattr(st, "secrets") and "postgres" in st.secrets:
            return psycopg2.connect(st.secrets["postgres"]["url"])
        
        # 2. Fallback to hardcoded URL
        return psycopg2.connect(NEON_URL)
    except Exception as e:
        print(f"❌ Cloud Connection Error: {e}")
        return None

def init_db():
    """Creates the tables in the Neon Cloud Database. (Renamed from init_cloud_db)"""
    conn = get_connection()
    if not conn: return
    cur = conn.cursor()

    print("☁️ Initializing Cloud Schema...")
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS products_master (
            product_key SERIAL PRIMARY KEY,
            brand VARCHAR(50),
            model_name TEXT UNIQUE,
            category VARCHAR(50)
        );
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS sellers (
            seller_id SERIAL PRIMARY KEY,
            seller_name VARCHAR(50) UNIQUE
        );
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS product_listings (
            listing_id SERIAL PRIMARY KEY,
            product_key INTEGER REFERENCES products_master(product_key),
            seller_id INTEGER REFERENCES sellers(seller_id),
            url TEXT UNIQUE
        );
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS daily_prices (
            price_id SERIAL PRIMARY KEY,
            listing_id INTEGER REFERENCES product_listings(listing_id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            regular_price INTEGER DEFAULT 0,
            special_price INTEGER DEFAULT 0,
            cash_price INTEGER DEFAULT 0,
            effective_price INTEGER NOT NULL,
            stock_status INTEGER DEFAULT 1
        );
    ''')
    
    cur.execute("INSERT INTO sellers (seller_name) VALUES ('Ryans'), ('Star Tech') ON CONFLICT DO NOTHING;")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Cloud Vault Ready.")

def fetch_war_room_intel():
    """
    CRUCIAL MISSING FUNCTION: Fetches the data for your Dashboard and Profiling tabs.
    Returns the exact columns your app.py expects.
    """
    cols = ['brand', 'model_name', 'category', 'seller_name', 'regular_price', 
            'special_price', 'cash_price', 'effective_price', 'stock_status', 'timestamp']
    
    conn = get_connection()
    if not conn: return pd.DataFrame(columns=cols)
    
    query = """
    SELECT 
        p.brand, p.model_name, p.category, s.seller_name, 
        d.regular_price, d.special_price, d.cash_price, d.effective_price, 
        d.stock_status, d.timestamp
    FROM daily_prices d
    LEFT JOIN product_listings l ON d.listing_id = l.listing_id
    LEFT JOIN products_master p ON l.product_key = p.product_key
    LEFT JOIN sellers s ON l.seller_id = s.seller_id
    ORDER BY d.timestamp DESC;
    """
    try:
        df = pd.read_sql(query, conn)
        if df.empty: return pd.DataFrame(columns=cols)
        return df
    except Exception as e:
        print(f"❌ DB Fetch Error: {e}")
        return pd.DataFrame(columns=cols)
    finally:
        conn.close()

# Aliases to ensure app.py doesn't crash regardless of what it calls
fetch_master_intel = fetch_war_room_intel

if __name__ == "__main__":
    init_db()
