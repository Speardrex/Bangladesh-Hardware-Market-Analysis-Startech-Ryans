import psycopg2
import os

# 🔴 PASTE YOUR NEON CONNECTION STRING HERE
# It looks like: postgres://user:pass@ep-xyz.aws.neon.tech/neondb?sslmode=require
NEON_URL = "postgresql://neondb_owner:npg_hz9HmZADkX3G@ep-morning-cloud-a1do81xx-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def get_cloud_connection():
    try:
        return psycopg2.connect(NEON_URL)
    except Exception as e:
        print(f"❌ Cloud Connection Error: {e}")
        return None

def init_cloud_db():
    """Creates the tables in the Neon Cloud Database."""
    conn = get_cloud_connection()
    if not conn: return
    cur = conn.cursor()

    print("☁️ Initializing Cloud Schema...")
    
    # 1. Products Master
    cur.execute('''
        CREATE TABLE IF NOT EXISTS products_master (
            product_key SERIAL PRIMARY KEY,
            brand VARCHAR(50),
            model_name TEXT UNIQUE,
            category VARCHAR(50)
        );
    ''')

    # 2. Sellers
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sellers (
            seller_id SERIAL PRIMARY KEY,
            seller_name VARCHAR(50) UNIQUE
        );
    ''')

    # 3. Product Listings
    cur.execute('''
        CREATE TABLE IF NOT EXISTS product_listings (
            listing_id SERIAL PRIMARY KEY,
            product_key INTEGER REFERENCES products_master(product_key),
            seller_id INTEGER REFERENCES sellers(seller_id),
            url TEXT UNIQUE
        );
    ''')

    # 4. Daily Prices
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
    
    # Seed Sellers
    cur.execute("INSERT INTO sellers (seller_name) VALUES ('Ryans'), ('Star Tech') ON CONFLICT DO NOTHING;")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Cloud Vault Ready.")

if __name__ == "__main__":
    init_cloud_db()
