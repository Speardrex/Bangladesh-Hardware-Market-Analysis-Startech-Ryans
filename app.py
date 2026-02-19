import streamlit as st
import pandas as pd
import dashboard_ui as ui # ✅ UI Module
from database import init_db, get_connection, fetch_war_room_intel
from Ingestion import run_full_infiltration 

st.set_page_config(page_title="DataNexus Architect", layout="wide")

#  GLOBAL THEME INJECTION
ui.apply_enterprise_theme()

# Initialize Database
init_db()

# --- 2. DATA ACCESS ---
def fetch_master_intel():
    """Fetches the latest competitive snapshot from the PostgreSQL Vault."""
    conn = get_connection()
    if not conn: return pd.DataFrame()
    
    query = """
    SELECT p.brand, p.model_name, p.category, s.seller_name, 
           d.regular_price, d.special_price, d.cash_price, d.effective_price, 
           d.stock_status, d.timestamp
    FROM products_master p
    JOIN product_listings l ON p.product_key = l.product_key
    JOIN sellers s ON l.seller_id = s.seller_id
    JOIN daily_prices d ON l.listing_id = d.listing_id
    WHERE d.timestamp = (SELECT MAX(timestamp) FROM daily_prices WHERE listing_id = d.listing_id);
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- 3. NAVIGATION ---
if 'page' not in st.session_state: st.session_state.page = "MISSION BRIEF"

with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>DATANEXUS</h2>", unsafe_allow_html=True)
    if st.button("MISSION BRIEF"): st.session_state.page = "MISSION BRIEF"
    if st.button("INGESTION"): st.session_state.page = "01_INGESTION"
    if st.button("PROFILING"): st.session_state.page = "02_PROFILING"
    if st.button("DASHBOARD"): st.session_state.page = "03_WAR_ROOM"

# --- 4. PAGE ROUTING ---
if st.session_state.page == "MISSION BRIEF":
    st.title("🛡️ MISSION BRIEFING")
    st.info("System Ready. PostgreSQL Vault Connected.")

elif st.session_state.page == "01_INGESTION":
    st.title("📡 INGESTION HUB")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 TRIGGER GLOBAL INFILTRATION"):
            with st.spinner("Infiltrating Nodes..."):
                run_full_infiltration()
            st.success("Vault Updated.")
    with col2:
        st.write("Status: **Ready**")

elif st.session_state.page == "02_PROFILING":
    st.title("🔬 FORENSIC AUDIT")
    df = fetch_master_intel()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Vault is empty.")

elif st.session_state.page == "03_WAR_ROOM":
    # 1. Fetch data strictly using your read-only vault intel
    master_intel = fetch_war_room_intel() 
    
    # 2. Paint the Design perfectly
    ui.render_dashboard(master_intel)