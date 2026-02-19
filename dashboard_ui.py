import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. ENTERPRISE DESIGN SYSTEM (CSS) ---
def apply_enterprise_theme():
    """
    Styles the Native Streamlit Containers to look like 'Figma Cards'.
    Ensures visual stability and enterprise aesthetic.
    """
    st.markdown("""
        <style>
        :root {
            --bg-color: #f8f9fa;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --semantic-green: #10b981;
            --semantic-red: #ef4444;
        }

        .stApp {
            background-color: var(--bg-color);
            font-family: 'Inter', sans-serif;
        }
        
        /* Targets st.container(border=True) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 24px !important;
            border: 1px solid #e5e7eb;
            min-height: 420px; 
        }
        
        h3 { font-size: 16px !important; font-weight: 700 !important; color: var(--text-primary) !important; margin: 0 !important; }
        p { font-size: 12px !important; color: var(--text-secondary) !important; margin-bottom: 15px !important; }

        div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 800 !important; }
        
        div[data-testid="stToolbar"] { display: none; }
        header { visibility: hidden; }
        
        /* Filter Container Styling */
        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        
        /* ------------------------------------------- */
        /* 🎨 NEW: SIDEBAR ALIGNMENT & BUTTON STYLES */
        /* ------------------------------------------- */
        
        /* Sidebar Container Alignment */
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            align-items: center; /* Horizontally center content */
            padding-top: 2rem;
        }

        /* Unified Button Styling */
        div.stButton > button {
            width: 100%; /* Fill the container width */
            border: none;
            background-color: transparent;
            color: #6b7280; /* Matches --text-secondary */
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            font-weight: 500;
            padding: 12px 20px;
            text-align: center; /* Center text inside button */
            display: flex;
            justify-content: center; /* Center content horizontally */
            align-items: center; /* Center content vertically */
            transition: all 0.2s ease-in-out;
            border-radius: 8px; /* Softer edges */
            margin-bottom: 8px;
        }

        /* Button Hover State */
        div.stButton > button:hover {
            background-color: #f3f4f6; /* Light gray hover */
            color: #111827; /* Darker text on hover */
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Active/Focus State */
        div.stButton > button:active, div.stButton > button:focus {
            background-color: #e5e7eb;
            color: #111827;
            border: none;
            outline: none;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 2. COMPONENT BUILDERS ---

def render_section_header(title, color="#ef4444"):
    """Creates the bold section headers with a left accent bar."""
    st.markdown(f"""
        <div style="margin-top: 30px; margin-bottom: 15px; display: flex; align-items: center;">
            <div style="width: 4px; height: 20px; background: {color}; margin-right: 10px; border-radius: 2px;"></div>
            <span style="font-size: 18px; color: #374151; font-weight: 600;">{title}</span>
        </div>
    """, unsafe_allow_html=True)

def render_global_filters(df):
    with st.container(border=True):
        st.markdown("**🌍 Global Market Context**")
        c1, c2, c3 = st.columns(3)

        scope = c1.selectbox(
            "Product Scope",
            ["All Comparable", "Budget", "Mid", "Premium"],
            key="scope"
        )

        perspective = c2.selectbox(
            "Seller Perspective",
            ["Market View", "Highlight: Star Tech", "Highlight: Ryans"],
            key="perspective"
        )

        time_window = c3.selectbox(
            "Time Window",
            ["Snapshot", "Last 14 Days", "Last 30 Days"],
            key="time_window"
        )

    return scope, perspective, time_window


# --- 3. CHART GENERATORS ---

def plot_competitiveness_gauge(value):
    """KPI 1: Competitiveness Index Gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 34, 'color': '#111827'}},
        gauge={'axis': {'range': [0, 100], 'visible': False}, 'bar': {'color': "#10b981"}, 'bgcolor': "#f3f4f6", 'borderwidth': 0}
    ))
    fig.update_layout(height=215, margin=dict(l=30, r=30, t=10, b=50), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_fair_value_trend(df):
    """KPI 2: Price Drift Trend"""
    if df.empty or 'pricing_gap' not in df.columns:
        return go.Figure().update_layout(height=130)

    daily = df.groupby('timestamp')['pricing_gap'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['timestamp'], y=daily['pricing_gap'], 
        mode='lines', line=dict(color='#ef4444', width=3), 
        fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", annotation_text="Fair Value")
    fig.update_layout(
        height=155, margin=dict(l=0, r=0, t=10, b=0), 
        xaxis=dict(visible=False), yaxis=dict(visible=False), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        showlegend=False
    )
    return fig

def compute_market_coverage_ratio(focus_df, benchmark_df):
    """KPI 4: Market Coverage Ratio - What % of market models does seller carry?"""
    if benchmark_df.empty or focus_df.empty:
        return 0.0
    
    # Get unique models (handle NaN values)
    seller_models = set(focus_df['model_name'].dropna().unique())
    market_models = set(benchmark_df['model_name'].dropna().unique())
    
    if not market_models:  # No market data
        return 0.0
    
    # Calculate coverage
    common_models = seller_models.intersection(market_models)
    coverage_ratio = (len(common_models) / len(market_models)) * 100
    
    return round(coverage_ratio, 1)


def plot_seller_coverage_progress(coverage_ratio):
    """Progress bar for Seller View - Market Coverage Ratio"""
    fig = go.Figure()
    
    # Background bar
    fig.add_trace(go.Bar(
        x=[100],
        y=[''],
        orientation='h',
        marker_color='#f3f4f6',
        width=0.5,
        showlegend=False
    ))
    
    # Progress bar with conditional color
    if coverage_ratio >= 66:
        color = '#10b981'
    elif coverage_ratio >= 33:
        color = '#f59e0b'
    else:
        color = '#ef4444'
    
    fig.add_trace(go.Bar(
        x=[coverage_ratio],
        y=[''],
        orientation='h',
        marker_color=color,
        width=0.5,
        text=f"{coverage_ratio}%",
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='white', size=16, family="Arial Black"),
        showlegend=False
    ))
    
    fig.update_layout(
        height=125,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay'
    )
    
    return fig


def plot_coverage_breakdown(focus_df, benchmark_df):
    """Simplified coverage breakdown - NO expanders, NO lists"""
    if benchmark_df.empty or focus_df.empty:
        return "No data"
    
    seller_models = set(focus_df['model_name'].dropna().unique())
    market_models = set(benchmark_df['model_name'].dropna().unique())
    
    common = seller_models.intersection(market_models)
    missing = market_models - seller_models
    
    # Just show the numbers, no expanders
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Models Carried", len(common))
    
    with col2:
        st.metric("Market Models", len(market_models))
    
    # Optional: Simple status message
    coverage_pct = (len(common) / len(market_models) * 100) if market_models else 0
    
    if coverage_pct >= 66:
        st.success("✅ Strong market coverage")
    elif coverage_pct >= 33:
        st.warning("⚠️ Moderate coverage")
    else:
        st.error("🔴 Limited coverage")

def plot_promo_bubble(df):
    """KPI 5: Promotional Intensity Benchmarking"""
    if df.empty: return go.Figure().update_layout(height=300)
    
    safe_reg = df['regular_price'].replace(0, 1)
    df = df.copy()
    df['discount_pct'] = ((df['regular_price'] - df['effective_price']) / safe_reg) * 100
    df['is_discounted'] = df['effective_price'] < df['regular_price']
    
    bubble_data = df.groupby('seller_name').agg(
        avg_discount=('discount_pct', 'mean'), 
        promo_coverage=('is_discounted', 'mean'), 
        sku_count=('model_name', 'count')
    ).reset_index()
    
    bubble_data['promo_coverage'] = bubble_data['promo_coverage'] * 100
    
    fig = px.scatter(
        bubble_data, x='avg_discount', y='promo_coverage', size='sku_count', 
        color='seller_name', hover_name='seller_name', size_max=45
    )
    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=10, b=20), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_stacked_segments(df):
    """KPI 6: Multi-Seller Target Segments (Budget/Mid/Premium)"""
    if df.empty: return go.Figure().update_layout(height=300)
    
    df = df.copy()
    df['segment'] = pd.cut(df['effective_price'], bins=[0, 40000, 80000, float('inf')], labels=['Budget', 'Mid', 'Premium'])
    
    # observed=False ensures that we don't drop sellers even if they are missing a segment
    target_segment = df.groupby(['seller_name', 'segment'], observed=False).size().reset_index(name='product_count')
    
    # Filter out any lingering 0-product rows from potential categorical cross-products if necessary
    # target_segment = target_segment[target_segment['product_count'] > 0]

    color_map = {'Budget': '#10b981', 'Mid': '#3b82f6', 'Premium': '#ef4444'}
    fig = px.bar(
        target_segment, x='product_count', y='seller_name', color='segment', 
        orientation='h', color_discrete_map=color_map, barmode='stack',
        category_orders={"segment": ["Budget", "Mid", "Premium"]}
    )
    fig.update_layout(
        height=320, margin=dict(l=0, r=20, t=10, b=20), 
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        xaxis=dict(showgrid=True, gridcolor='#f3f4f6'), yaxis=dict(title=None, categoryorder='total descending')
    )
    return fig

def plot_price_position_stock_health(df, full_market_df=None, filters=None):
    """KPI 6: Price Position vs Stock Health - Works with 3 global filters"""
    if df.empty: 
        return go.Figure().update_layout(
            height=500, 
            title="Price Position vs Stock Health",
            annotations=[dict(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )]
        )
    
    # Create copies
    working_df = df.copy()
    market_source = full_market_df.copy() if full_market_df is not None else df.copy()
    
    # Handle timestamp column if present
    for df_copy in [working_df, market_source]:
        if 'timestamp' in df_copy.columns and 'date' not in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
        elif 'date' not in df_copy.columns:
            df_copy['date'] = pd.Timestamp.today().date()
    
    # Apply your 3 global filters to BOTH datasets
    if filters:
        # 1. Product Scope (Segment/Brand/Model)
        if 'segment' in filters and filters['segment']:
            working_df = working_df[working_df['segment'] == filters['segment']]
            market_source = market_source[market_source['segment'] == filters['segment']]
        
        if 'brand' in filters and filters['brand']:
            brands = [filters['brand']] if isinstance(filters['brand'], str) else filters['brand']
            working_df = working_df[working_df['brand'].isin(brands)]
            market_source = market_source[market_source['brand'].isin(brands)]
        
        # 2. Seller Perspective
        if 'seller_name' in filters and filters['seller_name']:
            sellers = [filters['seller_name']] if isinstance(filters['seller_name'], str) else filters['seller_name']
            working_df = working_df[working_df['seller_name'].isin(sellers)]
            market_source = market_source[market_source['seller_name'].isin(sellers)]
        
        # 3. Time Window
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            working_df = working_df[
                (working_df['date'] >= start_date) & 
                (working_df['date'] <= end_date)
            ]
            market_source = market_source[
                (market_source['date'] >= start_date) & 
                (market_source['date'] <= end_date)
            ]
    
    # Check if we have data after filtering
    if working_df.empty:
        return go.Figure().update_layout(
            height=500,
            title="Price Position vs Stock Health",
            annotations=[dict(
                text="No data matches the selected filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )]
        )
    
    # 1. Calculate market floor (from filtered market source)
    market_floor = market_source.groupby('model_name')['effective_price'].min()
    
    # 2. Calculate price delta
    working_df['price_delta'] = working_df['effective_price'] - working_df['model_name'].map(market_floor)
    
    # 3. Calculate stock coverage
    if filters and 'date_range' in filters:
        start_date, end_date = filters['date_range']
        total_days = (end_date - start_date).days + 1
    else:
        # Calculate from actual data range
        min_date = working_df['date'].min()
        max_date = working_df['date'].max()
        total_days = (max_date - min_date).days + 1 if len(working_df) > 1 else 30
    
    stock_days = working_df.groupby('model_name')['date'].nunique()
    working_df['stock_depth'] = working_df['model_name'].map(stock_days)
    working_df['stock_coverage'] = working_df['stock_depth'] / total_days
    
    # 4. Add segment (even if not filtered, for visualization)
    working_df['segment'] = pd.cut(
        working_df['effective_price'], 
        bins=[0, 40000, 80000, float('inf')],
        labels=['Budget', 'Mid-Range', 'Premium'],
        include_lowest=True
    ).fillna('Budget')
    
    # 5. Create visualization
    fig = go.Figure()
    
    colors = {'Budget': '#3B82F6', 'Mid-Range': '#10B981', 'Premium': '#F59E0B'}
    
    for segment in ['Budget', 'Mid-Range', 'Premium']:
        segment_df = working_df[working_df['segment'] == segment]
        if not segment_df.empty:
            customdata = list(zip(
                segment_df.get('brand', 'N/A'),
                segment_df['effective_price'],
                segment_df['model_name'].map(market_floor),
                segment_df['stock_coverage'] * 100
            ))
            
            fig.add_trace(go.Scatter(
                x=segment_df['price_delta'],
                y=segment_df['stock_coverage'],
                mode='markers',
                name=segment,
                marker=dict(
                    size=12,
                    color=colors[segment],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=segment_df['model_name'],
                customdata=customdata,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Brand: %{customdata[0]}<br>"
                    "Segment: " + segment + "<br>"
                    "Price: ৳%{customdata[1]:,}<br>"
                    "Market Floor: ৳%{customdata[2]:,}<br>"
                    "Price Delta: ৳%{x:,}<br>"
                    "Stock Coverage: %{customdata[3]:.1f}%<extra></extra>"
                )
            ))
    
    # Add quadrant lines
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # 6. Update title to show active global filters
    title = "Price Position vs Stock Health"
    if filters:
        filter_text = []
        
        # Product Scope
        if 'segment' in filters and filters['segment']:
            filter_text.append(f"Segment: {filters['segment']}")
        if 'brand' in filters and filters['brand']:
            brands = filters['brand']
            if isinstance(brands, list):
                brand_text = ', '.join(brands[:2])
                if len(brands) > 2: brand_text += f" (+{len(brands)-2})"
            else:
                brand_text = brands
            filter_text.append(f"Brand: {brand_text}")
        
        # Seller Perspective
        if 'seller_name' in filters and filters['seller_name']:
            sellers = filters['seller_name']
            if isinstance(sellers, list):
                seller_text = ', '.join(sellers[:2])
                if len(sellers) > 2: seller_text += f" (+{len(sellers)-2})"
            else:
                seller_text = sellers
            filter_text.append(f"Seller: {seller_text}")
        
        # Time Window
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            if isinstance(start_date, str):
                filter_text.append(f"Period: {start_date} to {end_date}")
            else:
                filter_text.append(f"Period: {start_date.date()} to {end_date.date()}")
        
        if filter_text:
            title += f"<br><sub>{' | '.join(filter_text)}</sub>"
    
    fig.update_layout(
        title={'text': title, 'font': {'size': 18}},
        xaxis_title="Price Above/Below Market (₹৳)",
        yaxis_title="Stock Coverage (%)",
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        hovermode='closest',
        yaxis=dict(
            tickformat=".0%",
            range=[0, 1.05],
            showgrid=True,
            gridcolor='#f3f4f6'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#f3f4f6',
            zeroline=True,
            zerolinecolor='#d1d5db'
        )
    )
    
    return fig

# --- 4. MAIN RENDERER ---

def render_dashboard(df):
    apply_enterprise_theme()
    st.markdown("""<h1 style="font-size: 28px; font-weight: 800; color: #111827;">Pricing Analytics</h1>""", unsafe_allow_html=True)
    
    # ✅ IMPORT ALL NEEDED LIBRARIES AT THE TOP
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import re
    
    # 0. STRATEGIC DATA NORMALIZATION & CLEANING
    df = df[df['effective_price'] > 0].copy()
    
    for col in ['category', 'brand', 'seller_name']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    scope, perspective, time_window = render_global_filters(df)

    # 🎯 Determine which seller to focus on
    if perspective == "Market View":
        sel_seller = "Market Overview"
        dashboard_mode = "market"
    elif "Highlight: Ryans" in perspective:
        sel_seller = "Ryans"
        dashboard_mode = "seller"
    elif "Highlight: Star Tech" in perspective:
        sel_seller = "Star Tech"
        dashboard_mode = "seller"
    else:
        sel_seller = "Market Overview"
        dashboard_mode = "market"

    # 🎯 FUNCTION: Create comparable product groups
    def create_product_group(model_name):
        """Group similar products together for fair comparison"""
        model_lower = str(model_name).lower()
        
        # Extract brand from beginning of string
        brands = ['apple', 'acer', 'asus', 'msi', 'lenovo', 'dell', 'hp', 'samsung']
        brand = 'other'
        for b in brands:
            if model_lower.startswith(b.lower()):
                brand = b
                break
        
        # Extract product type
        if 'macbook' in model_lower or 'laptop' in model_lower:
            product_type = 'laptop'
        elif 'desktop' in model_lower or 'pc' in model_lower:
            product_type = 'desktop'
        elif 'monitor' in model_lower:
            product_type = 'monitor'
        elif 'tablet' in model_lower:
            product_type = 'tablet'
        elif 'phone' in model_lower:
            product_type = 'phone'
        else:
            product_type = 'computer'
        
        # Extract RAM if available
        ram_match = re.search(r'(\d+)\s*gb', model_lower)
        ram = ram_match.group(1) if ram_match else 'unknown'
        
        return f"{brand}_{product_type}_{ram}gb"

    # 🎯 CRITICAL FIX 1: Create TWO separate DataFrames
    # -------------------------------------------------
    
    # A. FULL MARKET (for ranking - NO filters except stock)
    full_market_df = df[df['stock_status'] == 1].copy()
    
    # Remove duplicates in FULL market
    full_market_df = full_market_df.drop_duplicates(
        subset=['model_name', 'seller_name', 'effective_price']
    ).copy()
    
    # B. FILTERED MARKET (for display - WITH filters)
    filtered_df = df[df['stock_status'] == 1].copy()
    
    if time_window != "Snapshot" and 'timestamp' in filtered_df.columns:
        days = 14 if "14" in time_window else 30
        latest_date = filtered_df['timestamp'].max()
        cutoff_date = latest_date - pd.Timedelta(days=days)
        filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff_date]
    
    # Remove duplicates in filtered market too
    filtered_df = filtered_df.drop_duplicates(
        subset=['model_name', 'seller_name', 'effective_price']
    ).copy()
    
    # 🎯 CRITICAL FIX 2: Apply product grouping
    # -------------------------------------------------
    
    # Apply grouping to full market
    full_market_df['product_group'] = full_market_df['model_name'].apply(create_product_group)
    
    # Calculate ranks within COMPARABLE product groups
    full_market_df['temp_rank'] = full_market_df.groupby('product_group')['effective_price'].rank(method='dense', pct=True)
    
    
    # 🎯 CRITICAL FIX 3: Apply scope filter
    # -------------------------------------------------
    benchmark_df = filtered_df.copy()
    if scope != "All Comparable":
        scope_mapping = {"Budget": "Budget", "Mid": "Mid-Range", "Premium": "Premium"}
        if scope in scope_mapping:
            benchmark_df = benchmark_df[benchmark_df['category'] == scope_mapping[scope]]
    
    # 🎯 Create rank_map for all sellers
    rank_map = {}
    for idx, row in full_market_df.iterrows():
        key = f"{row['model_name']}_{row['seller_name'].strip().lower()}"
        rank_map[key] = row['temp_rank']
    
    # 🎯 PRICE DRIFT CALCULATION FUNCTION (MOVED HERE)
    # -------------------------------------------------
    
    def calculate_price_drift(focus_df, benchmark_df):
        """Calculate price drift percentage over time"""
        # Use 'timestamp' if 'date' doesn't exist
        date_column = 'date' if 'date' in focus_df.columns else 'timestamp'
        
        if focus_df.empty or date_column not in focus_df.columns:
            return None, pd.Series(dtype=float), None, None
        
        try:
            # Convert date column if needed
            if not pd.api.types.is_datetime64_any_dtype(focus_df[date_column]):
                focus_df[date_column] = pd.to_datetime(focus_df[date_column])
            if not pd.api.types.is_datetime64_any_dtype(benchmark_df[date_column]):
                benchmark_df[date_column] = pd.to_datetime(benchmark_df[date_column])
            
            # Market reference per model per day
            market_daily_median = (
                benchmark_df
                .groupby([date_column, 'model_name'])['effective_price']
                .median()
            )
            
            # Map market median to focus products
            focus_df['market_median'] = focus_df.set_index(
                [date_column, 'model_name']
            ).index.map(market_daily_median)
            
            # Remove rows where market median is NaN (no benchmark data)
            focus_df = focus_df.dropna(subset=['market_median'])
            
            if focus_df.empty:
                return None, pd.Series(dtype=float), None, None
            
            # Normalized drift (percentage, not raw price)
            focus_df['price_drift_pct'] = (
                (focus_df['effective_price'] - focus_df['market_median']) / 
                focus_df['market_median'] * 100
            )
            
            # Aggregate over time
            drift_series = (
                focus_df
                .groupby(date_column)['price_drift_pct']
                .mean()
                .sort_index()
            )
            
            # Calculate summary metrics
            avg_drift = drift_series.mean() if not drift_series.empty else 0
            drift_volatility = drift_series.std() if not drift_series.empty else 0
            
            # Calculate percentage of days above/below market
            days_above = (drift_series > 0).sum()
            days_below = (drift_series < 0).sum()
            total_days = len(drift_series)
            
            pct_above = (days_above / total_days * 100) if total_days > 0 else 0
            pct_below = (days_below / total_days * 100) if total_days > 0 else 0
            
            return avg_drift, drift_series, drift_volatility, (pct_above, pct_below)
            
        except Exception as e:
            st.warning(f"Could not calculate price drift: {str(e)}")
            return None, pd.Series(dtype=float), None, None
        
    
    # 🎯 Create focus_df based on perspective
    if dashboard_mode == "seller":
        # SINGLE SELLER FOCUS
        focus_df = benchmark_df[benchmark_df['seller_name'].str.contains(sel_seller, case=False, na=False)].copy()
        
        # Calculate competitiveness score for this seller
        if not focus_df.empty:
            # Map ranks to focus_df
            focus_df['match_key'] = focus_df.apply(
                lambda row: f"{row['model_name']}_{row['seller_name'].strip().lower()}", 
                axis=1
            )
            
            focus_df['market_rank'] = focus_df['match_key'].map(rank_map).fillna(0.5)
            rank_score = (1 - focus_df['market_rank'].mean()) * 100
            
            # 🎯 NEW: Calculate price drift for seller
            avg_drift, drift_series, drift_volatility, drift_percentages = calculate_price_drift(
                focus_df.copy(), 
                benchmark_df.copy()
            )
        else:
            rank_score = 0
            avg_drift = None
            drift_series = pd.Series(dtype=float)
            drift_volatility = None
            drift_percentages = (0, 0)
    else:
        # MARKET VIEW - calculate average market competitiveness
        focus_df = benchmark_df.copy()
        
        # Calculate average score across all sellers
        seller_scores = []
        for seller in benchmark_df['seller_name'].unique():
            seller_data = benchmark_df[benchmark_df['seller_name'] == seller]
            
            # Calculate score for this seller
            seller_focus = full_market_df[full_market_df['seller_name'] == seller]
            if len(seller_focus) > 0:
                seller_focus['match_key'] = seller_focus.apply(
                    lambda row: f"{row['model_name']}_{row['seller_name'].strip().lower()}", 
                    axis=1
                )
                seller_ranks = seller_focus['match_key'].map(rank_map).fillna(0.5)
                seller_score = (1 - seller_ranks.mean()) * 100 if len(seller_ranks) > 0 else 50
                seller_scores.append(seller_score)
        
        # Calculate market average score
        rank_score = sum(seller_scores) / len(seller_scores) if seller_scores else 50
        
        # 🎯 NEW: For market view, calculate overall market drift
        avg_drift, drift_series, drift_volatility, drift_percentages = calculate_price_drift(
            benchmark_df.copy(), 
            benchmark_df.copy()
        )
    
    # 🎯 RENDER THE APPROPRIATE DASHBOARD
    # -------------------------------------------------
    
    if dashboard_mode == "seller":
        # 🔴 RED DASHBOARD: SINGLE SELLER FOCUS
        render_section_header(f"🔍 Performance Focus: {sel_seller}", "#ef4444")
        
        if not focus_df.empty:
            # Show seller stats
            col1, col2, col3= st.columns(3)
            with col1:
                st.metric("Total Products", f"{len(benchmark_df):,}")
            with col2:
                min_price = benchmark_df['effective_price'].min()
                st.metric("Lowest Product Price", f"৳{min_price:,.0f}")
            with col3:
                max_price = benchmark_df['effective_price'].max()
                st.metric("Highest Product Price", f"৳{max_price:,.0f}")
            
            # Main dashboard columns
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("### 🎯 Competitiveness Index")
                    st.markdown(f"**{sel_seller} vs Market**")
                    
                    # Competitiveness Gauge
                    st.plotly_chart(plot_competitiveness_gauge(rank_score), use_container_width=True)
                    
                    # Interpretation
                    if rank_score >= 70:
                        status = "✅ Highly Competitive"
                        explanation = f"{sel_seller} is cheaper than 70%+ of competitors"
                    elif rank_score >= 50:
                        status = "⚖️ Market Competitive"
                        explanation = f"{sel_seller} is competitively priced"
                    elif rank_score >= 30:
                        status = "⚠️ Less Competitive"
                        explanation = f"{sel_seller} is more expensive than most competitors"
                    else:
                        status = "🔴 Premium Positioning"
                        explanation = f"{sel_seller} positions as a premium retailer"
                    
                    st.success(f"**{status}**")
                    st.caption(explanation)
            
            with c2:
                with st.container(border=True):
                        fig_drift = go.Figure()
                        
                        # Add drift line
                        fig_drift.add_trace(go.Scatter(
                            x=drift_series.index,
                            y=drift_series.values,
                            mode='lines+markers',
                            name='Price Drift',
                            line=dict(color='#ef4444', width=2),
                            marker=dict(size=6),
                            hovertemplate='timestamp: %{x|%b %d}<br>Drift: %{y:.1f}%<extra></extra>'
                        ))
                        
                        # Add zero reference line
                        fig_drift.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Market Median",
                            annotation_position="bottom right"
                        )
                        
                        # Update layout
                        fig_drift.update_layout(
                            title=f"{sel_seller}'s Price Drift Over Time",
                            xaxis_title="timestamp",
                            yaxis_title="Drift from Market Median (%)",
                            hovermode="x unified",
                            showlegend=True,
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        
                        # Customize y-axis to show percentages
                        fig_drift.update_yaxes(ticksuffix="%")
                        
                        st.plotly_chart(fig_drift, use_container_width=True)
                        
                        
    else:
        # 🔵 BLUE DASHBOARD: MARKET OVERVIEW
        render_section_header("Level 1: Market Overview", "#3b82f6")
        
        if not benchmark_df.empty:
            # Show market stats
            col1, col2, col3= st.columns(3)
            with col1:
                st.metric("Total Products", f"{len(benchmark_df):,}")
            with col2:
                min_price = benchmark_df['effective_price'].min()
                st.metric("Lowest Product Price", f"৳{min_price:,.0f}")
            with col3:
                max_price = benchmark_df['effective_price'].max()
                st.metric("Highest Product Price", f"৳{max_price:,.0f}")
            
            # Main dashboard columns (SAME LAYOUT as seller view)
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("### 🎯 Market Competitiveness")
                    st.markdown("**Overall Market Health**")
                    
                    # Show the gauge (SAME as seller view)
                    st.plotly_chart(plot_competitiveness_gauge(rank_score), use_container_width=True)
                    
                    # Interpretation for market
                    if rank_score >= 60:
                        status = "✅ Healthy Market"
                        explanation = "Most sellers are competitively priced"
                    elif rank_score >= 40:
                        status = "⚖️ Balanced Market"
                        explanation = "Mixed pricing strategies among sellers"
                    else:
                        status = "⚠️ Premium Market"
                        explanation = "Market leans toward premium pricing"
                    
                    st.success(f"**{status}**")
                    st.caption(explanation)
            
            with c2:
                with st.container(border=True):
                        # Create line chart for market
                        fig_drift = go.Figure()
                        
                        # Add drift line
                        fig_drift.add_trace(go.Scatter(
                            x=drift_series.index,
                            y=drift_series.values,
                            mode='lines',
                            name='Market Drift',
                            line=dict(color='#3b82f6', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(59, 130, 246, 0.1)',
                            hovertemplate='timestamp: %{x|%b %d}<br>Drift: %{y:.1f}%<extra></extra>'
                        ))
                        
                        # Add zero reference line
                        fig_drift.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Baseline",
                            annotation_position="bottom right"
                        )
                        
                        # Update layout
                        fig_drift.update_layout(
                            title="Market Price Drift Over Time",
                            xaxis_title="timestamp",
                            yaxis_title="Average Drift from Market Median (%)",
                            hovermode="x unified",
                            showlegend=True,
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        # Customize y-axis
                        fig_drift.update_yaxes(ticksuffix="%")
                        
                        st.plotly_chart(fig_drift, use_container_width=True)
            
            # Category Breakdown (if available)
        #     if 'category' in benchmark_df.columns:
                
        #         st.markdown("---")
        #         st.subheader("📦 Category Analysis")
                
        #         # Show category distribution
        #         category_data = benchmark_df['category'].value_counts().reset_index()
        #         category_data.columns = ['Category', 'Count']
                
        #         fig_cat = px.pie(
        #             category_data,
        #             values='Count',
        #             names='Category',
        #             title="Product Distribution by Category"
        #         )
        #         st.plotly_chart(fig_cat, use_container_width=True)
                
        # else:
        #     st.warning("No products found with current filters")
    # --- LEVEL 2 ---
    render_section_header("Level 2: Consumer Impact", "#f97316")
    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.markdown(f"### Ideal Price Index:")
            if not focus_df.empty:
                ideal = focus_df.groupby('model_name').agg(
                    Best=('effective_price', 'min'), 
                    Avg=('effective_price', 'mean')
                ).reset_index()
                ideal['Savings %'] = (((ideal['Avg'] - ideal['Best']) / ideal['Avg']) * 100).round(1)
                st.dataframe(ideal.sort_values('Savings %', ascending=False).head(8), 
                             column_config={"Savings %": st.column_config.ProgressColumn("Savings %", format="%f%%", min_value=0, max_value=30)}, 
                             hide_index=True, use_container_width=True, height=350)
            else:
                st.warning(f"No focus data found for.")
    
    # Replace your existing KPI 4 section with:

    with c4:
     with st.container(border=True):
        
        if perspective == "Market View":
            # 🥧 Market View: Category Distribution Pie Chart
            st.markdown("### Market Composition")
            
            if not benchmark_df.empty and 'category' in benchmark_df.columns:
                # Show category distribution - USING YOUR EXISTING LOGIC
                category_data = benchmark_df['category'].value_counts().reset_index()
                category_data.columns = ['Category', 'Count']
                
                fig_cat = px.pie(
                    category_data,
                    values='Count',
                    names='Category',
                    title="",  # Remove title since we have header above
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                # Update layout to fit in border
                fig_cat.update_layout(
                    height=350,  # Adjust height to fit in border
                    margin=dict(l=20, r=20, t=10, b=20),  # Smaller margins
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_cat, use_container_width=True)
                
                        
            else:
                st.info("No category data available")
                
        else:  # <-- ONLY ONE ELSE HERE (for Seller View)
            # 📊 Seller View: Market Coverage Ratio
            st.markdown("### Market Coverage Ratio")
            
            # Calculate coverage
            coverage_ratio = compute_market_coverage_ratio(focus_df, benchmark_df)
            
            # Top metric
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric(
                    label="Coverage",
                    value=f"{coverage_ratio}%",
                    delta=None,
                    help="Percentage of market models carried"
                )
            
            with col_b:
                # Progress bar visualization
                fig = plot_seller_coverage_progress(coverage_ratio)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Optional: Breakdown on hover/expand
            with st.expander("Coverage Details", expanded=False):
                plot_coverage_breakdown(focus_df, benchmark_df)
            
            
        

    # --- RENDER LEVEL 3 & 4 (Market Comparison) ---
    render_section_header("Level 3: Seller Market Behavior", "#eab308")
    c5, c6 = st.columns(2)
    with c5:
        with st.container(border=True):
            st.markdown("### Promotional Intensity")
            # 🟢 USES BENCHMARK STREAM: All sellers show as bubbles
            st.plotly_chart(plot_promo_bubble(benchmark_df), use_container_width=True)
    with c6:
        with st.container(border=True):
            st.markdown("### Target Segment")
            # 🟢 USES BENCHMARK STREAM: Both Ryans and Star Tech bars show side-by-side
            st.plotly_chart(plot_stacked_segments(benchmark_df), use_container_width=True)

    render_section_header("Level 4: Product Diagnostics", "#10b981")
    with st.container(border=True):
     st.plotly_chart(plot_price_position_stock_health(benchmark_df), use_container_width=True)