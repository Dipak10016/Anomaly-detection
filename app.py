import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our anomaly detection module
# In a real project, you would import from the previous file
# For hackathon purposes, let's assume we have the data and models ready
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Sample data loading function (replace with your actual data)
def load_data():
    try:
        # Try to load the generated data from previous script
        df = pd.read_csv('supply_chain_with_anomalies.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        # If file doesn't exist, generate new data using the function from previous script
        # This would import the generate_sample_data function from the previous file
        # For demo purposes, we'll include a simplified version
        np.random.seed(42)
        n_samples = 1000
        
        # Date range for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
        selected_dates = np.random.choice(dates, n_samples)
        selected_dates = [date.strftime('%Y-%m-%d') for date in selected_dates]
        
        categories = ['Beverages', 'Personal Care', 'Home Care', 'Food', 'Snacks']
        regions = ['North', 'South', 'East', 'West', 'Central']
        suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
        
        df = pd.DataFrame({
            'date': pd.to_datetime(selected_dates),
            'category': np.random.choice(categories, n_samples),
            'region': np.random.choice(regions, n_samples),
            'supplier': np.random.choice(suppliers, n_samples),
            'order_quantity': np.random.normal(5000, 1000, n_samples),
            'lead_time_days': np.random.normal(5, 1, n_samples),
            'transport_time_days': np.random.normal(3, 0.5, n_samples),
            'inventory_level': np.random.normal(8000, 1500, n_samples),
            'demand_forecast': np.random.normal(4800, 900, n_samples),
            'production_capacity': np.random.normal(6000, 500, n_samples),
            'weather_impact': np.random.uniform(0, 0.3, n_samples),
        })
        
        # Create some derived metrics
        df['inventory_coverage_days'] = df['inventory_level'] / (df['demand_forecast'] / 30)
        df['capacity_utilization'] = df['demand_forecast'] / df['production_capacity']
        
        # Add random anomalies (5% of data)
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        df['anomaly_detected'] = 0
        df.loc[anomaly_indices, 'anomaly_detected'] = 1
        
        return df

# Initialize the Dash app
app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                title="Supply Chain Anomaly Detection")

# Load data
df = load_data()

# Define app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("FMCG Supply Chain Anomaly Detection", 
                style={"margin-bottom": "0px", "color": "white"}),
        html.H4("Real-time Monitoring & Root Cause Analysis Dashboard", 
                style={"margin-top": "0px", "color": "white"})
    ], style={"text-align": "center", "padding": "1rem", "background-color": "#2c3e50"}),
    
    # Filters row
    html.Div([
        html.Div([
            html.P("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df['date'].min().date(),
                max_date_allowed=df['date'].max().date(),
                start_date=df['date'].min().date(),
                end_date=df['date'].max().date()
            )
        ], style={"width": "25%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.P("Region:"),
            dcc.Dropdown(
                id='region-filter',
                options=[{"label": r, "value": r} for r in sorted(df['region'].unique())],
                value=[],
                multi=True,
                placeholder="Select regions..."
            )
        ], style={"width": "20%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.P("Category:"),
            dcc.Dropdown(
                id='category-filter',
                options=[{"label": c, "value": c} for c in sorted(df['category'].unique())],
                value=[],
                multi=True,
                placeholder="Select categories..."
            )
        ], style={"width": "20%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.P("Supplier:"),
            dcc.Dropdown(
                id='supplier-filter',
                options=[{"label": s, "value": s} for s in sorted(df['supplier'].unique())],
                value=[],
                multi=True,
                placeholder="Select suppliers..."
            )
        ], style={"width": "20%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.P("Show Anomalies Only:"),
            dcc.RadioItems(
                id='anomaly-filter',
                options=[
                    {'label': 'All Data', 'value': 'all'},
                    {'label': 'Anomalies Only', 'value': 'anomalies'}
                ],
                value='all',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            )
        ], style={"width": "15%", "display": "inline-block", "padding": "10px"})
    ], style={"background-color": "#f2f2f2", "padding": "10px", "margin": "10px 0px"}),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.H4("Total Records", style={"text-align": "center"}),
            html.P(id="total-records", style={"text-align": "center", "font-size": "24px", "font-weight": "bold"})
        ], className="kpi-card"),
        
        html.Div([
            html.H4("Detected Anomalies", style={"text-align": "center"}),
            html.P(id="total-anomalies", style={"text-align": "center", "font-size": "24px", "font-weight": "bold", "color": "#e74c3c"})
        ], className="kpi-card"),
        
        html.Div([
            html.H4("Anomaly Rate", style={"text-align": "center"}),
            html.P(id="anomaly-rate", style={"text-align": "center", "font-size": "24px", "font-weight": "bold"})
        ], className="kpi-card"),
        
        html.Div([
            html.H4("Most Affected Region", style={"text-align": "center"}),
            html.P(id="most-affected-region", style={"text-align": "center", "font-size": "24px", "font-weight": "bold"})
        ], className="kpi-card"),
    ], style={"display": "flex", "justify-content": "space-between", "margin": "20px 0px"}),
    
    # Main charts row
    html.Div([
        # Left column - Time series
        html.Div([
            html.H3("Supply Chain Metrics Over Time", style={"text-align": "center"}),
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': 'Lead Time (days)', 'value': 'lead_time_days'},
                    {'label': 'Transport Time (days)', 'value': 'transport_time_days'},
                    {'label': 'Inventory Coverage (days)', 'value': 'inventory_coverage_days'},
                    {'label': 'Capacity Utilization (%)', 'value': 'capacity_utilization'},
                    {'label': 'Demand Forecast', 'value': 'demand_forecast'},
                    {'label': 'Order Quantity', 'value': 'order_quantity'}
                ],
                value='lead_time_days',
                clearable=False
            ),
            dcc.Graph(id="time-series-chart")
        ], style={"width": "49%", "display": "inline-block", "vertical-align": "top"}),
        
        # Right column - Anomaly distribution
        html.Div([
            html.H3("Anomaly Distribution by Category", style={"text-align": "center"}),
            dcc.Graph(id="category-anomaly-chart")
        ], style={"width": "49%", "display": "inline-block", "vertical-align": "top"})
    ]),
    
    # Second charts row
    html.Div([
        # Left column - Geographic distribution
        html.Div([
            html.H3("Regional Anomaly Distribution", style={"text-align": "center"}),
            dcc.Graph(id="regional-chart")
        ], style={"width": "49%", "display": "inline-block", "vertical-align": "top"}),
        
        # Right column - Correlation matrix
        html.Div([
            html.H3("Feature Correlation Matrix", style={"text-align": "center"}),
            dcc.Graph(id="correlation-chart")
        ], style={"width": "49%", "display": "inline-block", "vertical-align": "top"})
    ]),
    
    # Anomaly table
    html.Div([
        html.H3("Detected Anomalies", style={"text-align": "center"}),
        dash_table.DataTable(
            id='anomaly-table',
            columns=[
                {"name": "Date", "id": "date"},
                {"name": "Region", "id": "region"},
                {"name": "Category", "id": "category"},
                {"name": "Supplier", "id": "supplier"},
                {"name": "Lead Time", "id": "lead_time_days"},
                {"name": "Transport Time", "id": "transport_time_days"},
                {"name": "Inventory Coverage", "id": "inventory_coverage_days"},
                {"name": "Capacity Utilization", "id": "capacity_utilization"},
                {"name": "Risk Score", "id": "risk_score"}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '100px'
            },
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'risk_score', 'filter_query': '{risk_score} > 80'},
                    'backgroundColor': '#e74c3c',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'risk_score', 'filter_query': '{risk_score} > 50 && {risk_score} <= 80'},
                    'backgroundColor': '#f39c12',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'risk_score', 'filter_query': '{risk_score} <= 50'},
                    'backgroundColor': '#27ae60',
                    'color': 'white'
                }
            ],
            page_size=10
        )
    ], style={"margin": "20px 0px"}),
    
    # Footer
    html.Div([
        html.P("FMCG Supply Chain Anomaly Detection Dashboard | Hackathon Project", 
               style={"margin-bottom": "0px", "color": "white"})
    ], style={"text-align": "center", "padding": "1rem", "background-color": "#2c3e50", "margin-top": "20px"})
], style={"max-width": "1200px", "margin": "0 auto", "font-family": "Arial, sans-serif"})

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .kpi-card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 15px;
                width: 23%;
            }
            
            @media (max-width: 768px) {
                .kpi-card {
                    width: 48%;
                    margin-bottom: 10px;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# Define callback functions
@app.callback(
    [Output("total-records", "children"),
     Output("total-anomalies", "children"),
     Output("anomaly-rate", "children"),
     Output("most-affected-region", "children")],
    [Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("region-filter", "value"),
     Input("category-filter", "value"),
     Input("supplier-filter", "value"),
     Input("anomaly-filter", "value")]
)
def update_kpis(start_date, end_date, regions, categories, suppliers, anomaly_filter):
    # Filter data based on inputs
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date)
    ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(suppliers)]
    
    if anomaly_filter == 'anomalies':
        filtered_df = filtered_df[filtered_df['anomaly_detected'] == 1]
    
    # Calculate KPIs
    total_records = len(filtered_df)
    total_anomalies = filtered_df['anomaly_detected'].sum()
    anomaly_rate = f"{(total_anomalies / total_records * 100):.2f}%" if total_records > 0 else "0.00%"
    
    # Most affected region
    if total_anomalies > 0:
        most_affected_region = filtered_df[filtered_df['anomaly_detected'] == 1]['region'].mode()[0]
    else:
        most_affected_region = "N/A"
    
    return total_records, total_anomalies, anomaly_rate, most_affected_region


@app.callback(
    Output("time-series-chart", "figure"),
    [Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("region-filter", "value"),
     Input("category-filter", "value"),
     Input("supplier-filter", "value"),
     Input("anomaly-filter", "value"),
     Input("metric-selector", "value")]
)
def update_time_series(start_date, end_date, regions, categories, suppliers, anomaly_filter, metric):
    # Filter data
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date)
    ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(suppliers)]
    
    if anomaly_filter == 'anomalies':
        filtered_df = filtered_df[filtered_df['anomaly_detected'] == 1]
    
    # Create time series chart
    fig = px.line(
        filtered_df, 
        x="date", 
        y=metric, 
        title=f"{metric.replace('_', ' ').title()} Over Time",
        labels={"date": "Date", metric: metric.replace('_', ' ').title()}
    )
    
    # Highlight anomalies
    if anomaly_filter == 'all':
        anomalies_df = filtered_df[filtered_df['anomaly_detected'] == 1]
        fig.add_trace(
            go.Scatter(
                x=anomalies_df['date'],
                y=anomalies_df[metric],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Anomalies'
            )
        )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode="x unified"
    )
    
    return fig


@app.callback(
    Output("category-anomaly-chart", "figure"),
    [Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("region-filter", "value"),
     Input("category-filter", "value"),
     Input("supplier-filter", "value"),
     Input("anomaly-filter", "value")]
)
def update_category_anomaly_chart(start_date, end_date, regions, categories, suppliers, anomaly_filter):
    # Filter data
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date)
    ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(suppliers)]
    
    if anomaly_filter == 'anomalies':
        filtered_df = filtered_df[filtered_df['anomaly_detected'] == 1]
    
    # Group by category and count anomalies
    category_anomalies = filtered_df.groupby('category')['anomaly_detected'].sum().reset_index()
    
    # Create bar chart
    fig = px.bar(
        category_anomalies, 
        x="category", 
        y="anomaly_detected", 
        title="Anomalies by Category",
        labels={"category": "Category", "anomaly_detected": "Number of Anomalies"}
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of Anomalies"
    )
    
    return fig


@app.callback(
    Output("regional-chart", "figure"),
    [Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("region-filter", "value"),
     Input("category-filter", "value"),
     Input("supplier-filter", "value"),
     Input("anomaly-filter", "value")]
)
def update_regional_chart(start_date, end_date, regions, categories, suppliers, anomaly_filter):
    # Filter data
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date)
    ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(suppliers)]
    
    if anomaly_filter == 'anomalies':
        filtered_df = filtered_df[filtered_df['anomaly_detected'] == 1]
    
    # Group by region and count anomalies
    regional_anomalies = filtered_df.groupby('region')['anomaly_detected'].sum().reset_index()
    
    # Create bar chart
    fig = px.bar(
        regional_anomalies, 
        x="region", 
        y="anomaly_detected", 
        title="Anomalies by Region",
        labels={"region": "Region", "anomaly_detected": "Number of Anomalies"}
    )
    
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Number of Anomalies"
    )
    
    return fig


@app.callback(
    Output("correlation-chart", "figure"),
    [Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("region-filter", "value"),
     Input("category-filter", "value"),
     Input("supplier-filter", "value"),
     Input("anomaly-filter", "value")]
)
def update_correlation_chart(start_date, end_date, regions, categories, suppliers, anomaly_filter):
    # Filter data
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date)
    ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(suppliers)]
    
    if anomaly_filter == 'anomalies':
        filtered_df = filtered_df[filtered_df['anomaly_detected'] == 1]
    
    # Select numerical features for correlation
    numerical_features = [
        'order_quantity', 'lead_time_days', 'transport_time_days', 
        'inventory_level', 'demand_forecast', 'production_capacity', 
        'inventory_coverage_days', 'capacity_utilization'
    ]
    
    corr = filtered_df[numerical_features].corr()
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis',
            zmin=-1,
            zmax=1
        )
    )
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig


@app.callback(
    Output("anomaly-table", "data"),
    [Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("region-filter", "value"),
     Input("category-filter", "value"),
     Input("supplier-filter", "value"),
     Input("anomaly-filter", "value")]
)
def update_anomaly_table(start_date, end_date, regions, categories, suppliers, anomaly_filter):
    # Filter data
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date)
    ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(suppliers)]
    
    if anomaly_filter == 'anomalies':
        filtered_df = filtered_df[filtered_df['anomaly_detected'] == 1]
    
    # Add a risk score (for demo purposes)
    filtered_df['risk_score'] = np.random.randint(0, 100, size=len(filtered_df))
    
    # Prepare data for table
    table_data = filtered_df[[
        'date', 'region', 'category', 'supplier', 
        'lead_time_days', 'transport_time_days', 
        'inventory_coverage_days', 'capacity_utilization', 
        'risk_score'
    ]].to_dict('records')
    
    return table_data


# Run the app
if __name__ == "__main__":
    app.server.run(debug=True)