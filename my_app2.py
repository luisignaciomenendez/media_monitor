import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load your dataset (update the CSV path if necessary)
#read parquet. 

df = pd.read_parquet('final_22_01_25_chatgpt_reduced.parquet')


df['date'] = pd.to_datetime(df['date'])

# Define tone metrics for individual parties and their labels
tone_metrics = ["sent_PSOE", "sent_PP", "sent_UP", "sent_VOX"]
tone_labels = {
    "sent_PSOE": "PSOE",
    "sent_PP": "PP",
    "sent_UP": "UP",
    "sent_VOX": "VOX"
}

# Define channel order and colors
channels_order = ['tve', 't5', 'la6', 'cuatro', 'a3']
channel_colors = {
    'tve': 'purple',
    't5': 'blue',
    'la6': 'green',
    'cuatro': 'red',
    'a3': 'orange'
}

# Get sorted list of channels from data (if available)
channels = sorted(df['channel'].unique())

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

app.layout = html.Div([
    html.H1("Interactive Tone Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Select Tone Metric:"),
            dcc.Dropdown(
                id='tone-metric',
                options=[{'label': tone_labels[m], 'value': m} for m in tone_metrics],
                value='sent_PSOE',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20px'}),
        
        html.Div([
            html.Label("Select Channels:"),
            dcc.Dropdown(
                id='channels-dropdown',
                options=[{'label': ch.upper(), 'value': ch} for ch in channels_order],
                value=channels_order,
                multi=True
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20px'}),
        
        html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df['date'].min().date(),
                max_date_allowed=df['date'].max().date(),
                start_date=df['date'].min().date(),
                end_date=df['date'].max().date()
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20px'})
    ], style={'padding': '20px 0'}),
    
    dcc.Graph(id='time-series-graph'),
    dcc.Graph(id='box-plot-graph')
], style={'maxWidth': '1200px', 'margin': '0 auto', 'fontFamily': 'Arial'})

@app.callback(
    [Output('time-series-graph', 'figure'),
     Output('box-plot-graph', 'figure')],
    [Input('tone-metric', 'value'),
     Input('channels-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_graphs(tone_metric, selected_channels, start_date, end_date):
    # Filter data by date range and selected channels
    mask = (
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] <= pd.to_datetime(end_date)) &
        (df['channel'].isin(selected_channels))
    )
    filtered_df = df[mask]
    
    # --- Time Series Graph ---
    # Group data by date and channel, taking the mean tone
    ts_df = filtered_df.groupby(['date', 'channel'])[tone_metric].mean().reset_index()
    ts_df = ts_df.sort_values('date')
    
    # Apply 7-day rolling average per channel (smoothing)
    ts_df['smoothed'] = ts_df.groupby('channel')[tone_metric].transform(
        lambda x: x.rolling(window=7, min_periods=1, center=True).mean()
    )
    
    fig_ts = go.Figure()
    for ch in selected_channels:
        df_ch = ts_df[ts_df['channel'] == ch]
        fig_ts.add_trace(go.Scatter(
            x=df_ch['date'],
            y=df_ch['smoothed'],
            mode='lines+markers',
            name=ch.upper(),
            line=dict(color=channel_colors.get(ch, 'gray'))
        ))
    
    fig_ts.update_layout(
        title=f"Time Series of {tone_labels[tone_metric]} Tone (7-Day Smoothed)",
        xaxis_title="Date",
        yaxis_title="Average Tone",
        hovermode='x unified'
    )
    
    # --- Box Plot Graph ---
    fig_box = go.Figure()
    for ch in selected_channels:
        df_ch = filtered_df[filtered_df['channel'] == ch]
        fig_box.add_trace(go.Box(
            y=df_ch[tone_metric],
            name=ch.upper(),
            boxmean=True,
            marker_color=channel_colors.get(ch, 'gray')
        ))
    
    fig_box.update_layout(
        title=f"Distribution of {tone_labels[tone_metric]} Tone by Channel",
        xaxis_title="Channel",
        yaxis_title="Tone"
    )
    
    return fig_ts, fig_box

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
