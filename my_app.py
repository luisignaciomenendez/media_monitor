import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Read data
df_group = pd.read_csv('topic_composition.csv')
df_entropy = pd.read_csv('entropy.csv')

# Convert date columns to proper date format
df_group['date'] = pd.to_datetime(df_group['date']).dt.date
df_entropy['date'] = pd.to_datetime(df_entropy['date']).dt.date

# Sort entropy data by date
df_entropy = df_entropy.sort_values('date')

# Compute smoothed entropy (rolling average over 7 days)
df_entropy['smoothed_entropy'] = df_entropy['entropy'].rolling(window=7, center=True, min_periods=1).mean()

# Define channel-specific colors and positions for clustering
channel_colors = {
    'tve': 'purple',
    'a3': 'orange',
    'la6': 'green',
    't5': 'blue',
    'cuatro': 'red'
}

channel_positions = {
    'tve': (0, 0),
    'a3': (3, 0),
    'la6': (-3, 0),
    't5': (0, 3),
    'cuatro': (0, -3)
}

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment (e.g., gunicorn)

# Function to get topic composition for a specific date
def get_topic_composition(date, df):
    df_filtered = df[df['date'] == date.date()]
    topics = df_filtered['words_topic'].tolist()
    rel_times = df_filtered['rel_time'].tolist()
    channels = df_filtered['channel'].tolist()
    return topics, rel_times, channels

# Layout of the app
app.layout = html.Div([
    html.H1("Interactive Topic Composition", style={'textAlign': 'center'}),
    dcc.Graph(id='entropy-graph'),
    dcc.Graph(id='topic-composition')
], style={'backgroundColor': '#f9f9f9', 'padding': '20px'})

# Callback to create the enhanced entropy time series plot
@app.callback(
    Output('entropy-graph', 'figure'),
    [Input('topic-composition', 'figure')]  # Dummy input to trigger update
)
def update_entropy_plot(_):
    fig = go.Figure()

    # Plot the raw entropy time series (light, dotted line)
    fig.add_trace(go.Scatter(
        x=df_entropy['date'],
        y=df_entropy['entropy'],
        mode='lines+markers',
        marker=dict(size=6, color='lightblue'),
        line=dict(color='lightblue', dash='dot'),
        name='Raw Entropy',
        opacity=0.5
    ))
    
    # Plot the smoothed entropy (solid, prominent line)
    fig.add_trace(go.Scatter(
        x=df_entropy['date'],
        y=df_entropy['smoothed_entropy'],
        mode='lines',
        line=dict(color='darkblue', width=3),
        name='Smoothed Entropy'
    ))
    
    # Update layout with range slider, selectors, and transitions
    fig.update_layout(
        title="Entropy Time Series with Smoothing",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis_title="Entropy",
        hovermode='x unified',
        template="plotly_white",
        transition={'duration': 500}
    )
    return fig

# Callback to update the topic composition graph based on click in the entropy graph
@app.callback(
    Output('topic-composition', 'figure'),
    [Input('entropy-graph', 'clickData')]
)
def update_topic_composition(clickData):
    if clickData is None:
        return go.Figure()  # Return an empty figure until a click occurs

    # Extract the clicked date
    clicked_date = pd.to_datetime(clickData['points'][0]['x'])
    topics, rel_times, channels = get_topic_composition(clicked_date, df_group)
    
    fig = go.Figure()
    
    # Normalize bubble sizes relative to the maximum rel_time for consistency
    max_rel = max(rel_times) if rel_times else 1

    # Add dummy traces for each channel to generate a legend
    for channel, color in channel_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=channel.upper(),
            showlegend=True
        ))
    
    # Add bubble traces per channel with refined Gaussian jitter for natural clustering
    for channel, color in channel_colors.items():
        channel_indices = [i for i, ch in enumerate(channels) if ch == channel]
        if not channel_indices:
            continue
        
        center_x, center_y = channel_positions[channel]
        x_positions = [center_x + np.random.normal(scale=0.5) for _ in channel_indices]
        y_positions = [center_y + np.random.normal(scale=0.5) for _ in channel_indices]
        sizes = [((rel_times[i] / max_rel) * 100) for i in channel_indices]

        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers',
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.8,
                sizemode='area'
            ),
            text=[f"{topics[i]}: {rel_times[i]:.2%}" for i in channel_indices],
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add annotations for each channel cluster center
    for channel, pos in channel_positions.items():
        fig.add_annotation(
            x=pos[0],
            y=pos[1],
            text=channel.upper(),
            showarrow=False,
            font=dict(color=channel_colors[channel], size=14, family="Arial")
        )
    
    fig.update_layout(
        title=f"Topic Composition for {clicked_date.date()}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        template="plotly_white",
        transition={'duration': 500},
        plot_bgcolor='#ffffff'
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
