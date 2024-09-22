#%% 

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
# Assuming df_group and df_entropy are already created

df_group=pd.read_csv('topic_composition.csv')
df_entropy=pd.read_csv('entropy.csv')


#%% 


# Ensure dates are in the correct format
df_group['date'] = pd.to_datetime(df_group['date']).dt.date
df_entropy['date'] = pd.to_datetime(df_entropy['date']).dt.date

# Define colors for channels
channel_colors = {
    'tve': 'purple',
    'a3': 'orange',
    'la6': 'green',
    't5': 'blue',
    'cuatro': 'red'
}

# Cluster centers for each channel (adjust values to customize positions)
channel_positions = {
    'tve': (0, 0),      # Center for TVE
    'a3': (3, 0),       # Center for A3
    'la6': (-3, 0),     # Center for La6
    't5': (0, 3),       # Center for T5
    'cuatro': (0, -3),  # Center for Cuatro
}

# Initialize Dash app
app = dash.Dash(__name__)

# Function to get topic composition for a specific date
def get_topic_composition(date, df):
    df_filtered = df[df['date'] == date]
    topics = df_filtered['words_topic'].tolist()
    rel_times = df_filtered['rel_time'].tolist()
    channels = df_filtered['channel'].tolist()
    return topics, rel_times, channels

# Layout of the app
app.layout = html.Div([
    html.H1("Interactive Topic Composition"),
    dcc.Graph(id='entropy-graph'),  # Entropy time series plot
    dcc.Graph(id='topic-composition'),  # Topic composition bubble plot
])

# Create the initial entropy time series plot
@app.callback(
    Output('entropy-graph', 'figure'),
    [Input('topic-composition', 'figure')]  # Trigger an update if topic composition changes
)
def update_entropy_plot(_):
    fig = go.Figure()

    # Add entropy time series
    fig.add_trace(go.Scatter(
        x=df_entropy['date'],
        y=df_entropy['entropy'],
        mode='lines+markers',
        marker=dict(size=10, color='blue'),
        name='Entropy Time Series'
    ))

    fig.update_layout(
        title="Entropy Time Series",
        xaxis_title="Date",
        yaxis_title="Entropy",
        hovermode='closest',
        template="plotly_dark"
    )

    return fig
import numpy as np

@app.callback(
    Output('topic-composition', 'figure'),
    [Input('entropy-graph', 'clickData')]
)
def update_topic_composition(clickData):
    if clickData is None:
        return go.Figure()  # Empty figure before any click

    # Get clicked date
    clicked_date = pd.to_datetime(clickData['points'][0]['x']).date()

    # Get topic composition for clicked date
    topics, rel_times, channels = get_topic_composition(clicked_date, df_group)

    # Create bubble plot
    fig = go.Figure()

    # Get unique topics to create sub-clusters within each channel
    unique_topics = list(set(topics))

    for channel in channel_colors.keys():
        channel_indices = [i for i, ch in enumerate(channels) if ch == channel]

        if channel_indices:
            # Get cluster center for this channel
            center_x, center_y = channel_positions[channel]

            # For each unique topic, create sub-clusters with some offsets
            # Adjust the random offsets for more organic placement
            for i in channel_indices:
                # Use more random offsets not strictly tied to rel_time
                topic_offset_x = np.random.uniform(-1, 1) * (1 - rel_times[i]) * 2  # Randomize x positioning
                topic_offset_y = np.random.uniform(-1, 1) * (1 - rel_times[i]) * 2  # Randomize y positioning

                fig.add_trace(go.Scatter(
                    x=[center_x + topic_offset_x],  # Position relative to channel center with random x offset
                    y=[center_y + topic_offset_y],  # Position relative to channel center with random y offset
                    mode='markers',
                    marker=dict(
                        size=[rel_times[i] * 1200],  # Bubble size proportional to rel_time
                        color=channel_colors[channel],  # Channel-specific color
                        opacity=0.8,
                        sizemode='area'
                    ),
                    name=f"{channel}: {topics[i]}",
                    text=[f"{topics[i]}: {rel_times[i]:.2%}"],  # Show topic and percentage on hover
                    hoverinfo='text',
                    showlegend=False
                ))



    fig.update_layout(
        title=f"Topic Composition for {clicked_date}",
        xaxis_title=None,  # Remove x-axis title
        xaxis_showticklabels=False,  # Hide x-axis labels
        yaxis_showticklabels=False,  # Hide y-axis labels
        yaxis_title=None,  # Remove y-axis title
        template="plotly_dark"
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
