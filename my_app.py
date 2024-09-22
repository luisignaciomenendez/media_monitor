#%% 

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
# Assuming df_group and df_entropy are already created

df_group=pd.read_csv('topic_composition.csv')
df_entropy=pd.read_csv('entropy.csv')


#%% 


df_group['date'] = pd.to_datetime(df_group['date']).dt.date  # Ensure the date column is in the correct format
df_entropy['date'] = pd.to_datetime(df_entropy['date']).dt.date  # Ensure the date column is in the correct format

channel_colors = {
    'tve': 'purple',
    'a3': 'orange',
    'la6': 'green',
    't5': 'blue',
    'cuatro': 'red'
}

channel_positions = {
    'tve': (0, 0),      # Center for TVE
    'a3': (3, 0),       # Center for A3
    'la6': (-3, 0),     # Center for La6
    't5': (0, 3),       # Center for T5
    'cuatro': (0, -3),  # Center for Cuatro
}

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Use this for gunicorn


# Function to get topic composition for a specific date
def get_topic_composition(date, df):
    df_filtered = df[df['date'] == date.date()]  # Use date.date() to ensure proper comparison
    topics = df_filtered['words_topic'].tolist()
    rel_times = df_filtered['rel_time'].tolist()
    channels = df_filtered['channel'].tolist()
    return topics, rel_times, channels


# Layout of the app
app.layout = html.Div([
    html.H1("Interactive Topic Composition"),
    dcc.Graph(id='entropy-graph'),  # Entropy time series plot
    dcc.Graph(id='topic-composition'),  # Topic composition plot
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
        template="plotly"
    )

    return fig

# Callback to update the topic composition plot based on click
@app.callback(
    Output('topic-composition', 'figure'),
    [Input('entropy-graph', 'clickData')]  # Get the clicked point from entropy graph
)

def update_topic_composition(clickData):
    if clickData is None:
        return go.Figure()  # Empty figure before any click

    # Get clicked date
    clicked_date = pd.to_datetime(clickData['points'][0]['x'])

    # Get topic composition for clicked date
    topics, rel_times, channels = get_topic_composition(clicked_date, df_group)

    # Create bubble plot
    fig = go.Figure()

    # Get unique topics to create sub-clusters within each channel
    unique_topics = list(set(topics))

    for channel in channel_colors.keys():
        fig.add_trace(go.Scatter(
            x=[None],  # No actual data for the legend trace
            y=[None],  # No actual data for the legend trace
            mode='markers',
            marker=dict(
                size=10,  # Dummy size for the legend
                color=channel_colors[channel]
            ),
            name=channel,  # Show only the channel name in the legend
            showlegend=True  # Enable legend for this dummy trace
        ))

# Now proceed with adding the actual bubble traces
    for channel in channel_colors.keys():
        channel_indices = [i for i, ch in enumerate(channels) if ch == channel]

        if channel_indices:
            # Get cluster center for this channel
            center_x, center_y = channel_positions[channel]

            # Adjust positions for more random distribution around the center
            x_positions = [center_x + np.random.uniform(-1, 1) for _ in channel_indices]
            y_positions = [center_y + np.random.uniform(-1, 1) for _ in channel_indices]

            # Add actual bubble trace without showing the legend
            fig.add_trace(go.Scatter(
                x=x_positions,  # Position near the cluster center
                y=y_positions,  # Position near the cluster center
                mode='markers',
                marker=dict(
                    size=[rel_times[i] * 3000 for i in channel_indices],  # Bubble size proportional to rel_time
                    color=channel_colors[channel],  # Channel-specific color
                    opacity=0.8,
                    sizemode='area'
                ),
                text=[f"{topics[i]}: {rel_times[i]:.2%}" for i in channel_indices],  # Show topic and percentage on hover
                hoverinfo='text',
                showlegend=False  # Disable legend for the actual bubble trace
            ))



    fig.update_layout(
        title=f"Topic Composition for {clicked_date}",
        xaxis_title=None,  # Remove x-axis title
        xaxis_showticklabels=False,  # Hide x-axis labels
        yaxis_showticklabels=False,  # Hide y-axis labels
        yaxis_title=None,  # Remove y-axis title
        height=600,  # Set height to ensure proper display
        template="plotly"
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8051)

#%% 

# Run the app

# if __name__ == '__main__':
#     app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))

