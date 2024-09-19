import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Assuming df_group and df_entropy are already created

df_group=pd.read_csv('./data/topic_composition.csv')
df_entropy=pd.read_csv('./data/entropy.csv')

df_group['date'] = pd.to_datetime(df_group['date']).dt.date  # Ensure the date column is in the correct format
df_entropy['date'] = pd.to_datetime(df_entropy['date']).dt.date  # Ensure the date column is in the correct format

channel_colors = {
    'tve': 'purple',
    'a3': 'orange',
    'la6': 'green',
    't5': 'blue',
    'cuatro': 'red'
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
        template="plotly_dark"
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

    # Add trace with channel-specific bubble colors
    # Create bar plot for each channel and topic

    for channel in channel_colors.keys():
        channel_indices = [i for i, ch in enumerate(channels) if ch == channel]
        
        if channel_indices:
            # Sort topics and relative times in descending order
            sorted_indices = sorted(channel_indices, key=lambda i: rel_times[i], reverse=True)
            
            # Combine topic words and percentages for display inside the bars
            text_labels = [f"{topics[i]}: {rel_times[i]:.2%}" for i in sorted_indices]
            
            fig.add_trace(go.Bar(
                x=[channel] * len(sorted_indices),  # One bar per channel
                y=[rel_times[i] for i in sorted_indices],  # Stack relative times of topics, sorted
                text=text_labels,  # Display topic words and percentage inside the bar
                textposition='inside',  # Show text inside the bar
                name=channel,
                marker=dict(
                    color=channel_colors[channel]  # Channel-specific color
                ),
                width=0.7,  # Make bars wider
                showlegend=True
            ))

    # Update layout to refine the plot
    fig.update_layout(
        title="Topic Composition by Channel (Ordered by Relative Time)",
        xaxis_title="Channel",
        yaxis_title="Relative Time Spent on Topics",
        bargap=0.1,  # Reduce the gap between bars
        barmode='stack',  # Stack the bars
        uniformtext_minsize=10,  # Uniform text size
        uniformtext_mode='hide',  # Hide text that doesn’t fit
        template="plotly_dark"
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
