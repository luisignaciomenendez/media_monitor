#%% LAST DESCRIPTIVES BEFORE STATA
# !pip3 install --upgrade bertopic
import pandas as pd
import os
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bertopic import BERTopic
from unidecode import unidecode
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

#%%  version of bertopic
import bertopic
bertopic.__version__




#%%

# df_exp=pd.read_csv('/Users/luisignaciomenendezgarcia/Dropbox/MediaBias/Data/geca_clean/matched_03_11.csv')
# df_exp=pd.read_csv('/Users/luisignaciomenendezgarcia/Dropbox/MediaBias/Data/geca_clean/matched_06_11.csv')
df_exp=pd.read_csv('/Users/luisignaciomenendezgarcia/Dropbox/MediaBias/Data/vtt_minutes/panel_vtt_09_09_2024.csv')
#%% 
df_exp.rename(columns={'date':'date','channel':'channel','text':'noticias'},inplace=True)

#%%


psoe = ["partido socialista", "psoe", "socialistas",  "pedro sanchez", "susana diaz", "ximo puebla","carmen calvo","jose luis abalos", "meritxell batet","margarita robles","jose luis abalos","maria jesus montero"]
pp = ["partido popular", "pp", "feijoo", "cospedal", "juanma moreno", "ayuso","rafael hernando","almeida","cristobal montoro", "rafael hernando",
    "montoro",
    "feijo",
    "ignacio cosido",
    "fernando martinez-maillo",
    "xavier garcia albiol",
    "alicia sanchez-camacho",
    "zoido",
    "rafael catala",
    "ruiz-gallardon",
    "esperanza aguirre",
    "aznar",
    "rajoy"]


up = ["unidas podemos", "pablo iglesias", "montero", "yolanda diaz", "alberto garzon","belarra","echenique",
    "rafa mayoral",
    "isabel serra",
    "yolanda diaz"
    "gloria elizo",
    "noelia vera",
    "juan antonio delgado",
    "sergio pascual",
    "rita maestre",
    "pilar garrido",
    "josé maria guzman",
    "alejandra jacinto",
    "jose luis ramos",
    "maria pozo",
    "alberto rodriguez",
    "rosa maria medina",
    "lucia muñoz",
    "maria merchan"]
cs = [ "cs", "erregureña", "carrizosa", "albert rivera", "arrimadas"]
vox = ["vox","abascal", "tamames","espinosa de los monteros", "rocio monasterio", "ortega smith","jorge bucay"]

montero=['montero','irene montero','monterio']
feijoo=['feijoo','feijo','feijóo']
sanchez=['pedro sanchez','presidente sanchez']
abascal=['abascal','santiago abascal']
diaz=['yolanda diaz','ministra diaz']

#%%

def count_matches(string, lst):
    #! important: decode the accents in the string
    count = 0
    new_str=unidecode(string.lower())
    for word in lst:
        if word in new_str:
            count += 1
    return count

df_exp['total_words'] = df_exp['noticias'].apply(lambda x: len(x.split()))

df_exp['pp_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, pp))
df_exp['psoe_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, psoe))
df_exp['up_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, up))
df_exp['cs_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, cs))
df_exp['vox_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, vox))
# df_exp['montero_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, montero))
df_exp['diaz_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, diaz))
df_exp['feijoo_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, feijoo))
df_exp['sanchez_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, sanchez))
df_exp['abascal_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, abascal))

# df_exp['my_list_matches'] = df_exp['noticias'].apply(lambda x: count_matches(x, my_list))

#%%

#* Get rid of the very first tve news

# df_exp.query('psoe_matches>0')['noticias']

# noticia=df_exp.iloc[0].noticias

min_date=min(df_exp[df_exp['channel']=='a3']['date'])

# min_date='2023-02-01'
# max_date='2023-03-01'
df_exp=df_exp[(df_exp['date']>min_date) ]

df_exp['date'] = pd.to_datetime(df_exp['date'])



#%%
import pandas as pd
import plotly.express as px

# Assuming df_exp is already loaded and processed

# Convert 'date' column to datetime
df_exp['date'] = pd.to_datetime(df_exp['date'])

# Calculate relative mentions
df_exp['pp_mentions_rel'] = df_exp['pp_matches'] / df_exp['total_words']
df_exp['psoe_mentions_rel'] = df_exp['psoe_matches'] / df_exp['total_words']
df_exp['up_mentions_rel'] = df_exp['up_matches'] / df_exp['total_words']
df_exp['cs_mentions_rel'] = df_exp['cs_matches'] / df_exp['total_words']
df_exp['vox_mentions_rel'] = df_exp['vox_matches'] / df_exp['total_words']

# Group by date and channel, then calculate the average relative mentions
df_avg_mentions = df_exp.groupby(['date', 'channel']).agg({
    'pp_mentions_rel': 'mean',
    'psoe_mentions_rel': 'mean',
    'up_mentions_rel': 'mean',
    'cs_mentions_rel': 'mean',
    'vox_mentions_rel': 'mean'
}).reset_index()


#%% 


import pandas as pd
import plotly.express as px

# Assuming df_exp is already loaded and processed

# Convert 'date' column to datetime
df_exp['date'] = pd.to_datetime(df_exp['date'])

# Weekly aggregation
df_exp['week'] = df_exp['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Calculate relative mentions
df_exp['pp_mentions_rel'] = df_exp['pp_matches'] / df_exp['total_words']
df_exp['psoe_mentions_rel'] = df_exp['psoe_matches'] / df_exp['total_words']
df_exp['up_mentions_rel'] = df_exp['up_matches'] / df_exp['total_words']
df_exp['cs_mentions_rel'] = df_exp['cs_matches'] / df_exp['total_words']
df_exp['vox_mentions_rel'] = df_exp['vox_matches'] / df_exp['total_words']

# Group by week and channel, then calculate the average relative mentions
df_avg_mentions_weekly = df_exp.groupby(['week', 'channel']).agg({
    'pp_mentions_rel': 'mean',
    'psoe_mentions_rel': 'mean',
    'up_mentions_rel': 'mean',
    'cs_mentions_rel': 'mean',
    'vox_mentions_rel': 'mean'
}).reset_index()

# Channel color mapping
channel_colors = {
    'tve': 'purple',
    'a3': 'orange',
    'la6': 'green',
    't5': 'blue',
    'cuatro': 'red'
}

# Restructure data for Plotly: Long-format to accommodate multiple parties/politicians
df_avg_mentions_long = pd.melt(df_avg_mentions_weekly, 
                               id_vars=['week', 'channel'], 
                               value_vars=['pp_mentions_rel', 'psoe_mentions_rel', 'up_mentions_rel', 'cs_mentions_rel', 'vox_mentions_rel'],
                               var_name='Party', value_name='Average Mentions')

# Create the interactive plot using Plotly
fig = px.line(
    df_avg_mentions_long, 
    x='week', 
    y='Average Mentions',
    color='channel',  # Color by channel
    # facet_col='channel',  # Facet by channel
    labels={
        "Average Mentions": "Average Relative Mentions",
        "week": "Week"
    },
    title="Weekly Average Relative Mentions Per Channel",
    color_discrete_map=channel_colors,  # Set colors for channels
    template="plotly_dark"
)

# Remove legend for parties
fig.update_layout(showlegend=True, legend_title="Channel")

# Format the x-axis (date)
fig.update_xaxes(
    dtick="M1",  # Monthly ticks
    tickformat="%b %Y",  # Format the ticks as 'Month Year'
    ticklabelmode="period"
)

# Show the interactive plot
fig.show()



#%% 


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming df_exp is already loaded and processed

# Convert 'date' column to datetime
df_exp['date'] = pd.to_datetime(df_exp['date'])

# Weekly aggregation
df_exp['week'] = df_exp['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Calculate relative mentions
df_exp['pp_mentions_rel'] = df_exp['pp_matches'] / df_exp['total_words']
df_exp['psoe_mentions_rel'] = df_exp['psoe_matches'] / df_exp['total_words']
df_exp['up_mentions_rel'] = df_exp['up_matches'] / df_exp['total_words']
df_exp['cs_mentions_rel'] = df_exp['cs_matches'] / df_exp['total_words']
df_exp['vox_mentions_rel'] = df_exp['vox_matches'] / df_exp['total_words']


df_exp['pp_mentions_rel']= df_exp['pp_mentions_rel']*100
df_exp['psoe_mentions_rel']= df_exp['psoe_mentions_rel']*100
df_exp['up_mentions_rel']= df_exp['up_mentions_rel']*100

df_exp['vox_mentions_rel']= df_exp['vox_mentions_rel']*100

# Group by week and channel, then calculate the average relative mentions
df_avg_mentions_weekly = df_exp.groupby(['week', 'channel']).agg({
    'pp_mentions_rel': 'mean',
    'psoe_mentions_rel': 'mean',
    'up_mentions_rel': 'mean',
    'cs_mentions_rel': 'mean',
    'vox_mentions_rel': 'mean'
}).reset_index()

# Channel color mapping
channel_colors = {
    'tve': 'purple',
    'a3': 'orange',
    'la6': 'green',
    't5': 'blue',
    'cuatro': 'red'
}

channel_line_styles = {
    'tve': 'solid',       # Solid line for 'tve'
    'a3': 'dash',         # Dashed line for 'a3'
    'la6': 'dot',         # Dotted line for 'la6'
    't5': 'dashdot',      # Dash-dot line for 't5'
    'cuatro': 'dash'      # Dashed line for 'cuatro'
}


#%% 

# Create an empty figure
fig = go.Figure()

# Add traces (lines) for each channel, but initially only show for one party (e.g., pp_mentions_rel)
for channel in df_avg_mentions_weekly['channel'].unique():
    fig.add_trace(go.Scatter(
        x=df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['week'],
        y=df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['pp_mentions_rel'],  # Default to PP mentions
        mode='lines',
        name=channel,
        line=dict(color=channel_colors[channel],dash=channel_line_styles[channel])
    ))

# Create the dropdown to switch between parties/politicians
dropdown_buttons = [
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['pp_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'PP',
        'method': 'restyle'
    },
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['psoe_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'PSOE',
        'method': 'restyle'
    },
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['up_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'UP',
        'method': 'restyle'
    },
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['vox_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'VOX',
        'method': 'restyle'
    }
]

# Update layout with dropdown and proper labeling
fig.update_layout(
    title="Weekly Average Relative Mentions Per Channel",
    xaxis_title="Week",
    yaxis_title="Av. Rel. Mentions x100",
    updatemenus=[{
        "buttons": dropdown_buttons,
        "direction": "down",
        "showactive": True
    }],
    template="plotly_dark"
)

# Format the x-axis (date)
fig.update_xaxes(
    dtick="M1",  # Monthly ticks
    tickformat="%b %Y",  # Format the ticks as 'Month Year'
    ticklabelmode="period"
)

# Show the interactive plot
fig.show()

# Save the interactive plot as an HTML file
fig.write_html("../figures/interactive_mentions_plot.html")

#%% 




# Assuming df_exp is already loaded and processed

# Convert 'date' column to datetime
df_exp['date'] = pd.to_datetime(df_exp['date'])

# Weekly aggregation
df_exp['week'] = df_exp['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Calculate relative mentions
df_exp['feijoo_mentions_rel'] = df_exp['feijoo_matches'] / df_exp['total_words']
df_exp['sanchez_mentions_rel'] = df_exp['sanchez_matches'] / df_exp['total_words']
df_exp['abascal_mentions_rel'] = df_exp['abascal_matches'] / df_exp['total_words']
df_exp['diaz_mentions_rel'] = df_exp['diaz_matches'] / df_exp['total_words']

df_exp['feijoo_mentions_rel']= df_exp['feijoo_mentions_rel']*100
df_exp['sanchez_mentions_rel']= df_exp['sanchez_mentions_rel']*100
df_exp['abascal_mentions_rel']= df_exp['abascal_mentions_rel']*100
df_exp['diaz_mentions_rel']= df_exp['diaz_mentions_rel']*100

# Group by week and channel, then calculate the average relative mentions
df_avg_mentions_weekly = df_exp.groupby(['week', 'channel']).agg({
    'feijoo_mentions_rel': 'mean',
    'sanchez_mentions_rel': 'mean',
    'abascal_mentions_rel': 'mean',
    'diaz_mentions_rel': 'mean'
}).reset_index()

# Channel color mapping
channel_colors = {
    'tve': 'purple',
    'a3': 'orange',
    'la6': 'green',
    't5': 'blue',
    'cuatro': 'red'
}

channel_line_styles = {
    'tve': 'solid',       # Solid line for 'tve'
    'a3': 'dash',         # Dashed line for 'a3'
    'la6': 'dot',         # Dotted line for 'la6'
    't5': 'dashdot',      # Dash-dot line for 't5'
    'cuatro': 'dash'      # Dashed line for 'cuatro'
}


#%% 

# Create an empty figure
fig = go.Figure()

# Add traces (lines) for each channel, but initially only show for one party (e.g., pp_mentions_rel)
for channel in df_avg_mentions_weekly['channel'].unique():
    fig.add_trace(go.Scatter(
        x=df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['week'],
        y=df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['feijoo_mentions_rel'],  # Default to PP mentions
        mode='lines',
        name=channel,
        line=dict(color=channel_colors[channel],dash=channel_line_styles[channel])
    ))

# Create the dropdown to switch between parties/politicians
dropdown_buttons = [
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['feijoo_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'Feijóo',
        'method': 'restyle'
    },
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['sanchez_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'Pedro Sánchez',
        'method': 'restyle'
    },
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['diaz_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'Yolanda Díaz',
        'method': 'restyle'
    },
    {
        'args': [{'y': [df_avg_mentions_weekly[df_avg_mentions_weekly['channel'] == channel]['abascal_mentions_rel']
                        for channel in df_avg_mentions_weekly['channel'].unique()]}],
        'label': 'Abascal',
        'method': 'restyle'
    }
]

# Update layout with dropdown and proper labeling
fig.update_layout(
    title="Weekly Average Relative Mentions Per Channel",
    xaxis_title="Week",
    yaxis_title="Av. Rel. Mentions x100",
    updatemenus=[{
        "buttons": dropdown_buttons,
        "direction": "down",
        "showactive": True
    }],
    template="plotly_dark"
)

# Format the x-axis (date)
fig.update_xaxes(
    dtick="M1",  # Monthly ticks
    tickformat="%b %Y",  # Format the ticks as 'Month Year'
    ticklabelmode="period"
)

# Show the interactive plot
fig.show()

# Save the interactive plot as an HTML file
fig.write_html("../figures/interactive_mentions_plot_politicians.html")

#%% Now im gonna fit bert:



model = BERTopic(language='spanish',embedding_model='paraphrase-multilingual-MiniLM-L12-v2', calculate_probabilities=True)


topics,probs = model.fit_transform(df_exp['noticias'].to_list())
topics

#%% 


new_topics = model.reduce_outliers(df_exp['noticias'].to_list(), topics)
model.update_topics(df_exp['noticias'].to_list(), topics=new_topics)
#%% 


def get_word_count_by_document(text,topic,dict_topics):
    new_dict={}
    for i in dict(dict_topics[topic]).keys(): 
        new_dict[i]=text.count(i)
    return new_dict

def get_word_proportion_by_document(text,topic,dict_topics):
    new_dict={}
    for i in dict(dict_topics[topic]).keys(): 
        new_dict[i]=(text.count(i)/len(text))
    return new_dict



dict_topics=model.get_topics()


df_exp['topic_text']=new_topics
df_exp['probs_text']=[i for i in probs]
# df_exp['max_prob']=[max(i) for i in df_exp['probs_text']]
df_exp['count_words']=[get_word_count_by_document(i,j,dict_topics) for i,j in zip(df_exp['noticias'],df_exp['topic_text'])]
df_exp['tf_words']=[get_word_proportion_by_document(i,j,dict_topics) for i,j in zip(df_exp['noticias'],df_exp['topic_text'])]
#only get the first 5 words for each topic

df_exp['words_topic']=[[j[0] for j in dict_topics.get(i)] for i in df_exp['topic_text']]
df_exp['words_topic']=df_exp['words_topic'].apply(lambda x: x[:5])

#%% Entropy:

import numpy as np
def calculate_entropy(df,base2=False):
    # Only include non-zero relative times to avoid log(0)
    if not base2:
        filtered_df = df[df['rel_time'] > 0]
    # Calculate entropy
        entropy = -np.sum(filtered_df['rel_time'] * np.log(filtered_df['rel_time']))
    #use the log base 2
    if base2:
        entropy = -np.sum(df['rel_time'] * np.log2(df['rel_time']))
    return entropy

#%% 
df_exp['unique_channels']=df_exp.groupby('date')['channel'].transform('nunique')

df_exp['political']=df_exp.apply(lambda x: 1 if x['psoe_matches']>0 or x['pp_matches']>0 or x['up_matches']>0 or x['vox_matches']>0 or x['cs_matches']>0 else 0,axis=1)
df_exp['time']=1#transform words_topic to string
df_exp['words_topic']=[str(i) for i in df_exp['words_topic']]

df_exp['political'].value_counts()


#%% 

df2=df_exp.query('unique_channels>2')

# df2=df2.query('topic_text!=30')
df_group=df2.groupby(['date','topic_text','words_topic'])[['time','political']].sum().reset_index()
# df_group=df2.groupby(['date','seccion'])['time'].sum().reset_index()
df_group['total_time']=df_group.groupby('date')['time'].transform('sum')
df_group['rel_time']=df_group['time']/df_group['total_time']
df_group['rel_pol']=df_group['political']/df_group['total_time']

df_entropy = df_group.groupby('date').apply(calculate_entropy).reset_index(name='entropy')


#%% 

import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd

df_group = df_exp.groupby(['date', 'channel', 'topic_text','words_topic'])[['time', 'political']].sum().reset_index()

# Step 2: Calculate total time spent on each date (across all topics and channels)
df_group['total_time'] = df_group.groupby('date')['time'].transform('sum')

# Step 3: Calculate relative time for each topic and channel (as a proportion of total time for that day)
df_group['rel_time'] = df_group['time'] / df_group['total_time']



# Assuming df_group and df_entropy are already created

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
    html.H1("Entropy Time Series"),
    dcc.Graph(id='entropy-graph'),
    html.Hr(),
    html.H2("Topic Composition for Selected Date"),
    dcc.Graph(id='topic-composition')
])

# Create the initial entropy time series plot
@app.callback(
    dash.Output('entropy-graph', 'figure'),
    [dash.Input('topic-composition', 'figure')]  # Just to keep it interactive, we don't need input here
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
    dash.Output('topic-composition', 'figure'),
    [dash.Input('entropy-graph', 'clickData')]  # Get the clicked point from entropy graph
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

    fig.add_trace(go.Scatter(
        x=channels,
        y=topics,
        mode='markers',
        marker=dict(
            size=[r * 100 for r in rel_times],  # Bubble size represents relative time
            color=rel_times,  # Color based on relative time
            showscale=True
        ),
        name='Topic Composition'
    ))

    fig.update_layout(
        title=f"Topic Composition for {clicked_date.date()}",
        xaxis_title="Channel",
        yaxis_title="Topics",
        template="plotly_dark"
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

#%% 


import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd

# Assuming df_group and df_entropy are already created

# Initialize Dash app
app = dash.Dash(__name__)

# Function to get topic composition for a specific date
def get_topic_composition(date, df):
    df_filtered = df[df['date'] == date]
    topics = df_filtered['words_topic'].tolist()
    rel_times = df_filtered['rel_time'].tolist()
    channels = df_filtered['channel'].tolist()
    return topics, rel_times, channels

# Color mapping for channels
channel_colors = {
    'a3': 'orange',
    't5': 'blue',
    'tve': 'purple',
    'la6': 'green',
    'cuatro': 'red'
}

# Layout of the app
app.layout = html.Div([
    html.H1("Entropy Time Series"),
    dcc.Graph(id='entropy-graph'),
    html.Hr(),
    html.H2("Topic Composition for Selected Date"),
    dcc.Graph(id='topic-composition')
])

# Create the initial entropy time series plot
@app.callback(
    dash.Output('entropy-graph', 'figure'),
    [dash.Input('topic-composition', 'figure')]  # Just to keep it interactive, we don't need input here
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
    dash.Output('topic-composition', 'figure'),
    [dash.Input('entropy-graph', 'clickData')]  # Get the clicked point from entropy graph
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
    app.run_server(debug=True)

