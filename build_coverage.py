import pandas as pd
import plotly.graph_objects as go
import json

df = pd.read_csv("/sessions/pensive-sweet-darwin/mnt/media_monitor/webpage/topic_composition.csv")
channels = ['a3','la6','tve','t5']
chan_names = {'a3':'Antena 3','la6':'laSexta','tve':'TVE','t5':'Telecinco','cuatro':'Cuatro'}
chan_colors = {'a3':'#F4A300','la6':'#2E9E5B','tve':'#8E44AD','t5':'#1F77B4','cuatro':'#D62728'}

# Cuatro joined the dataset much later (2024-01-29) than the other 4 channels
# (which start 2022-12-03/04), so it is excluded from this chart entirely.
# Restrict to the "balanced panel": dates where all 4 remaining channels
# (Antena 3, laSexta, TVE, Telecinco) have data, so every channel is
# compared over exactly the same period.
present = df.groupby('date')['channel'].apply(set)
balanced_dates = present[present.apply(lambda s: set(channels).issubset(s))].index
df = df[df['date'].isin(balanced_dates) & df['channel'].isin(channels)].copy()

# Metric: % of each channel's POLITICAL segments (not all segments) devoted to topic X.
# 'political' is a per-(date,channel,topic) count of segments that mention a party
# (PSOE/PP/UP/VOX/Cs) -- a genuine minute-by-minute dictionary flag, summed per cell.
# Using it as both numerator and denominator avoids the confound where a channel that
# simply talks about politics more overall (e.g. Antena 3, ~30% political segments vs
# ~14-21% for the others) would mechanically "lead" on every political topic.
chan_pol = df.groupby(['date','channel'])['political'].sum().reset_index().rename(columns={'political':'chan_pol_time'})
df = df.merge(chan_pol, on=['date','channel'])
df['rel_pol'] = df['political'] / df['chan_pol_time'].replace(0, pd.NA)

g = df.groupby(['topic_text','channel'])['rel_pol'].mean().reset_index()
piv = g.pivot(index='topic_text', columns='channel', values='rel_pol').fillna(0).infer_objects(copy=False)*100

topics_es = {
 15: "Partidos políticos\n(PP/PSOE/Vox)",
 117:"Encuestas y\nsondeos electorales",
 76: "Fiscal General\n(malversación)",
 66: "Ley de amnistía\n(Puigdemont)",
 7:  "Cataluña e\nindependentismo",
 32: "Yolanda Díaz,\nSumar y Podemos",
 97: "Ayuso y la Comunidad\nde Madrid",
 19: "Feijóo frente al\nGobierno de España",
 26: "Caso Begoña Gómez",
 11: "Sánchez vs. Feijóo\n(liderazgo)",
 57: "Pacto PSOE-Bildu\n(investidura)",
}
topics_en = {
 15: "Political parties\n(PP/PSOE/Vox)",
 117:"Election polls",
 76: "Attorney General\n(corruption case)",
 66: "Amnesty law\n(Puigdemont)",
 7:  "Catalonia &\nindependence",
 32: "Yolanda Díaz,\nSumar & Podemos",
 97: "Ayuso & the\nMadrid region",
 19: "Feijóo vs. Spain's\ngovernment",
 26: "Begoña Gómez case",
 11: "Sánchez vs. Feijóo\n(party leaders)",
 57: "PSOE-Bildu pact\n(government deal)",
}

out = piv.loc[list(topics_es.keys()), channels].round(2)
out['spread'] = out[channels].max(axis=1) - out[channels].min(axis=1)
out = out.sort_values('spread', ascending=True)
order = out.index.tolist()


def build_fig(lang):
    topics = topics_es if lang == 'es' else topics_en
    title = "¿En qué temas se nota más el desacuerdo entre cadenas?" if lang == 'es' else "Where do Spanish TV channels disagree most?"
    xaxis_title = "% del tiempo político dedicado al tema (media del periodo)" if lang == 'es' else "% of a channel's political airtime devoted to topic (period average)"
    hover_unit = "de su tiempo político" if lang == 'es' else "of its political airtime"
    annot_text = ("Telecinco lidera 10 de los 11 temas de este<br>gráfico. Aquí dedica un 28% de su tiempo<br>político a la confrontación Sánchez-Feijóo,<br>frente a un 17-20% en el resto de cadenas"
                   if lang == 'es' else
                   "Telecinco leads 10 of the 11 topics shown<br>here. It devotes 28% of its political airtime<br>to the Sánchez-Feijóo clash, vs. 17-20%<br>on the other channels")

    labels = [topics[t] for t in order]

    fig = go.Figure()
    for t, label in zip(order, labels):
        vals = out.loc[t, channels].astype(float)
        fig.add_trace(go.Scatter(
            x=[vals.min(), vals.max()], y=[label, label],
            mode='lines', line=dict(color='#d8dde3', width=6),
            showlegend=False, hoverinfo='skip'
        ))
    for ch in channels:
        fig.add_trace(go.Scatter(
            x=out.loc[order, ch], y=labels, mode='markers',
            name=chan_names[ch],
            marker=dict(size=14, color=chan_colors[ch], line=dict(width=1, color='white')),
            hovertemplate=f"<b>{chan_names[ch]}</b><br>%{{y}}<br>%{{x:.2f}}% {hover_unit}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1d2d44', family='Inter, Helvetica Neue, Arial, sans-serif'), x=0.02, xanchor='left'),
        xaxis=dict(title=xaxis_title, ticksuffix='%', gridcolor='#eef1f4', zeroline=False),
        yaxis=dict(title=None, automargin=True),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter, Helvetica Neue, Arial, sans-serif', color='#2c3e50'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, title=None),
        margin=dict(l=190, r=40, t=110, b=60),
        height=620,
    )

    label_11 = topics[11]
    fig.add_annotation(
        x=out.loc[11, channels].astype(float).max(),
        y=label_11,
        text=annot_text,
        showarrow=True, arrowhead=2, ax=60, ay=-35,
        font=dict(size=11, color='#1d2d44'),
        bgcolor='#fdf6e3', bordercolor='#c8a356', borderwidth=1, borderpad=4,
        align='left'
    )
    return fig

fig_es = build_fig('es')
fig_en = build_fig('en')

PJE = __import__('plotly').utils.PlotlyJSONEncoder
chart_es_json = json.dumps(fig_es.to_plotly_json(), cls=PJE)
chart_en_json = json.dumps(fig_en.to_plotly_json(), cls=PJE)

NAV = """  <nav>
    <a href="index.html" data-es="Inicio" data-en="Home">Inicio</a>
    <a href="coverage.html" class="{cov}" data-es="Acuerdo y Desacuerdo" data-en="Media Agreement">Acuerdo y Desacuerdo</a>
    <a href="mentions.html" data-es="Menciones Políticas" data-en="Political Mentions">Menciones Políticas</a>
    <a href="politicaltone.html" data-es="Tono Político" data-en="Political Tone">Tono Político</a>
    <a href="methodology.html" data-es="Metodología" data-en="Methodology">Metodología</a>
    <a href="about.html" data-es="Sobre Nosotros" data-en="About Us">Sobre Nosotros</a>
  </nav>"""

FOOTER = """  <footer>
    <p>
      <img src="logo.png" alt="SMM Logo Small">
      &copy; 2025 <span>SMM</span> — Spanish Media Monitor.
      <span data-es="Datos bajo licencia" data-en="Data licensed under">Datos bajo licencia</span>
      <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="license">CC BY 4.0</a>
    </p>
  </footer>"""

page = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Acuerdo y Desacuerdo entre Medios – SMM</title>
  <link rel="stylesheet" href="style.css">
<link rel="icon" sizes="32x32" href="logo.png" type="image/png">
<link rel="apple-touch-icon" href="logo.png">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
</head>
<body>

  <!-- HEADER -->
  <header>
    <button id="lang-toggle">English</button>
    <img src="main_logo.png" alt="LOGO_Banner" class="banner">
  </header>

  <!-- NAVIGATION -->
{NAV.format(cov="active")}

  <!-- MAIN CONTENT -->
  <div class="content">
    <h2 data-es="Acuerdo y Desacuerdo entre Medios" data-en="Media Agreement &amp; Disagreement">Acuerdo y Desacuerdo entre Medios</h2>
    <p data-es="Para cada tema de actualidad, calculamos qué porcentaje del tiempo político de cada cadena (es decir, de los segmentos que mencionan a partidos o políticos) se dedica a ese tema. En el gráfico, cada punto representa una cadena y la línea gris muestra el rango entre la cadena que más y la que menos peso le dio al mismo tema dentro de su propia agenda política. Cuanto más larga la línea, mayor es el desacuerdo sobre la importancia de esa noticia dentro de la cobertura política; cuanto más corta, mayor el acuerdo entre cadenas."
       data-en="For each news topic, we calculate what share of each channel's political airtime (i.e. segments that mention parties or politicians) goes to that topic. In the chart, each dot represents a channel and the grey line shows the range between the channel that gives the topic the most and the least weight within its own political agenda. The longer the line, the bigger the disagreement about how newsworthy that story is within political coverage; the shorter it is, the more channels agree.">
      Para cada tema de actualidad, calculamos qué porcentaje del tiempo político de cada cadena (es decir, de los segmentos que mencionan a partidos o políticos) se dedica a ese tema. En el gráfico, cada punto representa una cadena y la línea gris muestra el rango entre la cadena que más y la que menos peso le dio al mismo tema dentro de su propia agenda política. Cuanto más larga la línea, mayor es el desacuerdo sobre la importancia de esa noticia dentro de la cobertura política; cuanto más corta, mayor el acuerdo entre cadenas.
    </p>

    <div class="chart-container">
      <div id="disagreement-chart"></div>
    </div>

    <p class="legend-note" style="color: var(--muted); font-size: 0.9rem; margin-top: 1rem;"
       data-es="Datos: SMM, dic. 2022 – sep. 2024. Incluye Antena 3, laSexta, TVE y Telecinco —las 4 cadenas con cobertura comparable durante todo este periodo—; Cuatro se incorporó más tarde (2024) y queda fuera de este gráfico para mantener la muestra equilibrada. Selección de 11 temas recurrentes donde más difieren las cadenas. La versión completa permitirá filtrar por fecha, ordenar por tema o cadena, y explorar todos los temas detectados automáticamente. Más detalles en la sección de <a href='methodology.html'>Metodología</a>."
       data-en="Data: SMM, Dec 2022 - Sep 2024. Includes Antena 3, laSexta, TVE and Telecinco — the 4 channels with comparable coverage over this whole period; Cuatro joined later (2024) and is excluded from this chart to keep the sample balanced. Selection of 11 recurring topics where channels diverge most. The full version will allow filtering by date range, sorting by topic or channel, and exploring all automatically detected topics. More details in the <a href='methodology.html'>Methodology</a> section.">
      Datos: SMM, dic. 2022 – sep. 2024. Incluye Antena 3, laSexta, TVE y Telecinco —las 4 cadenas con cobertura comparable durante todo este periodo—; Cuatro se incorporó más tarde (2024) y queda fuera de este gráfico para mantener la muestra equilibrada. Selección de 11 temas recurrentes donde más difieren las cadenas. La versión completa permitirá filtrar por fecha, ordenar por tema o cadena, y explorar todos los temas detectados automáticamente. Más detalles en la sección de <a href="methodology.html">Metodología</a>.
    </p>
  </div>

{FOOTER}

<script>
  var disagreementCharts = {{
    es: {chart_es_json},
    en: {chart_en_json}
  }};

  Plotly.newPlot('disagreement-chart', disagreementCharts.es.data, disagreementCharts.es.layout, {{responsive: true}});

  window.applyChartLang = function(lang) {{
    var c = disagreementCharts[lang] || disagreementCharts.es;
    Plotly.react('disagreement-chart', c.data, c.layout);
  }};
</script>
<script src="lang.js"></script>

</body>
</html>
"""

with open("/sessions/pensive-sweet-darwin/mnt/media_monitor/webpage/coverage.html", "w") as f:
    f.write(page)

print("done")
print(out)
