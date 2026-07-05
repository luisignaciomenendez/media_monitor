from __future__ import annotations

import ast
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "topic_composition.csv").exists() and (BASE_DIR / "entropy.csv").exists():
    DATA_DIR = BASE_DIR
else:
    DATA_DIR = BASE_DIR / "webpage" / "data"

CHANNEL_ORDER = ["tve", "a3", "la6", "t5", "cuatro"]
CHANNEL_LABELS = {
    "tve": "TVE",
    "a3": "Antena 3",
    "la6": "laSexta",
    "t5": "Telecinco",
    "cuatro": "Cuatro",
}
CHANNEL_COLORS = {
    "tve": "#5b3cc4",
    "a3": "#f18f01",
    "la6": "#139a43",
    "t5": "#2274a5",
    "cuatro": "#d1495b",
}

TOPIC_RULES = [
    ("Guerra de Ucrania", {"ucrania", "rusia", "putin", "ruso", "guerra"}),
    ("Israel y Gaza", {"israel", "gaza", "hamás", "franja", "israelí"}),
    ("Política nacional", {"sánchez", "feijóo", "psoe", "pp", "vox", "gobierno", "presidente"}),
    ("Cataluña y amnistía", {"cataluña", "esquerra", "illa", "amnistía", "puigdemont", "junts"}),
    ("Migración", {"migrantes", "inmigrantes", "migratoria", "canarias", "menores"}),
    ("Vivienda", {"vivienda", "viviendas", "alquiler", "piso", "precios"}),
    ("Empleo", {"empleo", "trabajo", "laboral", "trabajadores", "paro"}),
    ("Ola de calor", {"calor", "temperaturas", "temperatura", "grados", "ola"}),
    ("Lluvias y tormentas", {"lluvias", "agua", "litros", "tormentas", "precipitaciones"}),
    ("Sequía y embalses", {"sequía", "embalses", "restricciones", "litros", "agua"}),
    ("Incendios", {"incendio", "fuego", "llamas", "bomberos", "edificio"}),
    ("Violencia machista", {"mujer", "pareja", "hombre", "violencia", "crimen"}),
    ("Agresión sexual", {"menores", "sexual", "agresión", "menor", "víctima"}),
    ("Turismo", {"turismo", "turistas", "ocupación", "vacaciones", "hoteles"}),
    ("Fútbol", {"barça", "liga", "real", "gol", "equipo"}),
    ("Cine y cultura", {"película", "cine", "películas", "festival", "historia"}),
    ("Arte y museos", {"arte", "museo", "obra", "obras", "picasso"}),
    ("Música", {"música", "canciones", "disco", "canción", "banda"}),
    ("Gastronomía", {"plato", "sabor", "cocina", "jamón", "carne"}),
]


def entropy(shares):
    shares = np.array(shares, dtype=float)
    shares = shares[shares > 0]
    return -np.sum(shares * np.log(shares))


def safe_number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


def parse_keywords(raw: object) -> list[str]:
    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        parsed = []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip().lower() for item in parsed if str(item).strip()]


def topic_label(keywords: list[str]) -> str:
    keyword_set = set(keywords)
    best_label = None
    best_score = 0
    for label, rule_words in TOPIC_RULES:
        score = len(keyword_set & rule_words)
        if score > best_score:
            best_score = score
            best_label = label
    if best_label and best_score >= 2:
        return best_label
    if not keywords:
        return "Historia sin etiqueta"
    return " / ".join(word.replace("_", " ").title() for word in keywords[:3])


def load_original_entropy() -> pd.DataFrame:
    """Replicate original entropy values and normalize them to 0-100."""
    entropy_df = pd.read_csv(DATA_DIR / "entropy.csv")
    entropy_df = entropy_df.loc[:, ~entropy_df.columns.duplicated()].copy()
    entropy_df["date"] = pd.to_datetime(entropy_df["date"]).dt.strftime("%Y-%m-%d")
    entropy_df["entropy"] = pd.to_numeric(entropy_df["entropy"], errors="coerce")
    entropy_df["entropy"] = entropy_df["entropy"].ffill().bfill().fillna(0.0)
    entropy_df = entropy_df.sort_values("date").copy()

    return entropy_df


def load_topics() -> pd.DataFrame:
    topics = pd.read_csv(DATA_DIR / "topic_composition.csv")
    topics = topics.loc[:, ~topics.columns.duplicated()].copy()
    topics["date"] = pd.to_datetime(topics["date"]).dt.strftime("%Y-%m-%d")
    topics = topics[topics["channel"].isin(CHANNEL_ORDER)].copy()
    topics["time"] = pd.to_numeric(topics["time"], errors="coerce").fillna(0.0)
    topics["rel_time"] = pd.to_numeric(topics["rel_time"], errors="coerce").fillna(0.0)
    topics["keywords"] = topics["words_topic"].map(parse_keywords)
    topics["label"] = topics["keywords"].map(topic_label)
    topics["keyword_hint"] = topics["keywords"].map(lambda values: safe_text(", ".join(values[:5]), "sin palabras clave"))
    return topics


def add_normalized_entropy(entropy_df: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily entropy by log(number of stories/topics that day)."""
    n_stories = (
        topics.groupby("date")["topic_text"]
        .nunique()
        .reset_index(name="n_stories")
    )
    entropy_df = entropy_df.merge(n_stories, on="date", how="left")
    entropy_df["n_stories"] = entropy_df["n_stories"].fillna(0)
    entropy_df["entropy_0_100"] = np.where(
        entropy_df["n_stories"] > 1,
        100 * entropy_df["entropy"] / np.log(entropy_df["n_stories"]),
        0,
    )
    entropy_df["entropy_smooth"] = (
        entropy_df["entropy_0_100"].rolling(window=7, center=True, min_periods=1).mean()
    )
    return entropy_df


def compute_jsd_series(topics: pd.DataFrame, valid_dates: set[str]) -> pd.DataFrame:
    """Daily JSD across whichever SMM channels are present that day.

    This keeps the date coverage aligned with the original explorer instead of
    dropping early dates just because Cuatro is missing.
    """
    g = (
        topics[topics["date"].isin(valid_dates)]
        .groupby(["date", "channel", "label"], as_index=False)["time"]
        .sum()
        .rename(columns={"time": "airtime"})
    )
    totals = g.groupby(["date", "channel"])["airtime"].transform("sum")
    g["share"] = np.where(totals > 0, g["airtime"] / totals, 0)

    rows = []
    for date, frame in g.groupby("date"):
        present_channels = [
            channel
            for channel in CHANNEL_ORDER
            if channel in set(frame["channel"]) and frame.loc[frame["channel"] == channel, "share"].sum() > 0
        ]
        if len(present_channels) < 2:
            continue
        pivot = (
            frame.pivot_table(index="label", columns="channel", values="share", aggfunc="sum", fill_value=0)
            .reindex(columns=present_channels, fill_value=0)
            .fillna(0)
        )
        matrix = pivot.to_numpy(dtype=float)
        mean_distribution = matrix.mean(axis=1)
        jsd = entropy(mean_distribution) - np.mean([entropy(matrix[:, i]) for i in range(matrix.shape[1])])
        jsd_0_100 = 100 * jsd / np.log(len(present_channels))
        rows.append(
            {
                "date": date,
                "jsd": jsd,
                "jsd_0_100": jsd_0_100,
                "jsd_smooth": jsd_0_100,
                "channels_present": len(present_channels),
            }
        )

    jsd_df = pd.DataFrame(rows).sort_values("date")
    jsd_df["jsd_smooth"] = jsd_df["jsd_0_100"].rolling(window=7, center=True, min_periods=1).mean()
    return jsd_df


def build_payload(topics: pd.DataFrame, entropy_df: pd.DataFrame) -> dict[str, dict]:
    merged = topics.merge(
        entropy_df[["date", "entropy", "entropy_0_100", "entropy_smooth"]],
        on="date",
        how="left",
    )

    payload: dict[str, dict] = {}
    for date, frame in merged.groupby("date"):
        matrix = (
            frame.groupby(["label", "channel"], as_index=False)
            .agg(rel_time=("rel_time", "sum"), raw_time=("time", "sum"), keyword_hint=("keyword_hint", "first"))
        )
        if matrix.empty:
            continue

        pivot = (
            matrix.pivot_table(index="label", columns="channel", values="rel_time", aggfunc="sum", fill_value=0.0)
            .reindex(columns=CHANNEL_ORDER, fill_value=0.0)
            .fillna(0.0)
        )
        pivot["mean_share"] = pivot[CHANNEL_ORDER].mean(axis=1)
        pivot["spread"] = pivot[CHANNEL_ORDER].max(axis=1) - pivot[CHANNEL_ORDER].min(axis=1)
        pivot["coverage"] = (pivot[CHANNEL_ORDER] > 0).sum(axis=1)

        ranked_shared = pivot.sort_values(["coverage", "mean_share", "spread"], ascending=[False, False, True]).head(8)
        ranked_divergent = pivot[pivot[CHANNEL_ORDER].max(axis=1) > 0].sort_values(
            ["spread", "mean_share"], ascending=[False, False]
        ).head(8)

        def rows_for(ranked: pd.DataFrame) -> tuple[list[str], list[list[float]], list[dict]]:
            labels = ranked.index.tolist()
            heatmap_values = []
            rows = []
            for label in labels:
                shares = [safe_number(ranked.loc[label, channel], 0.0) for channel in CHANNEL_ORDER]
                keyword_values = matrix.loc[matrix["label"] == label, "keyword_hint"]
                keyword_hint = safe_text(keyword_values.iloc[0] if not keyword_values.empty else "", "sin palabras clave")
                heatmap_values.append([round(value * 100, 2) for value in shares])
                rows.append(
                    {
                        "label": label,
                        "spread": round(safe_number(ranked.loc[label, "spread"], 0.0) * 100, 2),
                        "mean_share": round(safe_number(ranked.loc[label, "mean_share"], 0.0) * 100, 2),
                        "channels_present": int(safe_number(ranked.loc[label, "coverage"], 0.0)),
                        "keywords": keyword_hint,
                    }
                )
            return labels, heatmap_values, rows

        shared_labels, shared_heatmap, shared_rows = rows_for(ranked_shared)
        divergent_labels, divergent_heatmap, divergent_rows = rows_for(ranked_divergent)

        per_channel = frame.groupby(["channel", "label"], as_index=False).agg(rel_time=("rel_time", "sum"))
        channel_mix = []
        for channel in CHANNEL_ORDER:
            channel_frame = per_channel[per_channel["channel"] == channel].sort_values("rel_time", ascending=False)
            items = [
                {"label": safe_text(row["label"], "Historia sin etiqueta"), "share": round(safe_number(row["rel_time"], 0.0) * 100, 2)}
                for _, row in channel_frame.head(5).iterrows()
            ]
            channel_mix.append({"channel": CHANNEL_LABELS[channel], "color": CHANNEL_COLORS[channel], "items": items})

        dominant = frame.groupby("label", as_index=False)["rel_time"].sum().sort_values("rel_time", ascending=False).head(1)
        best_shared = shared_rows[0] if shared_rows else None
        best_divergent = divergent_rows[0] if divergent_rows else None

        payload[date] = {
            "date": date,
            "entropy_0_100": round(safe_number(frame["entropy_0_100"].iloc[0], 0.0), 2),
            "entropy_value": round(safe_number(frame["entropy"].iloc[0], 0.0), 3),
            "dominant_topic": safe_text(dominant.iloc[0]["label"] if not dominant.empty else "", "Sin historia dominante"),
            "top_shared_topic": safe_text(best_shared["label"] if best_shared else "", "Sin historia compartida clara"),
            "top_shared_keywords": safe_text(best_shared["keywords"] if best_shared else "", "sin palabras clave"),
            "top_shared_spread": safe_number(best_shared["spread"] if best_shared else 0, 0.0),
            "top_divergent_topic": safe_text(best_divergent["label"] if best_divergent else "", "Sin historia divergente clara"),
            "top_divergent_keywords": safe_text(best_divergent["keywords"] if best_divergent else "", "sin palabras clave"),
            "top_divergent_spread": safe_number(best_divergent["spread"] if best_divergent else 0, 0.0),
            "shared": {"labels": shared_labels, "heatmap_values": shared_heatmap, "rows": shared_rows},
            "divergent": {"labels": divergent_labels, "heatmap_values": divergent_heatmap, "rows": divergent_rows},
            "channel_mix": channel_mix,
        }

    return payload


def print_startup_summary() -> None:
    print("Rows:", len(topics_df))
    print("Entropy dates:", len(entropy_df))
    print("Payload dates:", len(payload))
    print("JSD dates:", len(jsd_df))
    print("Channels:", [CHANNEL_LABELS[channel] for channel in CHANNEL_ORDER])
    print("Topic labels:", topics_df["label"].nunique())
    print("\nNormalized entropy summary:")
    print(entropy_df["entropy_0_100"].describe().to_string())
    print("\nJSD summary:")
    print(jsd_df["jsd_0_100"].describe().to_string())
    print("\nTop 5 most divergent JSD days:")
    print(jsd_df.sort_values("jsd_0_100", ascending=False).head(5).to_string(index=False))
    print("\nTop 5 most similar JSD days:")
    print(jsd_df.sort_values("jsd_0_100", ascending=True).head(5).to_string(index=False))


def series_for_metric(metric: str) -> pd.DataFrame:
    if metric == "jsd":
        return jsd_df.rename(columns={"jsd_0_100": "value", "jsd_smooth": "smooth"})[["date", "value", "smooth"]]
    return entropy_df.rename(columns={"entropy_0_100": "value", "entropy_smooth": "smooth"})[["date", "value", "smooth"]]


def default_date_for_metric(metric: str) -> str:
    series = series_for_metric(metric)
    return str(series.loc[series["value"].idxmax(), "date"])


def metric_copy(metric: str) -> dict[str, str]:
    if metric == "jsd":
        return {
            "timeline_title": "Serie temporal de desacuerdo entre canales",
            "daily_name": "Desacuerdo diario",
            "smooth_name": "Media móvil 7 días",
            "selected_name": "Fecha seleccionada",
            "y_title": "Desacuerdo entre canales",
            "hover": "Desacuerdo entre canales",
            "card_label": "Desacuerdo entre canales",
            "card_help": "Escala 0-100. Más alto = agendas más separadas entre cadenas.",
            "topic_label": "Historia que más divide",
            "topic_help": "Mayor diferencia entre cadenas.",
            "lower_title": "Historias que impulsan el desacuerdo",
            "lower_subtitle": "Ranking de diferencia por historia en la fecha elegida",
            "heatmap_title": "Historias que más separan a las cadenas",
        }
    return {
        "timeline_title": "Serie temporal de entropía de la agenda",
        "daily_name": "Entropía diaria",
        "smooth_name": "Media móvil 7 días",
        "selected_name": "Fecha seleccionada",
        "y_title": "Entropía de la agenda",
        "hover": "Entropía",
        "card_label": "Entropía",
        "card_help": "Escala 0-100. Más alto = agenda repartida entre más historias.",
        "topic_label": "Historia principal",
        "topic_help": "Historia con más peso agregado.",
        "lower_title": "Historias principales",
        "lower_subtitle": "Mapa de calor por historia y canal en la fecha elegida",
        "heatmap_title": "Historias principales",
    }


def valid_selected_date(metric: str, selected_date: str | None) -> str:
    dates = set(series_for_metric(metric)["date"])
    if selected_date in dates:
        return str(selected_date)
    return default_date_for_metric(metric)


def make_timeline(metric: str, selected_date: str) -> go.Figure:
    copy = metric_copy(metric)
    series = series_for_metric(metric)
    selected_date = valid_selected_date(metric, selected_date)
    selected = series.loc[series["date"] == selected_date]
    selected_value = safe_number(selected["value"].iloc[0] if not selected.empty else 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["value"],
            mode="lines",
            name=copy["daily_name"],
            line=dict(color="#c8a356", width=2),
            hovertemplate="%{x}<br>" + copy["hover"] + ": <b>%{y:.1f}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["smooth"],
            mode="lines",
            name=copy["smooth_name"],
            line=dict(color="#1d2d44", width=3.2),
            hovertemplate="%{x}<br>Media móvil: <b>%{y:.1f}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_date],
            y=[selected_value],
            mode="markers",
            name=copy["selected_name"],
            marker=dict(size=10, color="#1d2d44"),
            hovertemplate="%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=60, r=20, t=10, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.12),
        xaxis=dict(gridcolor="#eef1f4", zeroline=False, showspikes=True, spikecolor="#c8a356", spikethickness=1),
        yaxis=dict(title=copy["y_title"], range=[0, 100], gridcolor="#eef1f4", zeroline=False),
        height=560,
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#2c3e50"),
    )
    return fig


def make_spread_bars(day: dict, metric: str) -> go.Figure:
    mode = "divergent" if metric == "jsd" else "shared"
    rows = list(day[mode]["rows"])
    if metric != "jsd":
        rows = sorted(
            rows,
            key=lambda row: (row.get("channels_present", 0), row.get("mean_share", 0), -row.get("spread", 0)),
            reverse=True,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[row.get("spread", 0) for row in rows],
            y=[row.get("label", "") for row in rows],
            orientation="h",
            marker=dict(color=["#1d2d44" if idx == 0 else "#d8dde3" for idx, _ in enumerate(rows)]),
            customdata=[
                [row.get("keywords", "sin palabras clave"), row.get("channels_present", 0), row.get("mean_share", 0)]
                for row in rows
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Diferencia entre cadenas: %{x:.1f} puntos<br>"
                "Peso medio: %{customdata[2]:.1f} puntos<br>"
                "Cadenas presentes: %{customdata[1]}<br>"
                "Palabras clave: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=145, r=15, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Diferencia entre cadenas (puntos porcentuales)", gridcolor="#eef1f4", zeroline=False),
        yaxis=dict(autorange="reversed"),
        height=420,
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#2c3e50"),
        showlegend=False,
    )
    return fig


def make_heatmap(day: dict, metric: str) -> go.Figure:
    mode = "divergent" if metric == "jsd" else "shared"
    values = day[mode]["heatmap_values"] or [[0, 0, 0, 0, 0]]
    labels = day[mode]["labels"] or ["Sin historias"]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=values,
            x=[CHANNEL_LABELS[channel] for channel in CHANNEL_ORDER],
            y=labels,
            colorscale=[[0, "#f9fafc"], [0.35, "#dfe7f0"], [0.65, "#c8a356"], [1, "#1d2d44"]],
            zmin=0,
            zmax=max(20, max(max(row) for row in values)),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}% del tiempo de esa cadena<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=170, r=20, t=8, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#2c3e50"),
    )
    return fig


def make_cards(day: dict, metric: str) -> list[html.Div]:
    copy = metric_copy(metric)
    if metric == "jsd":
        series = jsd_df[jsd_df["date"] == day["date"]]
        value = safe_number(series["jsd_0_100"].iloc[0] if not series.empty else 0.0)
        topic = day["top_divergent_topic"]
        detail = f"Diferencia entre cadenas: {day['top_divergent_spread']:.1f} puntos. Palabras clave: {day['top_divergent_keywords']}"
    else:
        value = day["entropy_0_100"]
        topic = day["top_shared_topic"]
        detail = f"Diferencia entre cadenas: {day['top_shared_spread']:.1f} puntos. Palabras clave: {day['top_shared_keywords']}"

    return [
        html.Div(
            [html.Div(copy["card_label"], className="k"), html.Div(f"{value:.1f}", className="v"), html.Div(copy["card_help"], className="s")],
            className="card",
        ),
        html.Div(
            [html.Div(copy["topic_label"], className="k"), html.Div(topic, className="v"), html.Div(detail, className="s")],
            className="card",
        ),
        html.Div(
            [
                html.Div("Historia más visible", className="k"),
                html.Div(day["dominant_topic"], className="v"),
                html.Div("Mayor peso agregado entre todas las cadenas.", className="s"),
            ],
            className="card",
        ),
    ]


def make_topic_list(day: dict, metric: str) -> list[html.Div]:
    mode = "divergent" if metric == "jsd" else "shared"
    return [
        html.Div(
            [
                html.Div(
                    [html.Strong(row["label"]), html.Span(f"{row['spread']:.1f} puntos de diferencia")],
                    className="row-top",
                ),
                html.Div(f"Palabras clave: {row['keywords']}", className="meta"),
                html.Div(f"Presente en {row['channels_present']} cadenas. Peso medio: {row['mean_share']:.1f}%.", className="meta"),
            ],
            className="row",
        )
        for row in day[mode]["rows"]
    ]


def make_channel_mix(day: dict) -> list[html.Div]:
    blocks = []
    for block in day["channel_mix"]:
        chips = [
            html.Span([html.Strong(f"{item['share']:.1f}%"), f" {item['label']}"], className="topic-chip")
            for item in block["items"]
        ]
        if not chips:
            chips = [html.Span("Sin historias", className="topic-chip")]
        blocks.append(
            html.Div(
                [html.H3(block["channel"]), *chips],
                className="channel-block",
                style={"background": f"linear-gradient(135deg, {block['color']}, #1d2d44)"},
            )
        )
    return blocks


topics_df = load_topics()
entropy_df = add_normalized_entropy(load_original_entropy(), topics_df)
payload = build_payload(topics_df, entropy_df)
jsd_df = compute_jsd_series(topics_df, set(entropy_df["date"]))
print_startup_summary()

app = dash.Dash(__name__)
server = app.server

app.index_string = """
<!DOCTYPE html>
<html lang="es">
  <head>
    {%metas%}
    <title>Entropía y desacuerdo entre canales</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --navy: #1d2d44;
        --navy-light: #2c3e50;
        --gold: #c8a356;
        --light: #f9fafc;
        --text: #2c3e50;
        --muted: #6c757d;
        --white: #ffffff;
        --grid: #eef1f4;
      }
      * { box-sizing: border-box; }
      body {
        font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
        margin: 0;
        background-color: var(--light);
        color: var(--text);
        line-height: 1.7;
      }
      .wrap {
        max-width: 1100px;
        margin: 0 auto;
        padding: 60px 20px 72px;
      }
      .panel {
        background-color: var(--white);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
        border-top: 3px solid var(--gold);
        margin-bottom: 30px;
      }
      .panel-title {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 10px;
        margin-bottom: 14px;
      }
      .panel-title h2 {
        margin: 0;
        color: var(--navy);
        font-size: 1.15rem;
        font-weight: 600;
      }
      .panel-title span {
        color: var(--muted);
        font-size: 0.9rem;
      }
      .controls {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 14px;
        margin-bottom: 18px;
        flex-wrap: wrap;
      }
      .controls label {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 600;
      }
      .metric-radio label {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-right: 14px;
      }
      .metric-radio input {
        accent-color: var(--gold);
      }
      .date-select {
        min-width: 220px;
      }
      .date-select .Select-control {
        border: 1px solid #d8dde3;
        border-radius: 8px;
      }
      .cards {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 18px;
      }
      .card {
        background: #fbfcfe;
        border: 1px solid #e7ebf0;
        border-radius: 10px;
        padding: 14px;
      }
      .card .k {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
        font-weight: 700;
      }
      .card .v {
        font-size: 26px;
        line-height: 1.1;
        font-weight: 800;
        color: var(--navy);
        overflow-wrap: anywhere;
      }
      .card .s {
        margin-top: 8px;
        font-size: 13px;
        color: var(--muted);
        line-height: 1.45;
      }
      .two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
      }
      .list {
        display: grid;
        gap: 10px;
        margin-top: 14px;
      }
      .row {
        border: 1px solid #e7ebf0;
        border-radius: 10px;
        padding: 12px;
        background: #fbfcfe;
      }
      .row-top {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: baseline;
        margin-bottom: 6px;
      }
      .row strong {
        font-size: 15px;
      }
      .meta {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.4;
      }
      .channel-blocks {
        display: grid;
        gap: 10px;
      }
      .channel-block {
        border-radius: 10px;
        padding: 14px;
        color: white;
      }
      .channel-block h3 {
        margin: 0 0 10px;
        font-size: 16px;
      }
      .topic-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 0 8px 8px 0;
        padding: 7px 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.18);
        font-size: 13px;
      }
      .footer-note {
        margin-top: 18px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
      }
      @media (max-width: 900px) {
        .two-col {
          grid-template-columns: 1fr;
        }
        .cards {
          grid-template-columns: 1fr;
        }
        .controls {
          justify-content: flex-start;
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
"""

initial_metric = "entropy"
initial_date = default_date_for_metric(initial_metric)

app.layout = html.Div(
    [
        dcc.Store(id="selected-date-store", data=initial_date),
        html.Section(
            [
                html.Div(
                    [
                        html.H2(id="timeline-title"),
                        html.Span("Haz clic en un día para actualizar el análisis inferior"),
                    ],
                    className="panel-title",
                ),
                html.Div(
                    [
                        html.Label("Métrica"),
                        dcc.RadioItems(
                            id="metric-selector",
                            options=[
                                {"label": "Entropía", "value": "entropy"},
                                {"label": "Desacuerdo entre canales", "value": "jsd"},
                            ],
                            value=initial_metric,
                            className="metric-radio",
                        ),
                    ],
                    className="controls",
                ),
                dcc.Graph(id="timeline", config={"displayModeBar": False}),
            ],
            className="panel",
        ),
        html.Section(
            [
                html.Div([html.H2("Día seleccionado"), html.Span(id="selected-date-label")], className="panel-title"),
                html.Div(
                    [
                        html.Label("Ir a la fecha"),
                        dcc.Dropdown(id="date-select", clearable=False, className="date-select"),
                    ],
                    className="controls",
                ),
                html.Div(id="cards", className="cards"),
                dcc.Graph(id="spread-bars", config={"displayModeBar": False}),
            ],
            className="panel",
        ),
        html.Div(
            [
                html.Section(
                    [
                        html.Div([html.H2(id="heatmap-panel-title"), html.Span(id="heatmap-panel-subtitle")], className="panel-title"),
                        dcc.Graph(id="heatmap", config={"displayModeBar": False}),
                        html.Div(id="topic-list", className="list"),
                    ],
                    className="panel",
                ),
                html.Section(
                    [
                        html.Div(
                            [html.H2("Mezcla editorial por canal"), html.Span("Historias más visibles en la fecha elegida")],
                            className="panel-title",
                        ),
                        html.Div(id="channel-mix", className="channel-blocks"),
                        html.Div(
                            "Las etiquetas de las historias son heurísticas construidas a partir de `words_topic`. "
                            "Aquí se puede añadir luego una capa de relabeling con IA si quieres nombres todavía más limpios.",
                            className="footer-note",
                        ),
                    ],
                    className="panel",
                ),
            ],
            className="two-col",
        ),
    ],
    className="wrap",
)


@app.callback(
    [Output("date-select", "options"), Output("date-select", "value"), Output("selected-date-store", "data")],
    [Input("metric-selector", "value"), Input("timeline", "clickData"), Input("date-select", "value")],
)
def sync_selected_date(metric, click_data, dropdown_date):
    ctx = dash.callback_context
    dates = series_for_metric(metric)["date"].tolist()
    options = [{"label": date, "value": date} for date in dates if date in payload]

    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    if triggered == "timeline" and click_data and click_data.get("points"):
        selected = str(click_data["points"][0]["x"])
    elif triggered == "date-select" and dropdown_date:
        selected = dropdown_date
    else:
        selected = dropdown_date or default_date_for_metric(metric)

    valid_options = {option["value"] for option in options}
    if selected not in valid_options:
        selected = default_date_for_metric(metric)
        if selected not in valid_options and options:
            selected = options[-1]["value"]
    return options, selected, selected


@app.callback(
    [
        Output("timeline", "figure"),
        Output("timeline-title", "children"),
        Output("selected-date-label", "children"),
        Output("cards", "children"),
        Output("spread-bars", "figure"),
        Output("heatmap", "figure"),
        Output("heatmap-panel-title", "children"),
        Output("heatmap-panel-subtitle", "children"),
        Output("topic-list", "children"),
        Output("channel-mix", "children"),
    ],
    [Input("metric-selector", "value"), Input("selected-date-store", "data")],
)
def render_dashboard(metric, selected_date):
    selected_date = valid_selected_date(metric, selected_date)
    if selected_date not in payload:
        selected_date = max(payload)

    copy = metric_copy(metric)
    day = payload[selected_date]
    return (
        make_timeline(metric, selected_date),
        copy["timeline_title"],
        selected_date,
        make_cards(day, metric),
        make_spread_bars(day, metric),
        make_heatmap(day, metric),
        copy["heatmap_title"],
        copy["lower_subtitle"],
        make_topic_list(day, metric),
        make_channel_mix(day),
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
