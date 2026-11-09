# ============================================================
# EcoBot — Web App Streamlit
# EcoGrid Solutions | Analista Energetico AI
# Versione: 1.2 | Python 3.10+ (testato su 3.12)
# ============================================================

import os
import re
import json
import warnings
# Sopprime i DeprecationWarning di LangChain PRIMA di qualsiasi import
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Backend non-interattivo: salva PNG senza GUI
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# LangChain 0.3.x: approccio moderno per la memoria conversazionale.
# Si usano direttamente HumanMessage e AIMessage invece di ConversationBufferMemory
# (che è ancora funzionante ma deprecated in 0.3.x).
from langchain_core.messages import HumanMessage, AIMessage

# ── Caricamento API Key dal file .env ─────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ── Configurazione Pagina Streamlit ──────────────────────────────────────────
st.set_page_config(
    page_title='EcoBot | EcoGrid Solutions',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Verifica API Key prima di tutto il resto
if not OPENAI_API_KEY:
    st.error('❌ OPENAI_API_KEY non trovata. Assicurati che il file .env esista con la tua chiave.')
    st.stop()

# ── CSS personalizzato ────────────────────────────────────────────────────────
st.markdown('''
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #2ecc71; }
    .metric-positive { color: #2ecc71; font-weight: bold; }
    .metric-negative { color: #e74c3c; font-weight: bold; }
    div[data-testid="stChatMessage"] { border-radius: 12px; }
</style>
''', unsafe_allow_html=True)

# ── Header principale ─────────────────────────────────────────────────────────
st.markdown('<p class="main-header">⚡ EcoBot — Analista Energetico AI</p>', unsafe_allow_html=True)
st.caption('Powered by EcoGrid Solutions | Comunità Energetiche Rinnovabili')
st.divider()

# ── Directory per i grafici generati da Streamlit ────────────────────────────
CHARTS_DIR = 'charts_streamlit'
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Inizializzazione Session State ────────────────────────────────────────────
# Streamlit è stateless: session_state mantiene i dati tra un rerun e l'altro
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_production' not in st.session_state:
    st.session_state.df_production = None
if 'df_consumption' not in st.session_state:
    st.session_state.df_consumption = None
if 'df_production_raw' not in st.session_state:
    # Copia GREZZA (pre-cleaning) della produzione: serve a detect_anomalies
    # per trovare i guasti originali dei sensori, che il cleaning ha già corretto
    st.session_state.df_production_raw = None
if 'chat_history' not in st.session_state:
    # Approccio moderno LangChain 0.3.x: lista di oggetti HumanMessage/AIMessage
    # Più semplice e senza deprecation warning rispetto a ConversationBufferMemory
    st.session_state.chat_history = []


# ──────────────────────────────────────────────────────────────────────────────
# FUNZIONI DI DATA CLEANING
# Separate dall'agente: il cleaning avviene al momento del caricamento CSV
# ──────────────────────────────────────────────────────────────────────────────

def clean_production_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Pulisce il DataFrame della produzione solare:
    - Converte timestamp
    - Rimuove valori negativi (fisicamente impossibili per un pannello solare)
    - Interpola i valori NaN con metodo lineare
    - Rimuove timestamp duplicati
    '''
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Valori negativi → impossibili fisicamente → NaN
    df.loc[df['kwh_produced'] < 0, 'kwh_produced'] = np.nan
    # Interpolazione lineare per serie temporali continue
    df['kwh_produced'] = df['kwh_produced'].interpolate(method='linear').fillna(0)
    # Rimozione duplicati e ordinamento temporale
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


def clean_consumption_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Pulisce il DataFrame dei consumi domestici:
    - Converte timestamp
    - Gestisce valori negativi e NaN
    - Rimuove timestamp duplicati
    '''
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.loc[df['kwh_consumed'] < 0, 'kwh_consumed'] = np.nan
    # Fallback: se interpolazione non basta, usa la media del periodo
    df['kwh_consumed'] = df['kwh_consumed'].interpolate(method='linear').fillna(
        df['kwh_consumed'].mean()
    )
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY DEI TOOL LANGCHAIN
# I tool sono funzioni che l'LLM può decidere autonomamente di chiamare.
# Usiamo una factory (make_tools) perché i tool devono accedere a
# st.session_state, che non è disponibile a livello di modulo.
# ──────────────────────────────────────────────────────────────────────────────

def make_tools():
    '''
    Crea e restituisce la lista dei 7 tool LangChain.
    Ogni tool è una funzione Python decorata con @tool che:
    - Riceve parametri in formato stringa (LangChain standard)
    - Accede ai DataFrame in st.session_state
    - Restituisce risultati come JSON string (facilmente interpretabile dall LLM)
    '''

    @tool
    def load_and_clean_data(dummy: str = '') -> str:
        '''
        Verifica lo stato dei dati caricati e restituisce un report di qualità.
        Usa questo tool come PRIMO PASSO prima di qualsiasi analisi.
        Il cleaning è già avvenuto al momento del caricamento CSV.
        '''
        try:
            df_prod = st.session_state.df_production
            df_cons = st.session_state.df_consumption
            if df_prod is None or df_cons is None:
                return json.dumps({
                    'status': 'error',
                    'message': 'Nessun dato caricato. Carica i file CSV dalla sidebar prima di procedere.'
                })
            report = {
                'status': 'success',
                'produzione': {
                    'righe': len(df_prod),
                    'periodo': f'{df_prod["timestamp"].min().strftime("%Y-%m-%d")} -> {df_prod["timestamp"].max().strftime("%Y-%m-%d")}',
                    'null_residui': int(df_prod['kwh_produced'].isna().sum()),
                    'min_kwh': round(float(df_prod['kwh_produced'].min()), 3),
                    'max_kwh': round(float(df_prod['kwh_produced'].max()), 3),
                },
                'consumi': {
                    'righe': len(df_cons),
                    'periodo': f'{df_cons["timestamp"].min().strftime("%Y-%m-%d")} -> {df_cons["timestamp"].max().strftime("%Y-%m-%d")}',
                    'null_residui': int(df_cons['kwh_consumed'].isna().sum()),
                    'min_kwh': round(float(df_cons['kwh_consumed'].min()), 3),
                    'max_kwh': round(float(df_cons['kwh_consumed'].max()), 3),
                }
            }
            return json.dumps(report, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})

    @tool
    def calculate_solar_production(period: str = 'total') -> str:
        '''
        Calcola le statistiche della produzione solare dell'azienda locale.
        Parametro period: "total" (tutto il periodo), "daily" (giornaliero), "weekly" (settimanale).
        Restituisce valori in kWh.
        '''
        try:
            df = st.session_state.df_production.copy()
            df['date'] = df['timestamp'].dt.date
            df['week'] = df['timestamp'].dt.isocalendar().week
            df['hour'] = df['timestamp'].dt.hour
            if period == 'daily':
                result = df.groupby('date')['kwh_produced'].sum()
                return json.dumps({
                    'period': 'daily',
                    'media_giornaliera_kwh': round(float(result.mean()), 2),
                    'giorno_migliore': str(result.idxmax()),
                    'max_kwh': round(float(result.max()), 2),
                    'ultimi_7_giorni': {str(k): round(v, 2) for k, v in result.tail(7).items()}
                }, ensure_ascii=False)
            elif period == 'weekly':
                result = df.groupby('week')['kwh_produced'].sum()
                return json.dumps({
                    'period': 'weekly',
                    'dati_settimanali': {str(k): round(v, 2) for k, v in result.items()}
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    'period': 'total',
                    'totale_kwh_prodotti': round(float(df['kwh_produced'].sum()), 2),
                    'media_oraria_kwh': round(float(df['kwh_produced'].mean()), 3),
                    'ora_picco_produzione': int(df.groupby('hour')['kwh_produced'].mean().idxmax()),
                    'giorni_coperti': int(df['timestamp'].dt.date.nunique()),
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})

    @tool
    def calculate_consumption(period: str = 'total') -> str:
        '''
        Calcola le statistiche dei consumi energetici domestici dell\'utente.
        Parametro period: "total", "daily", "hourly" (media per ora del giorno), "weekly".
        Restituisce valori in kWh.
        '''
        try:
            df = st.session_state.df_consumption.copy()
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['week'] = df['timestamp'].dt.isocalendar().week
            if period == 'daily':
                result = df.groupby('date')['kwh_consumed'].sum()
                return json.dumps({
                    'period': 'daily',
                    'media_giornaliera_kwh': round(float(result.mean()), 2),
                    'ultimi_7_giorni': {str(k): round(v, 2) for k, v in result.tail(7).items()}
                }, ensure_ascii=False)
            elif period == 'hourly':
                result = df.groupby('hour')['kwh_consumed'].mean()
                return json.dumps({
                    'period': 'hourly',
                    'ore_picco_consumo': result.nlargest(3).index.tolist(),
                    'media_per_ora': {str(k): round(v, 3) for k, v in result.items()}
                }, ensure_ascii=False)
            elif period == 'weekly':
                result = df.groupby('week')['kwh_consumed'].sum()
                return json.dumps({
                    'period': 'weekly',
                    'dati_settimanali': {str(k): round(v, 2) for k, v in result.items()}
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    'period': 'total',
                    'totale_kwh_consumati': round(float(df['kwh_consumed'].sum()), 2),
                    'media_oraria_kwh': round(float(df['kwh_consumed'].mean()), 3),
                    'ore_picco_consumo': df.groupby('hour')['kwh_consumed'].mean().nlargest(3).index.tolist(),
                    'giorni_coperti': int(df['timestamp'].dt.date.nunique()),
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})

    @tool
    def detect_anomalies(dummy: str = '') -> str:
        '''
        Analizza i dati GREZZI dei sensori IoT (prima del cleaning) per rilevare anomalie:
        - Valori mancanti NaN (sensore rotto o disconnesso)
        - Valori negativi (errore di calibrazione sensore)
        - Picchi anomali statistici (oltre 3 deviazioni standard)
        - Gap temporali (ore non registrate dal sensore)
        Usa questo tool per rispondere a domande su guasti o qualità dei dati.
        '''
        try:
            # Analizziamo la copia GREZZA: sui dati già puliti le anomalie
            # risulterebbero sempre zero (stessa logica del notebook)
            df_raw = st.session_state.df_production_raw.copy()
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            nan_count = int(df_raw['kwh_produced'].isna().sum())
            neg_count = int((df_raw['kwh_produced'] < 0).sum())
            # Outlier statistici (oltre 3 sigma)
            mean_v = float(df_raw['kwh_produced'].mean())
            std_v = float(df_raw['kwh_produced'].std())
            outliers = df_raw[df_raw['kwh_produced'] > mean_v + 3 * std_v]
            # Gap temporali
            expected = pd.date_range(df_raw['timestamp'].min(), df_raw['timestamp'].max(), freq='h')
            gap_count = len(expected.difference(df_raw['timestamp']))
            # Dettaglio dei record anomali
            nan_timestamps = [
                str(t) for t in df_raw[df_raw['kwh_produced'].isna()]['timestamp'].head(5).tolist()
            ]
            neg_records = df_raw[df_raw['kwh_produced'] < 0][['timestamp', 'kwh_produced']].head(3).values.tolist()
            quality = 'CRITICA ⚠️' if (nan_count + neg_count) > 10 else 'ACCETTABILE ✅'
            return json.dumps({
                'anomalie_nei_dati_grezzi': {
                    'valori_nan': nan_count,
                    'valori_negativi': neg_count,
                    'outlier_3sigma': len(outliers),
                    'gap_temporali_ore': gap_count
                },
                'nan_timestamps': nan_timestamps,
                'record_negativi': [[str(r[0]), round(float(r[1]), 2)] for r in neg_records],
                'valutazione_qualita': quality,
                'stato_dopo_cleaning': 'Tutti i valori anomali corretti tramite interpolazione lineare.'
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})

    @tool
    def compare_production_vs_consumption(period: str = 'daily') -> str:
        '''
        Confronta la produzione solare con i consumi domestici.
        Calcola: surplus energetico, deficit, tasso di autosufficienza.
        Parametro period: "daily" o "weekly".
        '''
        try:
            df_prod = st.session_state.df_production.copy()
            df_cons = st.session_state.df_consumption.copy()
            df = pd.merge(df_prod, df_cons, on='timestamp', how='inner')
            df['surplus'] = df['kwh_produced'] - df['kwh_consumed']
            df['date'] = df['timestamp'].dt.date
            total_prod = round(float(df['kwh_produced'].sum()), 2)
            total_cons = round(float(df['kwh_consumed'].sum()), 2)
            total_surplus = round(total_prod - total_cons, 2)
            self_suff = round(min(total_prod, total_cons) / total_cons * 100, 1)
            daily = df.groupby('date')['surplus'].sum()
            return json.dumps({
                'periodo': period,
                'totale_prodotto_kwh': total_prod,
                'totale_consumato_kwh': total_cons,
                'surplus_totale_kwh': total_surplus,
                'autosufficienza_pct': self_suff,
                'bilancio': 'SURPLUS ENERGETICO ✅' if total_surplus > 0 else 'DEFICIT ENERGETICO ⚠️',
                'giorni_in_surplus': int((daily > 0).sum()),
                'giorni_in_deficit': int((daily < 0).sum()),
                'valore_surplus_euro': round(max(0, total_surplus) * 0.25, 2)
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})

    @tool
    def plot_energy_chart(chart_type: str = 'daily_comparison') -> str:
        '''
        Genera e salva un grafico energetico come file PNG.
        Tipi disponibili:
        - "daily_comparison": produzione vs consumo (ultimi 14 giorni)
        - "hourly_pattern": profilo orario medio con aree surplus/deficit
        - "weekly_surplus": surplus/deficit per settimana (barre blu/arancio)
        - "production_timeseries": serie temporale completa della produzione
        Restituisce il percorso del file PNG salvato (importante per visualizzarlo).
        '''
        try:
            df_prod = st.session_state.df_production.copy()
            df_cons = st.session_state.df_consumption.copy()
            df = pd.merge(df_prod, df_cons, on='timestamp', how='inner')
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['week'] = df['timestamp'].dt.isocalendar().week
            df['surplus'] = df['kwh_produced'] - df['kwh_consumed']

            plt.style.use('dark_background')
            colors = {
                'prod': '#2ecc71',
                'cons': '#e74c3c',
                'surplus': '#3498db',
                'deficit': '#e67e22'
            }
            fig, ax = plt.subplots(figsize=(10, 5))

            if chart_type == 'daily_comparison':
                daily = df.groupby('date').agg(
                    kwh_produced=('kwh_produced', 'sum'),
                    kwh_consumed=('kwh_consumed', 'sum')
                ).tail(14)
                x = range(len(daily))
                w = 0.38
                # Niente emoji nelle etichette: i font di matplotlib non hanno i glifi
                ax.bar([i - w/2 for i in x], daily['kwh_produced'], w,
                       label='Produzione Solare', color=colors['prod'], alpha=0.87)
                ax.bar([i + w/2 for i in x], daily['kwh_consumed'], w,
                       label='Consumo Domestico', color=colors['cons'], alpha=0.87)
                ax.set_xticks(list(x))
                ax.set_xticklabels([str(d) for d in daily.index], rotation=45, ha='right', fontsize=7)
                ax.set_title('Produzione Solare vs Consumo Domestico — Ultimi 14 Giorni',
                             fontsize=13, fontweight='bold', pad=12)
                ax.set_ylabel('Energia (kWh)')
                ax.legend(fontsize=10)
                filename = 'daily_comparison.png'

            elif chart_type == 'hourly_pattern':
                hourly = df.groupby('hour').agg(
                    prod=('kwh_produced', 'mean'),
                    cons=('kwh_consumed', 'mean')
                )
                ax.plot(hourly.index, hourly['prod'], 'o-',
                        color=colors['prod'], lw=2.5, ms=5, label='Produzione media')
                ax.plot(hourly.index, hourly['cons'], 's-',
                        color=colors['cons'], lw=2.5, ms=5, label='Consumo medio')
                ax.fill_between(hourly.index, hourly['prod'], hourly['cons'],
                                where=hourly['prod'] >= hourly['cons'],
                                alpha=0.18, color=colors['prod'], label='Surplus')
                ax.fill_between(hourly.index, hourly['prod'], hourly['cons'],
                                where=hourly['prod'] < hourly['cons'],
                                alpha=0.18, color=colors['cons'], label='Deficit')
                ax.set_xticks(range(24))
                ax.set_xticklabels([f'{h}:00' for h in range(24)], rotation=45, ha='right', fontsize=7)
                ax.set_title('Profilo Orario Medio — Produzione vs Consumo',
                             fontsize=13, fontweight='bold', pad=12)
                ax.set_xlabel('Ora del Giorno')
                ax.set_ylabel('kWh medi')
                ax.legend(fontsize=9)
                filename = 'hourly_pattern.png'

            elif chart_type == 'weekly_surplus':
                weekly = df.groupby('week')['surplus'].sum()
                bar_colors = [colors['surplus'] if s >= 0 else colors['deficit']
                              for s in weekly.values]
                ax.bar(weekly.index.astype(str), weekly.values,
                       color=bar_colors, alpha=0.87, edgecolor='white', linewidth=0.5)
                ax.axhline(0, color='white', lw=1.5, ls='--', alpha=0.6)
                ax.set_title('Surplus / Deficit Energetico Settimanale',
                             fontsize=13, fontweight='bold', pad=12)
                ax.set_xlabel('Settimana')
                ax.set_ylabel('kWh  |  Blu = Surplus   Arancio = Deficit')
                filename = 'weekly_surplus.png'

            else:  # production_timeseries
                daily_p = df.groupby('date')['kwh_produced'].sum()
                x_vals = range(len(daily_p))
                ax.plot(x_vals, daily_p.values, color=colors['prod'], lw=1.8, alpha=0.9)
                ax.fill_between(x_vals, daily_p.values, alpha=0.25, color=colors['prod'])
                step = max(1, len(daily_p) // 10)
                ax.set_xticks(list(x_vals)[::step])
                ax.set_xticklabels(
                    [str(d) for d in list(daily_p.index)[::step]],
                    rotation=45, ha='right', fontsize=7
                )
                ax.set_title('Serie Temporale Produzione Solare',
                             fontsize=13, fontweight='bold', pad=12)
                ax.set_ylabel('kWh Prodotti')
                filename = 'production_timeseries.png'

            plt.tight_layout()
            filepath = os.path.join(CHARTS_DIR, filename)
            plt.savefig(filepath, dpi=130, bbox_inches='tight', facecolor='#0e1117')
            plt.close()

            return json.dumps({
                'status': 'success',
                'chart_type': chart_type,
                'filepath': filepath,
                'message': f'Grafico salvato in: {filepath}'
            }, ensure_ascii=False)

        except Exception as e:
            plt.close()
            return json.dumps({'status': 'error', 'message': str(e)})

    @tool
    def get_energy_advice(focus: str = 'general') -> str:
        '''
        Genera consigli energetici personalizzati basati sui dati reali dell\'utente.
        Parametro focus:
        - "general": panoramica e consigli generali
        - "peak_shifting": ottimizzazione fasce orarie per elettrodomestici
        - "storage": valutazione acquisto batteria di accumulo
        - "savings": stima risparmio economico in bolletta
        '''
        try:
            df_prod = st.session_state.df_production.copy()
            df_cons = st.session_state.df_consumption.copy()
            df = pd.merge(df_prod, df_cons, on='timestamp', how='inner')
            df['hour'] = df['timestamp'].dt.hour
            df['surplus'] = df['kwh_produced'] - df['kwh_consumed']

            hourly = df.groupby('hour').agg(
                avg_prod=('kwh_produced', 'mean'),
                avg_cons=('kwh_consumed', 'mean'),
                avg_surplus=('surplus', 'mean')
            )
            surplus_hours = sorted(hourly[hourly['avg_surplus'] > 0].index.tolist())
            peak_cons_hours = hourly['avg_cons'].nlargest(3).index.tolist()
            total_surplus = float(df['surplus'].sum())
            total_cons = float(df_cons['kwh_consumed'].sum())
            total_prod = float(df_prod['kwh_produced'].sum())
            self_suff = round(min(total_prod, total_cons) / total_cons * 100, 1)
            tariff = 0.25  # €/kWh tariffa media
            surplus_value = round(max(0, total_surplus) * tariff, 2)
            monthly_savings = round(total_cons * tariff * self_suff / 100, 2)

            advice_map = {
                'general': [
                    f'✅ Autosufficienza attuale: {self_suff}% del fabbisogno coperto da energia solare',
                    f'🔆 Ore ottimali per elettrodomestici (surplus solare): {surplus_hours[:5]}',
                    f'⚠️ Picchi di consumo rilevati alle ore {peak_cons_hours}: considera di redistribuirli',
                    f'💰 Surplus energetico totale del periodo: {round(max(0, total_surplus), 1)} kWh',
                    '🔋 Valuta una batteria di accumulo per sfruttare il surplus nelle ore notturne'
                ],
                'peak_shifting': [
                    f'🕐 Programma lavatrice e lavastoviglie nelle ore {surplus_hours[:3]}',
                    f'⚡ Evita i picchi nelle ore {peak_cons_hours}: sono le più costose in deficit solare',
                    '☀️ Fascia 10:00-15:00 (sole al massimo): ideale per elettrodomestici ad alta potenza',
                    '📱 Usa le funzioni di programmazione oraria degli elettrodomestici smart',
                    '🌡️ Carica il boiler o il climatizzatore durante le ore di surplus'
                ],
                'storage': [
                    f'🔋 Surplus mensile: {round(max(0, total_surplus), 1)} kWh → dimensione batteria consigliata: 5-10 kWh',
                    f'💰 Valore economico del surplus: ~€{surplus_value} al mese (tariffa €{tariff}/kWh)',
                    f'⚡ Con accumulo: potresti coprire il consumo serale ({peak_cons_hours}) con energia solare',
                    '🌱 ROI stimato per un sistema di accumulo in contesto CER: 5-7 anni',
                    '📊 In una CER, il surplus in eccesso può essere ceduto ad altri utenti per guadagno extra'
                ],
                'savings': [
                    f'💡 Consumo totale del periodo: {round(total_cons, 1)} kWh',
                    f'💰 Bolletta stimata senza energia solare: ~€{round(total_cons * tariff, 2)}',
                    f'✅ Risparmio stimato grazie alla CER: ~€{monthly_savings} ({self_suff}% autosufficienza)',
                    f'📈 Potenziale aggiuntivo: spostando i consumi nelle ore {surplus_hours[:3]} potresti arrivare al {min(100, self_suff + 10):.0f}% di autosufficienza',
                    '🏦 Considera l\'iscrizione alla CER locale per benefici fiscali aggiuntivi'
                ]
            }

            return json.dumps({
                'focus': focus,
                'autosufficienza_attuale': f'{self_suff}%',
                'ore_surplus_solare': surplus_hours,
                'ore_picco_consumo': peak_cons_hours,
                'risparmio_stimato_euro': monthly_savings,
                'consigli': advice_map.get(focus, advice_map['general'])
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})

    return [
        load_and_clean_data,
        calculate_solar_production,
        calculate_consumption,
        detect_anomalies,
        compare_production_vs_consumption,
        plot_energy_chart,
        get_energy_advice
    ]


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Prompt Engineering per EcoBot
# Definisce identità, ruolo, metodo di ragionamento e stile comunicativo
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = '''Sei EcoBot, l\'analista energetico virtuale di EcoGrid Solutions —
una startup specializzata in comunità energetiche rinnovabili (CER).

Il tuo compito è aiutare le famiglie a comprendere i propri dati energetici,
ottimizzare i consumi e risparmiare in bolletta, sfruttando al meglio l\'energia solare.

## Il tuo metodo di ragionamento (Chain-of-Thought)
Prima di rispondere, segui SEMPRE questi passaggi mentali:
1. COMPRENDI la domanda: vuole dati grezzi, analisi, visualizzazione o consiglio?
2. IDENTIFICA i tool necessari: quali strumenti uso? In quale ordine?
3. ESEGUI: chiama sempre load_and_clean_data prima se non hai ancora verificato i dati
4. INTERPRETA: cosa significano i numeri per questa famiglia specifica?
5. RISPONDI: chiaramente, con dati precisi + almeno un consiglio pratico finale

## Regole operative
- Usa SEMPRE load_and_clean_data come primo tool se i dati non sono stati verificati
- Quando generi un grafico, indica SEMPRE il filepath esatto restituito dal tool
- Esprimi l\'energia in kWh e spiega cosa significano i valori
- Segnala proattivamente anomalie o inefficienze che noti nei dati
- Concludi OGNI risposta con almeno un consiglio pratico personalizzato

## Stile di comunicazione
- Usa emoji per leggibilità (⚡ energia, 🏠 consumo, ☀️ solare, 💰 risparmio, 🔋 accumulo)
- Parla come un consulente energetico amico: professionale ma accessibile
- Evita il gergo tecnico; se lo usi, spiegalo in una parola
- Le tue risposte devono essere utili anche per chi non ha competenze tecniche

Il tuo obiettivo finale: aiutare ogni famiglia a ridurre la bolletta e vivere in modo più sostenibile! 🌱
'''


# ──────────────────────────────────────────────────────────────────────────────
# CREAZIONE AGENTE (con cache Streamlit)
# @st.cache_resource evita di ricreare l'agente ad ogni rerun della pagina
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def create_agent(_api_key: str):
    '''
    Factory dell\'agente LangChain con caching Streamlit.
    Usa create_openai_tools_agent: più affidabile di ReAct per modelli OpenAI
    perché sfrutta il native function calling nativo di GPT-4o.
    '''
    llm = ChatOpenAI(
        model='gpt-4o',
        temperature=0.3,     # Bassa variabilità: risposte più precise e coerenti
        api_key=_api_key
    )
    tools = make_tools()
    prompt = ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,      # Gestione automatica errori di parsing LLM
        max_iterations=10,               # Limite iterazioni: previene loop infiniti
        max_execution_time=120,          # Timeout 2 minuti
        verbose=False,                   # In produzione: verbose=False per non inquinare log
        return_intermediate_steps=True   # Espone le chiamate ai tool: serve per il path del grafico
    )
    return executor


def ask_agent(question: str) -> tuple:
    '''
    Funzione principale di interazione con EcoBot.

    Parametri:
        question: domanda dell\'utente in linguaggio naturale

    Restituisce:
        tuple (risposta_testuale: str, filepath_grafico: str | None)
        Il filepath del grafico è None se l\'agente non ha generato grafici.
    '''
    try:
        executor = create_agent(OPENAI_API_KEY)
        # Recupera la storia come lista di HumanMessage/AIMessage (approccio LangChain 0.3.x)
        history = st.session_state.chat_history

        response = executor.invoke({
            'input': question,
            'chat_history': history
        })

        answer = response['output']

        # Aggiorna la storia conversazionale con i nuovi messaggi
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))

        # ── Estrazione robusta del filepath del grafico ──────────────────────
        # Strategia primaria: leggiamo il path direttamente dal JSON restituito
        # dal tool plot_energy_chart attraverso gli intermediate_steps.
        # Cosi' non dipendiamo dal fatto che l'LLM ripeta la stringa nel testo.
        chart_path = None
        for action, observation in response.get('intermediate_steps', []):
            if action.tool == 'plot_energy_chart':
                try:
                    result = json.loads(observation)
                    candidate = result.get('filepath')
                    if candidate and os.path.exists(candidate):
                        chart_path = candidate.replace('\\', '/')
                except (json.JSONDecodeError, TypeError):
                    pass  # Se il tool ha fallito, ignoriamo e proviamo il fallback

        # Strategia di fallback: regex sul testo (per retro-compatibilita')
        if chart_path is None:
            match = re.search(r'charts_streamlit[/\\][\w_]+\.png', answer)
            if match:
                candidate = match.group(0).replace('\\', '/')
                if os.path.exists(candidate):
                    chart_path = candidate

        # Il grafico viene mostrato con st.image: rimuoviamo dal testo eventuali
        # immagini/link markdown a PNG locali scritti dall'LLM, che Streamlit
        # non può servire e mostrerebbe come immagini rotte.
        answer = re.sub(r'!?\[([^\]]*)\]\([^)]*\.png\)', r'\1', answer)

        return answer, chart_path

    except Exception as e:
        error_answer = f'⚠️ Si è verificato un errore durante l\'analisi: {str(e)}\n\nProva a riformulare la domanda o verifica che i file CSV siano stati caricati correttamente.'
        return error_answer, None


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Upload CSV e metriche rapide
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('## ☀️ EcoGrid Solutions')
    st.markdown('*Analista Energetico AI*')
    st.divider()

    st.markdown('### 📂 Carica i tuoi dati')
    st.caption('Carica i file CSV generati dal notebook per iniziare l\'analisi')

    prod_file = st.file_uploader(
        '⚡ Produzione Solare',
        type='csv',
        key='prod_upload',
        help='Carica solar_production_raw.csv'
    )
    cons_file = st.file_uploader(
        '🏠 Consumi Domestici',
        type='csv',
        key='cons_upload',
        help='Carica household_consumption.csv'
    )

    # Elaborazione file caricati
    if prod_file and cons_file:
        try:
            df_prod_raw = pd.read_csv(prod_file, parse_dates=['timestamp'])
            df_cons_raw = pd.read_csv(cons_file, parse_dates=['timestamp'])

            # Conserva i dati grezzi (per detect_anomalies) e applica il data cleaning
            st.session_state.df_production_raw = df_prod_raw.copy()
            st.session_state.df_production = clean_production_data(df_prod_raw)
            st.session_state.df_consumption = clean_consumption_data(df_cons_raw)
            st.session_state.data_loaded = True

            st.success('✅ Dati caricati e puliti con successo!')

            # ── Metriche rapide in sidebar ──
            st.markdown('### 📊 Riepilogo Periodo')
            total_prod = float(st.session_state.df_production['kwh_produced'].sum())
            total_cons = float(st.session_state.df_consumption['kwh_consumed'].sum())
            surplus = total_prod - total_cons
            self_suff = min(total_prod, total_cons) / total_cons * 100

            # Unità di misura nell'etichetta: nelle colonne strette della sidebar
            # un valore come "1145 kWh" verrebbe troncato ("1145 ...")
            col1, col2 = st.columns(2)
            with col1:
                st.metric('⚡ Produzione (kWh)', f'{total_prod:.0f}')
                st.metric('🏠 Consumo (kWh)', f'{total_cons:.0f}')
            with col2:
                st.metric('🔄 Surplus (kWh)', f'{surplus:.0f}',
                          delta=f'{surplus:+.0f}')
                st.metric('🌱 Autosufficienza', f'{self_suff:.1f}%')

        except Exception as e:
            st.error(f'❌ Errore nel caricamento: {str(e)}')
            st.caption('Verifica che i file siano CSV validi con colonne timestamp e kwh.')

    else:
        st.info('👆 Carica entrambi i file CSV per iniziare')

    # ── Domande di esempio (bottoni rapidi) ──
    st.divider()
    st.markdown('### 💬 Domande Rapide')
    example_qs = [
        '🔍 Ci sono stati guasti ai sensori?',
        '📊 Mostrami produzione vs consumo',
        '⏰ Quando conviene usare la lavatrice?',
        '🌱 Sono autosufficiente energeticamente?',
        '📈 Genera il grafico del profilo orario',
        '💰 Come posso ridurre la bolletta?',
    ]
    for q in example_qs:
        if st.button(q, key=f'btn_{q[:15]}', width='stretch'):
            st.session_state['pending_question'] = q.split(' ', 1)[1]  # Rimuove l\'emoji iniziale

    # ── Reset Chat ──
    st.divider()
    if st.button('🗑️ Resetta Conversazione', width='stretch', type='secondary'):
        st.session_state.messages = []
        st.session_state.chat_history = []   # Reset lista HumanMessage/AIMessage
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# AREA PRINCIPALE — Chat Interface
# ──────────────────────────────────────────────────────────────────────────────

# Messaggio di benvenuto (solo se la chat è vuota)
if not st.session_state.messages:
    with st.chat_message('assistant', avatar='⚡'):
        st.markdown('''
**Ciao! Sono EcoBot 👋**

Sono il tuo analista energetico personale di **EcoGrid Solutions**.

Ecco cosa posso fare per te:
- 📊 **Analizzare** produzione solare e consumi domestici
- 🔍 **Rilevare** anomalie e guasti ai sensori IoT
- 📈 **Visualizzare** trend con grafici chiari e leggibili
- 💡 **Consigliarti** come ottimizzare i consumi e risparmiare

**Per iniziare** → Carica i tuoi file CSV dalla sidebar ⬅️
        ''')

# Visualizza la storia della conversazione
for msg in st.session_state.messages:
    avatar = '👤' if msg['role'] == 'user' else '⚡'
    with st.chat_message(msg['role'], avatar=avatar):
        st.markdown(msg['content'])
        # Mostra il grafico se presente nel messaggio
        if msg.get('chart_path') and os.path.exists(msg['chart_path']):
            st.image(msg['chart_path'], width='stretch',
                     caption=f'📊 {msg["chart_path"].split("/")[-1]}')

# ── Gestione input (da text box o bottoni sidebar) ────────────────────────────
pending = st.session_state.pop('pending_question', None)
user_input = pending or st.chat_input(
    'Fai una domanda sui tuoi dati energetici...',
    disabled=not st.session_state.data_loaded
)

if not st.session_state.data_loaded and not pending:
    st.warning('⚠️ **Carica prima i file CSV dalla sidebar** per abilitare la chat con EcoBot.')

# ── Elaborazione della risposta ───────────────────────────────────────────────
if user_input and st.session_state.data_loaded:

    # 1. Mostra il messaggio dell\'utente
    clean_input = user_input.strip()
    st.session_state.messages.append({'role': 'user', 'content': clean_input})
    with st.chat_message('user', avatar='👤'):
        st.markdown(clean_input)

    # 2. Genera la risposta dell\'agente con spinner
    with st.chat_message('assistant', avatar='⚡'):
        with st.spinner('🤔 EcoBot sta analizzando i dati...'):
            answer, chart_path = ask_agent(clean_input)

        st.markdown(answer)

        # 3. Mostra il grafico se generato
        if chart_path and os.path.exists(chart_path):
            st.image(chart_path, width='stretch',
                     caption=f'📊 Grafico generato: {chart_path.split("/")[-1]}')

    # 4. Salva il messaggio nella storia
    st.session_state.messages.append({
        'role': 'assistant',
        'content': answer,
        'chart_path': chart_path
    })
