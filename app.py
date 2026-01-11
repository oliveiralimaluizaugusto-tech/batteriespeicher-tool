"""
Batteriespeicher-Optimierungstool
=================================

Entwickelt im Rahmen der Bachelorarbeit:
"Entwicklung eines Python-basierten Tools zur Bestimmung optimaler 
Betriebsstrategien und Dimensionierung von Großbatteriespeichern 
unter Einbindung von Wind- und Solarprofilen"

TH Köln - Fakultät für Anlagen, Energie und Maschinensysteme

Drei Modi:
- Modus A: Wirtschaftlichkeitsanalyse für Großspeicher
- Modus B: Peak Shaving für Industriekunden (PyPSA-Optimierung)
- Modus C: NVP-Überbauung (PyPSA-Optimierung)

Starten mit: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import io
import json

# Matplotlib für Plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# PyPSA für Optimierung
import pypsa

# =============================================================================
# Hilfsfunktionen
# =============================================================================

def calculate_annuity_factor(rate, years):
    """
    Berechnet den Annuitätenfaktor für die Kapitalkosten.
    
    Formel: ANF = (i × (1+i)^n) / ((1+i)^n - 1)
    
    Args:
        rate: Zinssatz (z.B. 0.05 für 5%)
        years: Lebensdauer in Jahren
    
    Returns:
        Annuitätenfaktor
    """
    if rate == 0 or rate is None:
        return 1 / years if years > 0 else 1
    return (rate * (1 + rate) ** years) / ((1 + rate) ** years - 1)


# =============================================================================
# Seitenkonfiguration
# =============================================================================
st.set_page_config(
    page_title="Batteriespeicher-Optimierung",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS für benutzerfreundliches Design
# =============================================================================
st.markdown("""
<style>
    /* Hauptüberschriften */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Modus-Auswahl Karten */
    .mode-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .mode-card:hover {
        transform: scale(1.02);
    }
    .mode-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .mode-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Info-Boxen */
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    /* Schritt-Anzeige */
    .step-indicator {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .step {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #ddd;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 1rem;
        font-weight: bold;
    }
    .step-active {
        background-color: #1f77b4;
        color: white;
    }
    .step-done {
        background-color: #28a745;
        color: white;
    }
    
    /* Kennzahlen-Karten */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State initialisieren
# =============================================================================
def init_session_state():
    """Initialisiert alle Session-State-Variablen."""
    defaults = {
        'current_mode': None,  # 'A' oder 'B'
        'current_step': 1,
        # Modus A
        'generation_profile': None,
        'price_profile': None,
        'fcr_prices': None,
        'afrr_prices': None,
        'optimal_storage_size': None,
        'optimal_power': None,
        'optimization_results': None,
        'capacity_allocation': None,
        # Modus B
        'load_profile': None,
        'peak_load': None,
        'target_limit': None,
        'peak_shaving_results': None,
        # Gemeinsam
        'technical_params': {
            'efficiency': 0.88,
            'soc_min': 0.10,
            'soc_max': 0.90,
            'cycle_life': 6000,
            'calendar_life': 15,
        },
        'economic_params': {
            'capex_energy': 250,  # €/kWh
            'capex_power': 80,    # €/kW
            'opex_rate': 2.0,     # % von CAPEX
            'discount_rate': 6.0, # %
            'project_lifetime': 15,
        },
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# Hilfsfunktionen
# =============================================================================
def load_csv_file(uploaded_file, expected_columns=None):
    """
    Lädt eine CSV- oder Excel-Datei und gibt einen DataFrame zurück.
    
    Erkennt automatisch:
    - Trennzeichen (Komma, Semikolon, Tab)
    - Dezimalformat (Punkt oder Komma)
    - Datumsformate (verschiedene)
    - Währungssymbole (€, EUR, $) und entfernt diese
    - Excel-Dateien (.xlsx, .xls)
    """
    if uploaded_file is None:
        return None
    
    try:
        filename = uploaded_file.name.lower()
        
        # Excel-Dateien
        if filename.endswith(('.xlsx', '.xls')):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        else:
            # CSV-Dateien
            content = uploaded_file.getvalue().decode('utf-8')
            
            # Trennzeichen erkennen (Reihenfolge: Tab, Semikolon, Komma)
            first_lines = content.split('\n')[:5]
            first_line = first_lines[0]
            
            if '\t' in first_line:
                sep = '\t'
            elif ';' in first_line:
                sep = ';'
            else:
                sep = ','
            
            # Dezimaltrennzeichen erkennen
            # Wenn Semikolon als Trennzeichen, dann meist Komma als Dezimal
            if sep == ';':
                decimal = ','
            else:
                decimal = '.'
            
            # CSV laden
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep, decimal=decimal)
        
        # Spaltennamen bereinigen (Leerzeichen entfernen)
        df.columns = df.columns.str.strip()
        
        # Erste Spalte als Zeitstempel interpretieren
        first_col = df.columns[0]
        df[first_col] = parse_datetime_column(df[first_col])
        
        if df[first_col] is not None and not df[first_col].isna().all():
            df.set_index(first_col, inplace=True)
        
        # Werte-Spalten bereinigen (Währungssymbole entfernen, in Zahlen konvertieren)
        for col in df.columns:
            df[col] = clean_numeric_column(df[col])
        
        return df
    
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Datei: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def parse_datetime_column(series):
    """
    Versucht verschiedene Datumsformate zu parsen.
    
    Unterstützte Formate:
    - 2024-01-01 00:00
    - 01.01.2024 00:00
    - Jan 1, 2024 1:00
    - 1/1/2024 0:00
    - und viele mehr (pandas dayfirst/yearfirst Kombinationen)
    """
    if series is None:
        return None
    
    # Bereits datetime?
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    
    # String-Bereinigung
    series = series.astype(str).str.strip()
    
    # Liste von Formaten zum Ausprobieren
    date_formats = [
        None,  # pandas automatische Erkennung
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%d.%m.%Y %H:%M',
        '%d.%m.%Y %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M',
        '%b %d, %Y %H:%M',  # Jan 1, 2024 01:00
        '%b %d, %Y %I:%M %p',  # Jan 1, 2024 1:00 AM
        '%d-%m-%Y %H:%M',
        '%Y/%m/%d %H:%M',
    ]
    
    for fmt in date_formats:
        try:
            if fmt is None:
                # Automatische Erkennung mit verschiedenen Optionen
                for dayfirst in [True, False]:
                    for yearfirst in [True, False]:
                        try:
                            result = pd.to_datetime(series, dayfirst=dayfirst, yearfirst=yearfirst)
                            if not result.isna().all():
                                return result
                        except:
                            continue
            else:
                result = pd.to_datetime(series, format=fmt)
                if not result.isna().all():
                    return result
        except:
            continue
    
    # Letzter Versuch: pandas infer_datetime_format
    try:
        return pd.to_datetime(series, infer_datetime_format=True)
    except:
        return series


def clean_numeric_column(series):
    """
    Bereinigt eine Spalte und konvertiert sie in numerische Werte.
    
    Entfernt:
    - Währungssymbole (€, EUR, $, USD)
    - Tausendertrennzeichen
    - Leerzeichen
    - Prozentzeichen
    """
    if series is None:
        return None
    
    # Bereits numerisch?
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # In String konvertieren
    series = series.astype(str)
    
    # Währungssymbole und andere Zeichen entfernen
    replacements = [
        ('€', ''),
        ('EUR', ''),
        ('$', ''),
        ('USD', ''),
        ('%', ''),
        (' ', ''),
        ('\xa0', ''),  # Non-breaking space
    ]
    
    for old, new in replacements:
        series = series.str.replace(old, new, regex=False)
    
    # Tausendertrennzeichen behandeln
    # Erkennen ob Komma oder Punkt als Tausendertrennzeichen verwendet wird
    sample = series.dropna().head(100)
    
    # Zähle Vorkommen von Punkt und Komma
    dots = sample.str.count(r'\.').sum()
    commas = sample.str.count(',').sum()
    
    # Wenn mehr Kommas als Punkte und Kommas nicht am Ende (Dezimal) → Komma ist Tausender
    # Typisch deutsch: 1.000,00 → Punkt ist Tausender, Komma ist Dezimal
    # Typisch englisch: 1,000.00 → Komma ist Tausender, Punkt ist Dezimal
    
    # Heuristik: Prüfe Position des letzten Trennzeichens
    def convert_number(val):
        if pd.isna(val) or val == '' or val == 'nan':
            return np.nan
        
        val = str(val).strip()
        
        # Finde Position von Punkt und Komma
        last_dot = val.rfind('.')
        last_comma = val.rfind(',')
        
        if last_dot > last_comma:
            # Punkt ist Dezimaltrennzeichen (englisch): 1,000.50
            val = val.replace(',', '')
        elif last_comma > last_dot:
            # Komma ist Dezimaltrennzeichen (deutsch): 1.000,50
            val = val.replace('.', '').replace(',', '.')
        # Sonst: nur eines vorhanden, als Dezimal interpretieren
        elif last_comma >= 0:
            val = val.replace(',', '.')
        
        try:
            return float(val)
        except:
            return np.nan
    
    return series.apply(convert_number)


def format_number(value, decimals=0, suffix=""):
    """Formatiert eine Zahl für die Anzeige."""
    if value is None:
        return "-"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.{decimals}f} Mio. {suffix}".strip()
    elif abs(value) >= 1_000:
        return f"{value:,.{decimals}f} {suffix}".strip()
    else:
        return f"{value:,.{decimals}f} {suffix}".strip()


def calculate_curtailment(generation_profile, max_grid_power, storage_capacity_mwh, 
                          storage_power_mw, efficiency, soc_min, soc_max):
    """
    Berechnet die Abregelung bei gegebener Speichergröße.
    
    Returns:
        dict mit Ergebnissen (Abregelung, gespeicherte Energie, etc.)
    """
    n = len(generation_profile)
    
    # Arrays initialisieren
    soc = np.zeros(n + 1)
    soc[0] = (soc_min + soc_max) / 2 * storage_capacity_mwh  # Start bei 50%
    
    curtailment = np.zeros(n)
    grid_feed_in = np.zeros(n)
    storage_charge = np.zeros(n)
    storage_discharge = np.zeros(n)
    
    soc_min_mwh = soc_min * storage_capacity_mwh
    soc_max_mwh = soc_max * storage_capacity_mwh
    
    dt = 0.25  # 15 Minuten = 0.25 Stunden
    
    for t in range(n):
        gen = generation_profile.iloc[t]
        
        if gen > max_grid_power:
            # Überschuss vorhanden
            excess = gen - max_grid_power
            
            # Wie viel kann gespeichert werden?
            available_capacity = soc_max_mwh - soc[t]
            max_charge_power = min(storage_power_mw, available_capacity / dt / efficiency)
            actual_charge = min(excess, max_charge_power)
            
            storage_charge[t] = actual_charge
            curtailment[t] = excess - actual_charge
            grid_feed_in[t] = max_grid_power
            
            soc[t + 1] = soc[t] + actual_charge * efficiency * dt
            
        else:
            # Erzeugung unter Anschlussleistung
            grid_feed_in[t] = gen
            
            # Kann der Speicher zusätzlich einspeisen?
            grid_headroom = max_grid_power - gen
            available_energy = soc[t] - soc_min_mwh
            max_discharge_power = min(storage_power_mw, available_energy / dt)
            actual_discharge = min(grid_headroom, max_discharge_power * efficiency)
            
            storage_discharge[t] = actual_discharge / efficiency
            grid_feed_in[t] += actual_discharge
            
            soc[t + 1] = soc[t] - actual_discharge / efficiency * dt
    
    return {
        'curtailment_mwh': curtailment.sum() * dt,
        'curtailment_profile': curtailment,
        'grid_feed_in_mwh': grid_feed_in.sum() * dt,
        'grid_feed_in_profile': grid_feed_in,
        'storage_cycles': (storage_charge.sum() * dt) / storage_capacity_mwh if storage_capacity_mwh > 0 else 0,
        'soc_profile': soc[:-1],
        'total_generation_mwh': generation_profile.sum() * dt,
    }


def optimize_storage_size(generation_profile, max_grid_power, price_profile,
                          technical_params, economic_params, 
                          capacity_range=(10, 500), power_range=(5, 250), steps=20):
    """
    Iterative Optimierung der Speichergröße.
    
    Variiert Kapazität und Leistung und berechnet für jede Kombination:
    - Vermiedene Abregelung
    - Arbitrage-Erlöse
    - Wirtschaftlichkeit (NPV)
    """
    results = []
    
    capacities = np.linspace(capacity_range[0], capacity_range[1], steps)
    
    for capacity in capacities:
        # Leistung als Funktion der Kapazität (E/P zwischen 1 und 4)
        for ep_ratio in [1, 2, 3, 4]:
            power = capacity / ep_ratio
            
            if power < power_range[0] or power > power_range[1]:
                continue
            
            # Abregelung berechnen
            curtail_result = calculate_curtailment(
                generation_profile=generation_profile,
                max_grid_power=max_grid_power,
                storage_capacity_mwh=capacity,
                storage_power_mw=power,
                efficiency=technical_params['efficiency'],
                soc_min=technical_params['soc_min'],
                soc_max=technical_params['soc_max']
            )
            
            # Arbitrage-Erlöse berechnen (vereinfacht)
            if price_profile is not None and len(price_profile) > 0:
                arbitrage_revenue = estimate_arbitrage_revenue(
                    capacity_mwh=capacity,
                    power_mw=power,
                    price_profile=price_profile,
                    efficiency=technical_params['efficiency'],
                    cycles_available=365 - curtail_result['storage_cycles']
                )
            else:
                arbitrage_revenue = 0
            
            # Erlös aus vermiedener Abregelung
            avg_price = price_profile.mean() if price_profile is not None else 50
            avoided_curtailment_revenue = curtail_result['curtailment_mwh'] * avg_price * -1  # Negativ, da vermieden
            # Korrektur: Vermiedene Abregelung ist positiv
            generation_without_storage = calculate_curtailment(
                generation_profile, max_grid_power, 0, 0,
                technical_params['efficiency'],
                technical_params['soc_min'],
                technical_params['soc_max']
            )
            avoided_curtailment = generation_without_storage['curtailment_mwh'] - curtail_result['curtailment_mwh']
            avoided_curtailment_revenue = avoided_curtailment * avg_price
            
            # Investitionskosten
            capex = (capacity * 1000 * economic_params['capex_energy'] + 
                    power * 1000 * economic_params['capex_power'])
            
            # Jährliche Betriebskosten
            opex = capex * economic_params['opex_rate'] / 100
            
            # Jährlicher Erlös
            annual_revenue = arbitrage_revenue + avoided_curtailment_revenue
            
            # NPV berechnen
            discount_rate = economic_params['discount_rate'] / 100
            lifetime = economic_params['project_lifetime']
            
            npv = -capex
            for year in range(1, lifetime + 1):
                npv += (annual_revenue - opex) / ((1 + discount_rate) ** year)
            
            results.append({
                'capacity_mwh': capacity,
                'power_mw': power,
                'ep_ratio': ep_ratio,
                'curtailment_mwh': curtail_result['curtailment_mwh'],
                'avoided_curtailment_mwh': avoided_curtailment,
                'arbitrage_revenue': arbitrage_revenue,
                'avoided_curtailment_revenue': avoided_curtailment_revenue,
                'total_revenue': annual_revenue,
                'capex': capex,
                'opex': opex,
                'npv': npv,
                'storage_cycles': curtail_result['storage_cycles'],
            })
    
    return pd.DataFrame(results)


def estimate_arbitrage_revenue(capacity_mwh, power_mw, price_profile, efficiency, cycles_available):
    """
    Schätzt die Arbitrage-Erlöse basierend auf Preisspreads.
    
    Vereinfachte Berechnung:
    - Identifiziert tägliche Preisspreads
    - Berechnet mögliche Zyklen und Erlöse
    """
    if price_profile is None or len(price_profile) == 0:
        return 0
    
    # Tägliche Preisspreads berechnen
    prices = price_profile.values if hasattr(price_profile, 'values') else price_profile
    
    # Gruppiere nach Tagen (96 Werte pro Tag bei 15-min-Auflösung)
    n_days = len(prices) // 96
    daily_spreads = []
    
    for day in range(n_days):
        day_prices = prices[day * 96:(day + 1) * 96]
        spread = np.max(day_prices) - np.min(day_prices)
        daily_spreads.append(spread)
    
    avg_spread = np.mean(daily_spreads) if daily_spreads else 30  # Default 30 €/MWh
    
    # Usable capacity
    usable_capacity = capacity_mwh * (0.9 - 0.1)  # SoC-Bereich
    
    # Maximale Zyklen pro Jahr (begrenzt durch verfügbare Kapazität)
    max_daily_cycles = min(1, power_mw * 24 / usable_capacity)  # Max 1 Zyklus pro Tag
    annual_cycles = min(365 * max_daily_cycles, cycles_available)
    
    # Erlös pro Zyklus
    revenue_per_cycle = usable_capacity * avg_spread * efficiency
    
    return annual_cycles * revenue_per_cycle


def optimize_capacity_allocation(storage_capacity_mwh, storage_power_mw, 
                                 price_profile, fcr_prices, afrr_prices,
                                 efficiency, soc_min, soc_max):
    """
    Optimiert die Aufteilung der Speicherkapazität zwischen FCR, aFRR und Arbitrage.
    
    FCR-Anforderungen:
    - Symmetrische Vorhaltung (positiv und negativ)
    - 30-Minuten-Lieferfähigkeit bei vollem Abruf
    
    Returns:
        dict mit optimaler Aufteilung und Erlösen
    """
    results = []
    
    usable_capacity = storage_capacity_mwh * (soc_max - soc_min)
    
    # FCR: Max. Leistung basierend auf 30-min Lieferfähigkeit
    # Symmetrisch: Muss in beide Richtungen liefern können
    max_fcr_power = usable_capacity / 0.5 / 2  # 0.5h * 2 (symmetrisch)
    max_fcr_power = min(max_fcr_power, storage_power_mw)
    
    # Durchschnittliche Preise
    avg_fcr_price = fcr_prices.mean() if fcr_prices is not None and len(fcr_prices) > 0 else 15  # €/MW/h
    avg_afrr_price = afrr_prices.mean() if afrr_prices is not None and len(afrr_prices) > 0 else 8  # €/MW/h
    
    # Iteriere über verschiedene Aufteilungen
    for fcr_share in np.arange(0, 1.01, 0.1):
        for afrr_share in np.arange(0, 1.01 - fcr_share, 0.1):
            arbitrage_share = 1 - fcr_share - afrr_share
            
            # Leistungsaufteilung
            fcr_power = min(max_fcr_power * fcr_share, storage_power_mw * fcr_share)
            afrr_power = storage_power_mw * afrr_share
            arbitrage_power = storage_power_mw * arbitrage_share
            
            # Kapazitätsaufteilung
            # FCR benötigt Kapazität für 30-min Vorhaltung (symmetrisch)
            fcr_capacity_needed = fcr_power * 0.5 * 2
            afrr_capacity_needed = afrr_power * 0.25 * 2  # 15-min Aktivierung
            arbitrage_capacity = usable_capacity - fcr_capacity_needed - afrr_capacity_needed
            
            if arbitrage_capacity < 0:
                continue
            
            # Erlöse berechnen (nur Leistungsvorhaltung)
            hours_per_year = 8760
            
            fcr_revenue = fcr_power * avg_fcr_price * hours_per_year
            afrr_revenue = afrr_power * avg_afrr_price * hours_per_year
            
            # Arbitrage-Erlöse
            if price_profile is not None and arbitrage_capacity > 0:
                arbitrage_revenue = estimate_arbitrage_revenue(
                    capacity_mwh=arbitrage_capacity / (soc_max - soc_min),
                    power_mw=arbitrage_power,
                    price_profile=price_profile,
                    efficiency=efficiency,
                    cycles_available=500  # Annahme
                )
            else:
                arbitrage_revenue = 0
            
            total_revenue = fcr_revenue + afrr_revenue + arbitrage_revenue
            
            results.append({
                'fcr_share': fcr_share,
                'afrr_share': afrr_share,
                'arbitrage_share': arbitrage_share,
                'fcr_power_mw': fcr_power,
                'afrr_power_mw': afrr_power,
                'arbitrage_power_mw': arbitrage_power,
                'fcr_revenue': fcr_revenue,
                'afrr_revenue': afrr_revenue,
                'arbitrage_revenue': arbitrage_revenue,
                'total_revenue': total_revenue,
            })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        optimal = df.loc[df['total_revenue'].idxmax()]
        return {
            'optimal': optimal.to_dict(),
            'all_results': df,
        }
    else:
        return None


def calculate_peak_shaving(load_profile, target_limit, storage_power_mw, 
                           storage_capacity_mwh, efficiency, soc_min, soc_max,
                           pv_surplus=None):
    """
    Berechnet das Peak Shaving Ergebnis mit PyPSA-Optimierung.
    
    Zielfunktion: Minimiere Netzbezugskosten + Spitzenlastkosten
    
    Komponenten:
    - Bus: Lastknoten
    - Load: Verbraucherlast
    - Generator (grid): Netzbezug (zu minimieren)
    - Store: Batteriespeicher
    - Generator (pv_surplus): PV-Überschuss (falls vorhanden)
    """
    n = len(load_profile)
    dt = 0.25  # 15 Minuten = 0.25 Stunden
    
    # Effizienz aufteilen (sqrt für Laden und Entladen)
    eta_charge = np.sqrt(efficiency)
    eta_discharge = np.sqrt(efficiency)
    
    # Nutzbare Kapazität
    usable_capacity = storage_capacity_mwh * (soc_max - soc_min)
    initial_soc = usable_capacity * 0.5  # Start bei 50% der nutzbaren Kapazität
    
    has_pv = pv_surplus is not None and len(pv_surplus) == n
    
    # PyPSA Netzwerk erstellen
    network = pypsa.Network()
    network.set_snapshots(range(n))
    
    # Zeitauflösung setzen (für korrekte Energieberechnung)
    network.snapshot_weightings.loc[:, "generators"] = dt
    network.snapshot_weightings.loc[:, "stores"] = dt
    
    # Bus (Lastknoten)
    network.add("Bus", "load_bus")
    
    # Last
    network.add("Load", "demand",
                bus="load_bus",
                p_set=load_profile.values)
    
    # Netzbezug - mit hohen Kosten für Spitzen
    # Grundlast günstig, alles über target_limit sehr teuer
    network.add("Generator", "grid_base",
                bus="load_bus",
                p_nom=target_limit,
                marginal_cost=1)  # Normale Netzkosten
    
    network.add("Generator", "grid_peak",
                bus="load_bus",
                p_nom=load_profile.max() * 2,  # Genug Kapazität
                marginal_cost=1000)  # Sehr teuer = vermeiden
    
    # Batteriespeicher
    network.add("Store", "battery",
                bus="load_bus",
                e_nom=usable_capacity,
                e_initial=initial_soc,
                e_cyclic=False,
                standing_loss=0,
                e_min_pu=0,
                e_max_pu=1)
    
    # Laden und Entladen über Links für Wirkungsgrad
    network.add("Bus", "battery_internal")
    
    network.add("Link", "charge",
                bus0="load_bus",
                bus1="battery_internal",
                p_nom=storage_power_mw,
                efficiency=eta_charge,
                marginal_cost=0.1)  # Kleine Kosten für Laden
    
    network.add("Link", "discharge",
                bus0="battery_internal",
                bus1="load_bus",
                p_nom=storage_power_mw,
                efficiency=eta_discharge,
                marginal_cost=0)
    
    # Store mit internem Bus verbinden
    network.stores.loc["battery", "bus"] = "battery_internal"
    
    # PV-Überschuss (falls vorhanden)
    if has_pv:
        # PV-Überschuss als negative Last (Einspeisung)
        network.add("Generator", "pv_surplus",
                    bus="load_bus",
                    p_nom=pv_surplus.max() if pv_surplus.max() > 0 else 1,
                    p_max_pu=np.clip(pv_surplus.values / max(pv_surplus.max(), 0.001), 0, 1),
                    marginal_cost=-0.01)  # Kleiner Bonus für PV-Nutzung
    
    # Optimieren
    try:
        status = network.optimize(solver_name="highs", solver_options={"threads": 1})
        
        if status[0] != "ok":
            st.warning(f"⚠️ Optimierung nicht optimal gelöst: {status}")
            return None
    except Exception as e:
        st.error(f"❌ PyPSA-Optimierungsfehler: {e}")
        return None
    
    # Ergebnisse extrahieren
    grid_base = network.generators_t.p["grid_base"].values
    grid_peak = network.generators_t.p["grid_peak"].values
    grid_power = grid_base + grid_peak
    
    # Speicherleistung: positiv = entladen, negativ = laden
    charge_power = network.links_t.p0["charge"].values
    discharge_power = network.links_t.p1["discharge"].values
    battery_power = discharge_power - charge_power
    
    soc = network.stores_t.e["battery"].values
    
    # SOC in absoluten Werten (mit soc_min Offset)
    soc_absolute = soc + soc_min * storage_capacity_mwh
    
    original_peak = load_profile.max()
    new_peak = grid_power.max()
    
    # Unmet peaks (wo grid_peak > 0)
    unmet_peaks = np.maximum(grid_power - target_limit, 0)
    
    result = {
        'original_peak_mw': original_peak,
        'new_peak_mw': new_peak,
        'peak_reduction_mw': original_peak - new_peak,
        'target_achieved': new_peak <= target_limit * 1.01,
        'unmet_peaks_count': (grid_peak > 0.01).sum(),
        'grid_power_profile': grid_power,
        'battery_power_profile': battery_power,
        'soc_profile': soc_absolute,
        'optimization_status': 'optimal',
        'objective_value': network.objective,
    }
    
    if has_pv:
        pv_used = network.generators_t.p["pv_surplus"].values if "pv_surplus" in network.generators.index else np.zeros(n)
        result['pv_to_battery_mwh'] = pv_used.sum() * dt
        result['pv_to_battery_profile'] = pv_used
    
    return result


def optimize_peak_shaving_storage(load_profile, target_limit, efficiency, soc_min, soc_max,
                                   max_power_mw, max_capacity_mwh, pv_surplus=None,
                                   capex_power=80, capex_energy=250,
                                   interest_rate=0.05, lifetime_years=15, opex_rate=0.02):
    """
    Optimiert Speicherleistung UND -kapazität in einem PyPSA-Lauf.
    
    Findet die minimale Speichergröße, die das Peak-Shaving-Ziel erreicht.
    
    Args:
        load_profile: Lastprofil (MW)
        target_limit: Ziel-Lastgrenze (MW)
        max_power_mw: Maximale Speicherleistung (MW)
        max_capacity_mwh: Maximale Speicherkapazität (MWh)
        capex_power: Kosten pro kW Leistung (€/kW)
        capex_energy: Kosten pro kWh Kapazität (€/kWh)
        interest_rate: Zinssatz für Annuität (z.B. 0.05 = 5%)
        lifetime_years: Lebensdauer für Annuität (Jahre)
        opex_rate: Betriebskosten als Anteil von CAPEX (z.B. 0.02 = 2%)
    
    Returns:
        dict mit optimaler Konfiguration und Ergebnissen
    """
    n = len(load_profile)
    dt = 0.25
    
    eta_charge = np.sqrt(efficiency)
    eta_discharge = np.sqrt(efficiency)
    
    has_pv = pv_surplus is not None and len(pv_surplus) == n
    
    # === Annuität berechnen (KORREKT!) ===
    annuity_factor = calculate_annuity_factor(interest_rate, lifetime_years)
    
    # Jährliche Kosten pro Einheit (CAPEX annualisiert + OPEX)
    annual_capex_energy = capex_energy * 1000 * annuity_factor  # €/MWh/Jahr
    annual_opex_energy = capex_energy * 1000 * opex_rate        # €/MWh/Jahr
    annual_cost_energy = annual_capex_energy + annual_opex_energy
    
    annual_capex_power = capex_power * 1000 * annuity_factor    # €/MW/Jahr
    annual_opex_power = capex_power * 1000 * opex_rate          # €/MW/Jahr
    annual_cost_power = annual_capex_power + annual_opex_power
    
    # PyPSA Netzwerk erstellen
    network = pypsa.Network()
    network.set_snapshots(range(n))
    
    network.snapshot_weightings.loc[:, "generators"] = dt
    network.snapshot_weightings.loc[:, "stores"] = dt
    network.snapshot_weightings.loc[:, "links"] = dt
    
    # Bus
    network.add("Bus", "load_bus")
    network.add("Bus", "battery_bus")
    
    # Last
    network.add("Load", "demand",
                bus="load_bus",
                p_set=load_profile.values)
    
    # Netzbezug - Grundlast bis target_limit günstig
    network.add("Generator", "grid_base",
                bus="load_bus",
                p_nom=target_limit,
                marginal_cost=1)
    
    # Spitzenlast sehr teuer (zu vermeiden)
    network.add("Generator", "grid_peak",
                bus="load_bus",
                p_nom=load_profile.max() * 2,
                marginal_cost=10000)  # Sehr hohe Kosten = starker Anreiz zur Vermeidung
    
    # Batteriespeicher mit ERWEITERBARER Kapazität
    network.add("Store", "battery",
                bus="battery_bus",
                e_nom=0,  # Startwert
                e_nom_extendable=True,
                e_nom_min=0,
                e_nom_max=max_capacity_mwh * (soc_max - soc_min),
                e_cyclic=False,
                e_initial_per_period=0.5,  # Start bei 50%
                standing_loss=0,
                capital_cost=annual_cost_energy)  # Korrekte Annuität!
    
    # Laden mit ERWEITERBARER Leistung
    network.add("Link", "charge",
                bus0="load_bus",
                bus1="battery_bus",
                p_nom=0,
                p_nom_extendable=True,
                p_nom_min=0,
                p_nom_max=max_power_mw,
                efficiency=eta_charge,
                marginal_cost=0.1,
                capital_cost=annual_cost_power / 2)  # Korrekte Annuität!
    
    # Entladen mit ERWEITERBARER Leistung
    network.add("Link", "discharge",
                bus0="battery_bus",
                bus1="load_bus",
                p_nom=0,
                p_nom_extendable=True,
                p_nom_min=0,
                p_nom_max=max_power_mw,
                efficiency=eta_discharge,
                marginal_cost=0,
                capital_cost=annual_cost_power / 2)  # Korrekte Annuität!
    
    # PV-Überschuss (falls vorhanden)
    if has_pv:
        network.add("Generator", "pv_surplus",
                    bus="load_bus",
                    p_nom=pv_surplus.max() if pv_surplus.max() > 0 else 1,
                    p_max_pu=np.clip(pv_surplus.values / max(pv_surplus.max(), 0.001), 0, 1),
                    marginal_cost=-0.01)
    
    # Optimieren
    try:
        status = network.optimize(solver_name="highs", solver_options={"threads": 1})
        
        if status[0] != "ok":
            return None
    except Exception as e:
        return None
    
    # Optimale Größen extrahieren
    optimal_capacity = network.stores.loc["battery", "e_nom_opt"]
    optimal_power_charge = network.links.loc["charge", "p_nom_opt"]
    optimal_power_discharge = network.links.loc["discharge", "p_nom_opt"]
    optimal_power = max(optimal_power_charge, optimal_power_discharge)
    
    # Ergebnisse extrahieren
    grid_base = network.generators_t.p["grid_base"].values
    grid_peak = network.generators_t.p["grid_peak"].values
    grid_power = grid_base + grid_peak
    
    charge_power = network.links_t.p0["charge"].values
    discharge_power = network.links_t.p1["discharge"].values
    battery_power = discharge_power - charge_power
    
    soc = network.stores_t.e["battery"].values
    
    # Kapazität zurückrechnen auf Gesamt (inkl. SOC-Grenzen)
    total_capacity = optimal_capacity / (soc_max - soc_min) if (soc_max - soc_min) > 0 else optimal_capacity
    soc_absolute = soc + soc_min * total_capacity
    
    original_peak = load_profile.max()
    new_peak = grid_power.max()
    
    result = {
        'optimal_capacity_mwh': total_capacity,
        'optimal_power_mw': optimal_power,
        'original_peak_mw': original_peak,
        'new_peak_mw': new_peak,
        'peak_reduction_mw': original_peak - new_peak,
        'target_achieved': new_peak <= target_limit * 1.01,
        'unmet_peaks_count': (grid_peak > 0.01).sum(),
        'grid_power_profile': grid_power,
        'battery_power_profile': battery_power,
        'soc_profile': soc_absolute,
        'optimization_status': 'optimal',
        'objective_value': network.objective,
        'ep_ratio': total_capacity / optimal_power if optimal_power > 0 else 0,
        # Wirtschaftliche Parameter
        'annuity_factor': annuity_factor,
        'annual_cost_energy': annual_cost_energy,
        'annual_cost_power': annual_cost_power,
        'annual_capex_energy': annual_capex_energy,
        'annual_opex_energy': annual_opex_energy,
    }
    
    if has_pv:
        pv_used = network.generators_t.p["pv_surplus"].values if "pv_surplus" in network.generators.index else np.zeros(n)
        result['pv_to_battery_mwh'] = pv_used.sum() * dt
        result['pv_to_battery_profile'] = pv_used
    
    return result


# =============================================================================
# Seiten-Funktionen
# =============================================================================
def show_start_page():
    """Zeigt die Startseite mit Modus-Auswahl."""
    
    st.markdown('<p class="main-header">🔋 Batteriespeicher-Optimierungstool</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Bestimmung optimaler Betriebsstrategien und Dimensionierung von Großbatteriespeichern</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Bitte wählen Sie einen Modus:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Modus A: Wirtschaftlichkeitsanalyse</h3>
            <p><strong>Für Großspeicher mit Netzdienstleistungen</strong></p>
            <ul>
                <li>CAPEX und OPEX eingeben</li>
                <li>Erlösschätzung (Benchmark oder eigene Daten)</li>
                <li>NPV, IRR, Amortisation berechnen</li>
                <li>Sensitivitätsanalyse durchführen</li>
            </ul>
            <p><em>Hinweis: Operative Optimierung erfolgt durch Direktvermarkter</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Modus A starten", key="mode_a", use_container_width=True):
            st.session_state.current_mode = 'A'
            st.session_state.current_step = 1
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🏭 Modus B: Peak Shaving</h3>
            <p><strong>Für Industriekunden mit hohen Lastspitzen</strong></p>
            <ul>
                <li>Analyse des Lastprofils</li>
                <li>Optional: Eigene PV-Erzeugung einbinden</li>
                <li>Berechnung der erforderlichen Speichergröße</li>
                <li>Einsparung durch vermiedene Leistungskosten</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏭 Modus B starten", key="mode_b", use_container_width=True):
            st.session_state.current_mode = 'B'
            st.session_state.current_step = 1
            st.rerun()
    
    st.markdown("---")
    
    # Modus C in voller Breite
    st.markdown("""
    <div class="info-box">
        <h3>⚡ Modus C: NVP-Überbauung</h3>
        <p><strong>Für die gemeinsame Nutzung von Netzverknüpfungspunkten</strong></p>
        <p>Basierend auf der BEE-Studie zur NVP-Überbauung: Dimensionierung von Batteriespeichern 
        bei installierter Erzeugungsleistung größer als die Netzanschlusskapazität.</p>
        <ul>
            <li>Kombination von Wind- und PV-Profilen (normiert oder absolut)</li>
            <li>Berechnung des Überbauungsfaktors und der EE-Überschüsse</li>
            <li>Parameterstudie für optimale Speicherleistung und -kapazität</li>
            <li>Verbesserung der Netzauslastung und Erfassungsgrad</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("⚡ Modus C starten", key="mode_c", use_container_width=True):
        st.session_state.current_mode = 'C'
        st.session_state.current_step = 1
        st.rerun()
    
    st.markdown("---")
    
    # Hinweise für Laien
    with st.expander("ℹ️ Hilfe: Welchen Modus soll ich wählen?"):
        st.markdown("""
        **Modus A - Wirtschaftlichkeitsanalyse** ist geeignet, wenn Sie:
        - Die Wirtschaftlichkeit eines Großspeichers bewerten möchten
        - CAPEX, Erlöse und Amortisation berechnen wollen
        - Sensitivitätsanalysen durchführen möchten
        - *Hinweis: Die operative Optimierung erfolgt durch den Direktvermarkter*
        
        **Modus B - Peak Shaving** ist geeignet, wenn Sie:
        - Ein Industrieunternehmen mit schwankendem Strombedarf sind
        - Hohe Leistungsspitzen haben, die teuer sind
        - Optional: Eine eigene PV-Anlage (z.B. Dach-PV) haben
        
        **Modus C - NVP-Überbauung** ist geeignet, wenn Sie:
        - Einen bestehenden Netzverknüpfungspunkt mit weiteren EE-Anlagen erweitern möchten
        - Die installierte Erzeugungsleistung die Anschlusskapazität übersteigt (Überbauung)
        - Den entstehenden Energieüberschuss durch einen Speicher nutzen möchten
        
        **Benötigte Daten:**
        - CSV-Dateien mit Zeitreihen (15-Minuten-Auflösung empfohlen)
        - Spalte 1: Zeitstempel
        - Spalte 2: Leistungswerte in MW oder kW
        """)


def show_mode_a():
    """Zeigt den Ablauf für Modus A: Wirtschaftlichkeitsanalyse Großspeicher."""
    
    # Header mit Zurück-Button
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Zurück"):
            st.session_state.current_mode = None
            st.rerun()
    with col_title:
        st.markdown("## 📊 Modus A: Wirtschaftlichkeitsanalyse Großspeicher")
    
    # Fortschrittsanzeige
    steps = ["Konfiguration", "Erlösschätzung", "Wirtschaftlichkeit"]
    current = st.session_state.current_step
    
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current:
                st.success(f"✓ {step}")
            elif i + 1 == current:
                st.info(f"→ {step}")
            else:
                st.text(f"○ {step}")
    
    st.markdown("---")
    
    # Schritt 1: Speichergröße und CAPEX
    if current == 1:
        show_mode_a_step1()
    elif current == 2:
        show_mode_a_step2()
    elif current == 3:
        show_mode_a_step3()


def show_mode_a_step1():
    """Modus A - Schritt 1: Speicherkonfiguration und Investitionsparameter."""
    
    st.markdown("### Schritt 1: Speicherkonfiguration")
    
    st.markdown("""
    <div class="info-box">
        <strong>📊 Wirtschaftlichkeitsanalyse Großspeicher</strong><br>
        Dieser Modus berechnet die Wirtschaftlichkeit eines Batteriegroßspeichers 
        basierend auf historischen Marktdaten. Die operative Optimierung (Aufteilung 
        auf FCR, aFRR, Arbitrage) erfolgt in der Praxis durch den Direktvermarkter.
    </div>
    """, unsafe_allow_html=True)
    
    # Speicherkonfiguration
    st.markdown("#### 🔋 Speicherkonfiguration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unit_a = st.radio(
            "Einheit:",
            ["MW / MWh", "kW / kWh"],
            horizontal=True,
            key="unit_a_radio"
        )
        is_mw = unit_a == "MW / MWh"
        st.session_state.unit_a_is_mw = is_mw
    
    with col2:
        storage_power = st.number_input(
            f"Speicherleistung ({'MW' if is_mw else 'kW'})",
            min_value=0.1 if is_mw else 1.0,
            max_value=10000.0,
            value=10.0 if is_mw else 1000.0,
            step=1.0,
            key="storage_power_a_input"
        )
        st.session_state.storage_power_a_mw = storage_power if is_mw else storage_power / 1000
    
    with col3:
        storage_capacity = st.number_input(
            f"Speicherkapazität ({'MWh' if is_mw else 'kWh'})",
            min_value=0.1 if is_mw else 1.0,
            max_value=100000.0,
            value=20.0 if is_mw else 2000.0,
            step=1.0,
            key="storage_capacity_a_input"
        )
        st.session_state.storage_capacity_a_mwh = storage_capacity if is_mw else storage_capacity / 1000
    
    # E/P-Verhältnis anzeigen
    if storage_power > 0:
        ep_ratio = storage_capacity / storage_power
        st.info(f"📐 E/P-Verhältnis: {ep_ratio:.1f} h (Speicherdauer bei Volllast)")
    
    st.markdown("---")
    
    # Technische Parameter
    st.markdown("#### ⚙️ Technische Parameter")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        efficiency = st.slider("Roundtrip-Wirkungsgrad (%)", 80, 98, 90, key="eff_a") / 100
        # Wert unter anderem Key speichern, da "eff_a" bereits Widget-Key ist
        st.session_state.efficiency_a_val = efficiency
    with col2:
        soc_min = st.slider("Minimaler SOC (%)", 0, 30, 10, key="soc_min_a") / 100
        st.session_state.soc_min_a_val = soc_min
    with col3:
        soc_max = st.slider("Maximaler SOC (%)", 70, 100, 90, key="soc_max_a") / 100
        st.session_state.soc_max_a_val = soc_max
    with col4:
        calendar_life = st.number_input("Kalendarische Lebensdauer (Jahre)", 10, 25, 15, key="cal_life_a")
        st.session_state.calendar_life_a_val = calendar_life
    
    st.markdown("---")
    
    # Investitionsparameter
    st.markdown("#### 💰 Investitionsparameter (CAPEX)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        capex_energy = st.number_input(
            "CAPEX Energie (€/kWh)",
            min_value=50,
            max_value=1000,
            value=250,
            step=10,
            help="Investitionskosten pro kWh Speicherkapazität"
        )
        st.session_state.capex_energy_a = capex_energy
    
    with col2:
        capex_power = st.number_input(
            "CAPEX Leistung (€/kW)",
            min_value=0,
            max_value=500,
            value=80,
            step=10,
            help="Investitionskosten pro kW Speicherleistung (Wechselrichter, etc.)"
        )
        st.session_state.capex_power_a = capex_power
    
    with col3:
        opex_rate = st.number_input(
            "OPEX (% von CAPEX/Jahr)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Jährliche Betriebskosten als Prozentsatz der Investition"
        )
        st.session_state.opex_rate_a = opex_rate
    
    # CAPEX berechnen und anzeigen
    total_capex = (storage_capacity * 1000 * capex_energy + 
                   storage_power * 1000 * capex_power) if is_mw else (
                   storage_capacity * capex_energy + storage_power * capex_power)
    
    annual_opex = total_capex * opex_rate / 100
    
    st.markdown("##### 💶 Investitionsübersicht")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gesamt-CAPEX", f"{total_capex:,.0f} €")
    with col2:
        st.metric("CAPEX pro kWh", f"{total_capex / (storage_capacity * 1000 if is_mw else storage_capacity):,.0f} €/kWh")
    with col3:
        st.metric("Jährliche OPEX", f"{annual_opex:,.0f} €/Jahr")
    
    st.session_state.total_capex_a = total_capex
    st.session_state.annual_opex_a = annual_opex
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("Weiter →", type="primary", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()


def show_mode_a_step2():
    """Modus A - Schritt 2: Marktdaten und Erlösschätzung."""
    
    st.markdown("### Schritt 2: Erlösschätzung")
    
    # Daten abrufen
    storage_power = st.session_state.get('storage_power_a_mw', 10)
    storage_capacity = st.session_state.get('storage_capacity_a_mwh', 20)
    is_mw = st.session_state.get('unit_a_is_mw', True)
    efficiency = st.session_state.get('efficiency_a_val', 0.9)
    
    st.markdown("""
    <div class="info-box">
        <strong>📈 Erlösschätzung</strong><br>
        Die tatsächlichen Erlöse hängen von der Vermarktungsstrategie des Direktvermarkters ab.
        Diese Berechnung verwendet Benchmark-Werte aus Marktdaten zur Abschätzung des Erlöspotenzials.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methode zur Erlösschätzung wählen
    st.markdown("#### 📊 Methode zur Erlösschätzung")
    
    estimation_method = st.radio(
        "Wie sollen die Erlöse geschätzt werden?",
        ["Benchmark-Werte (Branchendurchschnitt)", "Eigene Marktdaten hochladen"],
        horizontal=True,
        key="estimation_method_a"
    )
    
    if estimation_method == "Benchmark-Werte (Branchendurchschnitt)":
        st.markdown("---")
        st.markdown("#### 📈 Benchmark-Erlöse (Marktdurchschnitt 2023/2024)")
        
        st.markdown("""
        <div class="info-box">
            <strong>Quelle:</strong> Durchschnittswerte aus Marktstudien und veröffentlichten Daten 
            von Batteriespeicher-Projekten in Deutschland.
        </div>
        """, unsafe_allow_html=True)
        
        # Benchmark-Werte (basierend auf Marktdaten)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Anpassbare Benchmark-Werte")
            
            # Spezifische Erlöse (€/MW/Jahr für Leistung, €/MWh/Jahr für Kapazität)
            revenue_fcr = st.number_input(
                "FCR-Erlös (€/MW/Jahr)",
                min_value=50000,
                max_value=250000,
                value=120000,
                step=5000,
                help="Typisch: 100.000 - 150.000 €/MW/Jahr (stark schwankend)"
            )
            
            revenue_afrr = st.number_input(
                "aFRR-Erlös (€/MW/Jahr)",
                min_value=20000,
                max_value=100000,
                value=50000,
                step=5000,
                help="Typisch: 40.000 - 70.000 €/MW/Jahr"
            )
            
            revenue_arbitrage = st.number_input(
                "Arbitrage-Erlös (€/MWh/Jahr)",
                min_value=5000,
                max_value=50000,
                value=15000,
                step=1000,
                help="Typisch: 10.000 - 25.000 €/MWh/Jahr (abhängig von Preisvolatilität)"
            )
        
        with col2:
            st.markdown("##### Marktanteile (typische Aufteilung)")
            
            st.markdown("""
            Der Direktvermarkter optimiert die Aufteilung dynamisch.
            Für die Wirtschaftlichkeitsrechnung verwenden wir typische Durchschnittswerte:
            """)
            
            share_fcr = st.slider("FCR-Anteil (%)", 0, 100, 50, key="share_fcr_a") / 100
            share_afrr = st.slider("aFRR-Anteil (%)", 0, 100 - int(share_fcr*100), 20, key="share_afrr_a") / 100
            share_arbitrage = 1 - share_fcr - share_afrr
            
            st.info(f"Verbleibender Arbitrage-Anteil: {share_arbitrage*100:.0f}%")
        
        # Erlöse berechnen
        annual_revenue_fcr = revenue_fcr * storage_power * share_fcr
        annual_revenue_afrr = revenue_afrr * storage_power * share_afrr
        annual_revenue_arb = revenue_arbitrage * storage_capacity * share_arbitrage
        total_annual_revenue = annual_revenue_fcr + annual_revenue_afrr + annual_revenue_arb
        
        # Direktvermarkter-Provision abziehen
        st.markdown("---")
        st.markdown("##### 💼 Direktvermarkter-Konditionen")
        
        col1, col2 = st.columns(2)
        with col1:
            marketer_fee = st.number_input(
                "Provision Direktvermarkter (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Typisch: 10-20% der Erlöse"
            )
        
        net_revenue = total_annual_revenue * (1 - marketer_fee / 100)
        
        with col2:
            st.metric("Provision", f"{total_annual_revenue * marketer_fee / 100:,.0f} €/Jahr")
        
        st.session_state.annual_revenue_a = net_revenue
        st.session_state.revenue_details_a = {
            'fcr': annual_revenue_fcr,
            'afrr': annual_revenue_afrr,
            'arbitrage': annual_revenue_arb,
            'total_brutto': total_annual_revenue,
            'marketer_fee': marketer_fee,
            'total_netto': net_revenue
        }
    
    else:
        # Eigene Marktdaten hochladen
        st.markdown("---")
        st.markdown("#### 📁 Marktdaten hochladen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 💶 Day-Ahead-Preise")
            price_file = st.file_uploader(
                "Day-Ahead-Preise hochladen",
                type=['csv', 'xlsx', 'xls'],
                key="price_upload_a",
                help="Zeitreihe mit Strompreisen in €/MWh"
            )
            
            if price_file is not None:
                df = load_csv_file(price_file)
                if df is not None:
                    st.session_state.price_profile_a = df.iloc[:, 0]
                    prices = df.iloc[:, 0]
                    st.success(f"✓ {len(df)} Preisdaten geladen")
                    
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Mittelwert", f"{prices.mean():.1f} €/MWh")
                    with col_s2:
                        st.metric("Spread (P90-P10)", f"{np.percentile(prices, 90) - np.percentile(prices, 10):.1f} €/MWh")
                    with col_s3:
                        st.metric("Volatilität", f"{prices.std():.1f} €/MWh")
        
        with col2:
            st.markdown("##### ⚡ FCR-Preise (optional)")
            fcr_file = st.file_uploader(
                "FCR-Preise hochladen",
                type=['csv', 'xlsx', 'xls'],
                key="fcr_upload_a",
                help="Primärregelleistungspreise in €/MW/h"
            )
            
            if fcr_file is not None:
                df = load_csv_file(fcr_file)
                if df is not None:
                    st.session_state.fcr_prices_a = df.iloc[:, 0]
                    st.success(f"✓ FCR-Preise geladen (Mittel: {df.iloc[:, 0].mean():.1f} €/MW/h)")
            else:
                fcr_mean = st.number_input("Oder FCR-Durchschnittspreis (€/MW/h)", 5.0, 50.0, 15.0, key="fcr_default_a")
        
        # Erlöse aus hochgeladenen Daten berechnen
        if st.session_state.get('price_profile_a') is not None:
            prices = st.session_state.price_profile_a
            
            # Arbitrage-Erlös schätzen
            price_spread = np.percentile(prices, 90) - np.percentile(prices, 10)
            usable_capacity = storage_capacity * (st.session_state.get('soc_max_a_val', 0.9) - 
                                                   st.session_state.get('soc_min_a_val', 0.1))
            daily_cycles = min(2, storage_power / usable_capacity)
            annual_cycles = daily_cycles * 365
            annual_revenue_arb = price_spread * annual_cycles * usable_capacity * efficiency
            
            # FCR-Erlös
            if st.session_state.get('fcr_prices_a') is not None:
                fcr_mean = st.session_state.fcr_prices_a.mean()
            else:
                fcr_mean = st.session_state.get('fcr_default_a', 15)
            
            hours_per_year = len(prices) / 4 if len(prices) > 8760 else 8760
            annual_revenue_fcr = fcr_mean * storage_power * hours_per_year * 0.5  # 50% FCR-Anteil
            
            total_annual_revenue = annual_revenue_fcr + annual_revenue_arb
            
            # Provision
            marketer_fee = st.number_input("Provision Direktvermarkter (%)", 5, 30, 15, key="marketer_fee_a")
            net_revenue = total_annual_revenue * (1 - marketer_fee / 100)
            
            st.session_state.annual_revenue_a = net_revenue
            st.session_state.revenue_details_a = {
                'fcr': annual_revenue_fcr,
                'afrr': 0,
                'arbitrage': annual_revenue_arb,
                'total_brutto': total_annual_revenue,
                'marketer_fee': marketer_fee,
                'total_netto': net_revenue
            }
    
    # Erlösübersicht anzeigen
    if st.session_state.get('revenue_details_a'):
        st.markdown("---")
        st.markdown("#### 💰 Geschätzte Jahreserlöse")
        
        rev = st.session_state.revenue_details_a
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("FCR-Erlös", f"{rev['fcr']:,.0f} €")
        with col2:
            st.metric("aFRR-Erlös", f"{rev['afrr']:,.0f} €")
        with col3:
            st.metric("Arbitrage-Erlös", f"{rev['arbitrage']:,.0f} €")
        with col4:
            st.metric("**Netto-Erlös**", f"{rev['total_netto']:,.0f} €/Jahr", 
                     delta=f"-{rev['marketer_fee']}% Provision")
        
        # Spezifische Kennzahlen
        col1, col2 = st.columns(2)
        with col1:
            specific_revenue_mwh = rev['total_netto'] / (storage_capacity * 1000 if is_mw else storage_capacity)
            st.metric("Spezifischer Erlös", f"{specific_revenue_mwh:,.0f} €/kWh/Jahr")
        with col2:
            specific_revenue_mw = rev['total_netto'] / (storage_power * 1000 if is_mw else storage_power)
            st.metric("Spezifischer Erlös", f"{specific_revenue_mw:,.0f} €/kW/Jahr")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        if st.session_state.get('annual_revenue_a'):
            if st.button("Weiter →", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
        else:
            st.warning("⚠️ Bitte Erlöse schätzen lassen")


def show_mode_a_step3():
    """Modus A - Schritt 3: Wirtschaftlichkeitsanalyse und Ergebnisse."""
    
    st.markdown("### Schritt 3: Wirtschaftlichkeitsanalyse")
    
    # Daten abrufen
    storage_power = st.session_state.get('storage_power_a_mw', 10)
    storage_capacity = st.session_state.get('storage_capacity_a_mwh', 20)
    is_mw = st.session_state.get('unit_a_is_mw', True)
    
    total_capex = st.session_state.get('total_capex_a', 0)
    annual_opex = st.session_state.get('annual_opex_a', 0)
    annual_revenue = st.session_state.get('annual_revenue_a', 0)
    calendar_life = st.session_state.get('calendar_life_a_val', 15)
    
    unit_p = "MW" if is_mw else "kW"
    unit_e = "MWh" if is_mw else "kWh"
    
    # Finanzparameter
    st.markdown("#### 📊 Finanzparameter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        discount_rate = st.number_input(
            "Diskontierungszins (%)",
            min_value=2.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
            help="WACC oder gewünschte Mindestrendite"
        ) / 100
    
    with col2:
        revenue_degradation = st.number_input(
            "Erlösrückgang pro Jahr (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Erwarteter jährlicher Rückgang der Markterlöse"
        ) / 100
    
    with col3:
        capacity_degradation = st.number_input(
            "Kapazitätsverlust pro Jahr (%)",
            min_value=0.0,
            max_value=5.0,
            value=1.5,
            step=0.5,
            help="Batteriedegradation"
        ) / 100
    
    st.markdown("---")
    
    # Wirtschaftlichkeitsberechnung
    st.markdown("#### 💰 Wirtschaftlichkeitskennzahlen")
    
    # Cashflow-Berechnung
    years = list(range(calendar_life + 1))
    cashflows = [-total_capex]  # Jahr 0: Investition
    
    cumulative_cf = [-total_capex]
    annual_revenues = [0]
    annual_costs = [0]
    
    for year in range(1, calendar_life + 1):
        # Erlös mit Degradation
        year_revenue = annual_revenue * ((1 - revenue_degradation) ** (year - 1)) * ((1 - capacity_degradation) ** (year - 1))
        # Kosten
        year_cost = annual_opex
        # Netto-Cashflow
        year_cf = year_revenue - year_cost
        
        cashflows.append(year_cf)
        cumulative_cf.append(cumulative_cf[-1] + year_cf)
        annual_revenues.append(year_revenue)
        annual_costs.append(year_cost)
    
    # NPV berechnen
    npv = sum([cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cashflows)])
    
    # IRR berechnen (vereinfachte Newton-Methode)
    def calc_npv(rate, cfs):
        return sum([cf / ((1 + rate) ** i) for i, cf in enumerate(cfs)])
    
    irr = None
    try:
        # Bisection method für IRR
        low, high = -0.5, 1.0
        for _ in range(100):
            mid = (low + high) / 2
            if calc_npv(mid, cashflows) > 0:
                low = mid
            else:
                high = mid
        irr = mid
    except:
        irr = None
    
    # Amortisation berechnen
    payback = None
    for i, cum_cf in enumerate(cumulative_cf):
        if cum_cf >= 0:
            payback = i
            break
    
    # LCOE berechnen (Levelized Cost of Storage)
    total_energy_discharged = sum([storage_capacity * 1000 * 365 * 1.5 * ((1 - capacity_degradation) ** y) 
                                   for y in range(calendar_life)])  # ~1.5 Zyklen/Tag
    total_costs_discounted = sum([(annual_opex) / ((1 + discount_rate) ** y) 
                                  for y in range(1, calendar_life + 1)]) + total_capex
    lcos = total_costs_discounted / total_energy_discharged * 1000 if total_energy_discharged > 0 else 0
    
    # Ergebnisse anzeigen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        npv_color = "normal" if npv >= 0 else "inverse"
        st.metric("NPV (Kapitalwert)", f"{npv:,.0f} €", 
                 delta="Profitabel" if npv > 0 else "Nicht profitabel",
                 delta_color=npv_color)
    
    with col2:
        if irr:
            irr_delta = f"{(irr - discount_rate)*100:+.1f}% vs. WACC"
            st.metric("IRR (Interner Zinsfuß)", f"{irr*100:.1f}%", delta=irr_delta)
        else:
            st.metric("IRR", "n.a.")
    
    with col3:
        if payback:
            st.metric("Amortisation", f"{payback} Jahre",
                     delta="OK" if payback <= calendar_life * 0.6 else "Langsam",
                     delta_color="normal" if payback <= calendar_life * 0.6 else "inverse")
        else:
            st.metric("Amortisation", f">{calendar_life} Jahre", delta="Nicht erreicht", delta_color="inverse")
    
    with col4:
        st.metric("LCOS", f"{lcos:.1f} €/MWh", help="Levelized Cost of Storage")
    
    # Bewertung
    st.markdown("---")
    st.markdown("#### 📋 Bewertung")
    
    if npv > 0 and irr and irr > discount_rate and payback and payback <= calendar_life * 0.6:
        st.success("""
        ✅ **Investition empfohlen**
        
        Das Projekt zeigt eine positive Wirtschaftlichkeit mit akzeptabler Amortisationszeit.
        """)
    elif npv > 0:
        st.warning("""
        ⚠️ **Investition bedingt empfohlen**
        
        Das Projekt ist profitabel, aber die Amortisation ist relativ lang oder die Rendite knapp über dem Zielwert.
        """)
    else:
        st.error("""
        ❌ **Investition nicht empfohlen**
        
        Das Projekt erreicht unter den gegebenen Annahmen keine positive Wirtschaftlichkeit.
        """)
    
    st.markdown("---")
    
    # Cashflow-Diagramm
    st.markdown("#### 📈 Cashflow-Entwicklung")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Kumulierter Cashflow
    colors = ['#e74c3c' if cf < 0 else '#2ecc71' for cf in cumulative_cf]
    ax1.bar(years, [cf/1000000 for cf in cumulative_cf], color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    if payback:
        ax1.axvline(x=payback, color='#3498db', linestyle='--', label=f'Amortisation: {payback} Jahre')
        ax1.legend()
    ax1.set_xlabel('Jahr')
    ax1.set_ylabel('Kumulierter Cashflow (Mio. €)')
    ax1.set_title('Kumulierter Cashflow')
    ax1.grid(True, alpha=0.3)
    
    # Jährliche Erlöse und Kosten
    x = np.arange(1, calendar_life + 1)
    width = 0.35
    ax2.bar(x - width/2, [r/1000 for r in annual_revenues[1:]], width, label='Erlöse', color='#2ecc71', alpha=0.7)
    ax2.bar(x + width/2, [c/1000 for c in annual_costs[1:]], width, label='OPEX', color='#e74c3c', alpha=0.7)
    ax2.set_xlabel('Jahr')
    ax2.set_ylabel('Tausend €')
    ax2.set_title('Jährliche Erlöse und Kosten')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Sensitivitätsanalyse
    st.markdown("#### 🔍 Sensitivitätsanalyse")
    
    with st.expander("Sensitivität anzeigen"):
        
        # Parameter-Variation
        variations = [-20, -10, 0, 10, 20]
        
        sensitivity_data = {
            'Variation': [f"{v:+d}%" for v in variations],
            'CAPEX': [],
            'Erlöse': [],
            'OPEX': [],
        }
        
        for var in variations:
            # CAPEX-Sensitivität
            adj_capex = total_capex * (1 + var/100)
            adj_cfs = [-adj_capex] + cashflows[1:]
            sensitivity_data['CAPEX'].append(sum([cf / ((1 + discount_rate) ** i) for i, cf in enumerate(adj_cfs)]))
            
            # Erlös-Sensitivität
            adj_cfs = [cashflows[0]] + [cf * (1 + var/100) if cf > 0 else cf for cf in cashflows[1:]]
            sensitivity_data['Erlöse'].append(sum([cf / ((1 + discount_rate) ** i) for i, cf in enumerate(adj_cfs)]))
            
            # OPEX-Sensitivität
            adj_opex = annual_opex * (1 + var/100)
            adj_cfs = [cashflows[0]] + [annual_revenues[i] - adj_opex for i in range(1, len(annual_revenues))]
            sensitivity_data['OPEX'].append(sum([cf / ((1 + discount_rate) ** i) for i, cf in enumerate(adj_cfs)]))
        
        # Tornado-Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        
        y_pos = np.arange(3)
        
        # Berechne Spannen
        capex_range = max(sensitivity_data['CAPEX']) - min(sensitivity_data['CAPEX'])
        revenue_range = max(sensitivity_data['Erlöse']) - min(sensitivity_data['Erlöse'])
        opex_range = max(sensitivity_data['OPEX']) - min(sensitivity_data['OPEX'])
        
        ranges = [capex_range/1000000, revenue_range/1000000, opex_range/1000000]
        labels = ['CAPEX (±20%)', 'Erlöse (±20%)', 'OPEX (±20%)']
        
        ax.barh(y_pos, ranges, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('NPV-Spanne (Mio. €)')
        ax.set_title('Sensitivität des NPV auf Parameteränderungen')
        ax.grid(True, alpha=0.3, axis='x')
        
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        **Interpretation:** Je länger der Balken, desto sensitiver ist der NPV auf Änderungen dieses Parameters.
        Erlöse haben typischerweise den größten Einfluss, gefolgt von CAPEX.
        """)
    
    st.markdown("---")
    
    # Export
    st.markdown("#### 📥 Ergebnisse exportieren")
    
    export_data = {
        'Speicherkonfiguration': {
            'Leistung_MW': storage_power,
            'Kapazität_MWh': storage_capacity,
            'EP_Verhältnis_h': storage_capacity / storage_power,
        },
        'Investition': {
            'CAPEX_EUR': total_capex,
            'OPEX_EUR_pro_Jahr': annual_opex,
        },
        'Erlöse': st.session_state.get('revenue_details_a', {}),
        'Wirtschaftlichkeit': {
            'NPV_EUR': npv,
            'IRR_Prozent': irr * 100 if irr else None,
            'Amortisation_Jahre': payback,
            'LCOS_EUR_MWh': lcos,
            'Projektlaufzeit_Jahre': calendar_life,
            'Diskontierungszins_Prozent': discount_rate * 100,
        },
        'Empfehlung': 'Investition empfohlen' if npv > 0 and payback and payback <= calendar_life * 0.6 else 
                      'Bedingt empfohlen' if npv > 0 else 'Nicht empfohlen'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Als JSON herunterladen",
            data=json_str,
            file_name="wirtschaftlichkeitsanalyse_speicher.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col3:
        if st.button("🏠 Zur Startseite", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def show_mode_b():
    """Zeigt den Ablauf für Modus B: Peak Shaving."""
    
    # Header mit Zurück-Button
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Zurück"):
            st.session_state.current_mode = None
            st.rerun()
    with col_title:
        st.markdown("## 🏭 Modus B: Peak Shaving für Industriekunden")
    
    # Fortschrittsanzeige
    steps = ["Lastprofil", "Analyse", "Parameter", "Berechnung", "Ergebnisse"]
    current = st.session_state.current_step
    
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current:
                st.success(f"✓ {step}")
            elif i + 1 == current:
                st.info(f"→ {step}")
            else:
                st.text(f"○ {step}")
    
    st.markdown("---")
    
    if current == 1:
        show_mode_b_step1()
    elif current == 2:
        show_mode_b_step2()
    elif current == 3:
        show_mode_b_step3()
    elif current == 4:
        show_mode_b_step4()
    elif current == 5:
        show_mode_b_step5()


def show_mode_b_step1():
    """Modus B - Schritt 1: Lastprofil und optionales PV-Profil hochladen."""
    
    st.markdown("### Schritt 1: Profile hochladen")
    
    # Einheitenauswahl ganz oben
    st.markdown("#### ⚙️ Grundeinstellungen")
    
    col_unit, col_pv = st.columns(2)
    
    with col_unit:
        # Widget-Wert wird automatisch unter key gespeichert
        unit_b = st.radio(
            "Einheit für Leistungen:",
            ["kW", "MW"],
            horizontal=True,
            key="unit_b_radio"
        )
    
    with col_pv:
        # Widget-Wert wird automatisch unter key gespeichert
        has_pv = st.checkbox(
            "Eigene PV-Anlage vorhanden (z.B. Dach-PV)",
            key="has_pv_b",
            help="Aktivieren, wenn das Unternehmen eine eigene Photovoltaikanlage hat"
        )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <strong>📁 Benötigte Dateien:</strong>
        <ul>
            <li><strong>Lastprofil</strong> (Pflicht): Stromverbrauch des Unternehmens</li>
            <li><strong>PV-Profil</strong> (optional): Erzeugung der eigenen PV-Anlage</li>
        </ul>
        <p>Die PV-Erzeugung reduziert den Netzbezug. Überschüsse werden gespeichert.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Lastprofil (Pflicht)")
        load_file = st.file_uploader(
            "Lastprofil hochladen",
            type=['csv', 'xlsx', 'xls'],
            key="load_upload_b",
            help="CSV oder Excel mit Zeitstempel und Leistungswerten"
        )
        
        if load_file is not None:
            df = load_csv_file(load_file)
            if df is not None:
                values = df.iloc[:, 0].values
                if unit_b == "kW":
                    values = values / 1000  # Umrechnung in MW
                
                st.session_state.load_profile = pd.Series(values, index=df.index)
                st.success(f"✓ {len(df)} Datenpunkte geladen")
                
                with st.expander("Lastprofil-Vorschau"):
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(range(min(2000, len(df))), 
                           st.session_state.load_profile.values[:min(2000, len(df))],
                           linewidth=0.5, color='#e74c3c')
                    ax.set_ylabel('Last (MW)')
                    ax.set_title('Lastprofil (Ausschnitt)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
    
    with col2:
        if has_pv:
            st.markdown("#### ☀️ PV-Profil")
            
            # Option für normierte Profile
            pv_type = st.radio(
                "Art des PV-Profils:",
                ["Normiert (0-1)", "Absolute Werte"],
                horizontal=True,
                key="pv_type_b"
            )
            is_normalized_pv = pv_type == "Normiert (0-1)"
            
            if is_normalized_pv:
                p_pv_inst = st.number_input(
                    f"Installierte PV-Leistung ({unit_b})",
                    min_value=0.1,
                    max_value=100000.0,
                    value=100.0 if unit_b == "kW" else 0.1,
                    step=10.0 if unit_b == "kW" else 0.01,
                    key="p_pv_inst_b_input"
                )
                # In MW umrechnen für spätere Verwendung
                p_pv_inst_mw = p_pv_inst if unit_b == "MW" else p_pv_inst / 1000
            else:
                p_pv_inst_mw = None
            
            # Einspeisebegrenzung
            st.markdown("---")
            st.markdown("##### ⚡ Einspeisebegrenzung")
            feed_in_mode = st.radio(
                "PV-Einspeisung ins Netz:",
                ["Unbegrenzt", "Nulleinspeisung", "Begrenzte Einspeisung"],
                horizontal=True,
                key="feed_in_mode_b",
                help="Nulleinspeisung: PV nur für Eigenverbrauch. Begrenzt: Max. Einspeisung festlegen."
            )
            
            max_feed_in_mw = None
            if feed_in_mode == "Begrenzte Einspeisung":
                max_feed_in = st.number_input(
                    f"Max. Einspeisung ({unit_b})",
                    min_value=0.0,
                    max_value=100000.0,
                    value=100.0 if unit_b == "kW" else 0.1,
                    step=10.0 if unit_b == "kW" else 0.01,
                    key="max_feed_in_b"
                )
                max_feed_in_mw = max_feed_in if unit_b == "MW" else max_feed_in / 1000
            elif feed_in_mode == "Nulleinspeisung":
                max_feed_in_mw = 0.0
            
            st.markdown("---")
            
            pv_file = st.file_uploader(
                "PV-Profil hochladen",
                type=['csv', 'xlsx', 'xls'],
                key="pv_upload_b",
                help="Zeitreihe mit PV-Erzeugung"
            )
            
            if pv_file is not None:
                df = load_csv_file(pv_file)
                if df is not None:
                    values = df.iloc[:, 0].values
                    
                    if is_normalized_pv and p_pv_inst_mw is not None:
                        values = values * p_pv_inst_mw
                        st.success(f"✓ {len(df)} Datenpunkte geladen (skaliert mit {p_pv_inst_mw:.3f} MW)")
                    else:
                        if unit_b == "kW":
                            values = values / 1000
                        st.success(f"✓ {len(df)} Datenpunkte geladen")
                    
                    st.session_state.pv_profile_b = pd.Series(values, index=df.index)
                    # feed_in_mode wird automatisch unter key="feed_in_mode_b" gespeichert
                    # max_feed_in_mw separat speichern (kein Widget-Key-Konflikt)
                    st.session_state.max_feed_in_mw_value = max_feed_in_mw
                    
                    with st.expander("PV-Vorschau"):
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.fill_between(range(min(2000, len(df))), 
                                       st.session_state.pv_profile_b.values[:min(2000, len(df))],
                                       alpha=0.5, color='#f1c40f')
                        ax.set_ylabel('PV (MW)')
                        ax.set_title('PV-Erzeugung (Ausschnitt)')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
        else:
            st.session_state.pv_profile_b = None
            st.info("Keine PV-Anlage aktiviert")
    
    # Nettolast berechnen und anzeigen, wenn beide Profile vorhanden
    if st.session_state.get('load_profile') is not None:
        load = st.session_state.load_profile
        pv = st.session_state.get('pv_profile_b')
        
        if pv is not None:
            # Prüfen ob Längen übereinstimmen
            if len(pv) != len(load):
                st.warning(f"⚠️ Unterschiedliche Profillängen: Last hat {len(load)} Datenpunkte, PV hat {len(pv)} Datenpunkte.")
                
                # Option zum Anpassen anbieten
                min_len = min(len(load), len(pv))
                if st.checkbox("Profile auf gleiche Länge kürzen", key="trim_profiles_b"):
                    load = load.iloc[:min_len]
                    pv = pv.iloc[:min_len]
                    st.session_state.load_profile = load
                    st.session_state.pv_profile_b = pv
                    st.info(f"✓ Beide Profile auf {min_len} Datenpunkte gekürzt")
                else:
                    st.info("Bitte aktivieren Sie die Option oder laden Sie Profile mit gleicher Länge hoch.")
                    st.session_state.net_load_b = load
                    st.session_state.pv_surplus_b = None
                    pv = None  # Verhindert weitere Berechnung
            
            if pv is not None and len(pv) == len(load):
                # Einspeisebegrenzung anwenden
                feed_in_mode = st.session_state.get('feed_in_mode_b', 'Unbegrenzt')
                max_feed_in = st.session_state.get('max_feed_in_mw_value', None)
                
                if feed_in_mode == "Nulleinspeisung":
                    # PV wird auf Last begrenzt (kein Überschuss ins Netz)
                    pv_used = np.minimum(pv.values, load.values)
                    pv_curtailed = pv.values - pv_used
                    net_load = load.values - pv_used
                    pv_surplus = np.zeros(len(load))  # Kein Überschuss für Speicher vom Netz
                    
                    st.info(f"🔌 **Nulleinspeisung aktiv:** PV wird auf Eigenverbrauch begrenzt. "
                           f"Abregelung: {pv_curtailed.sum() * 0.25 / 1000:.1f} MWh/Jahr")
                    
                elif feed_in_mode == "Begrenzte Einspeisung" and max_feed_in is not None:
                    # PV wird so begrenzt, dass Einspeisung max_feed_in nicht übersteigt
                    # Nettolast >= -max_feed_in
                    pv_used = np.minimum(pv.values, load.values + max_feed_in)
                    pv_curtailed = pv.values - pv_used
                    net_load = load.values - pv_used
                    pv_surplus = np.maximum(pv_used - load.values, 0)
                    
                    st.info(f"🔌 **Einspeisebegrenzung aktiv:** Max. {max_feed_in*1000:.0f} kW Einspeisung. "
                           f"Abregelung: {pv_curtailed.sum() * 0.25 / 1000:.1f} MWh/Jahr")
                else:
                    # Unbegrenzt - Original-Logik
                    net_load = load.values - pv.values
                    pv_surplus = np.maximum(pv.values - load.values, 0)
                    pv_curtailed = np.zeros(len(load))
                
                pv_self_consumption = np.minimum(load.values, pv.values - pv_curtailed if feed_in_mode != "Unbegrenzt" else pv.values)
                
                st.session_state.net_load_b = pd.Series(net_load, index=load.index)
                st.session_state.pv_surplus_b = pd.Series(pv_surplus, index=load.index)
                st.session_state.pv_curtailed_b = pv_curtailed.sum() * 0.25  # MWh
                
                st.markdown("---")
                st.markdown("#### 📊 Zusammenfassung")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Max. Last (brutto)", f"{load.max()*1000:.0f} kW")
                with col2:
                    st.metric("Max. Netzbezug", f"{max(net_load.max(), 0)*1000:.0f} kW")
                with col3:
                    if feed_in_mode == "Unbegrenzt":
                        max_einspeisung = abs(min(net_load.min(), 0))
                        st.metric("Max. Einspeisung", f"{max_einspeisung*1000:.0f} kW")
                    else:
                        st.metric("PV-Abregelung", f"{pv_curtailed.sum() * 0.25:.0f} kWh")
                with col4:
                    total_pv_used = (pv.values - pv_curtailed).sum() * 0.25 if feed_in_mode != "Unbegrenzt" else pv.sum() * 0.25
                    self_cons_rate = pv_self_consumption.sum() / total_pv_used * 100 / 4 if total_pv_used > 0 else 0
                    st.metric("PV-Eigenverbrauch", f"{min(self_cons_rate, 100):.0f}%")
                
                # Visualisierung
                with st.expander("Last und PV-Erzeugung (Beispielwoche)"):
                    days = 7
                    end_idx = min(days * 96, len(load))
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    x = range(end_idx)
                    
                    ax.fill_between(x, load.values[:end_idx] * 1000, alpha=0.3, color='#e74c3c', label='Bruttolast')
                    if feed_in_mode != "Unbegrenzt":
                        pv_used_plot = pv.values[:end_idx] - pv_curtailed[:end_idx]
                        ax.fill_between(x, pv_used_plot * 1000, alpha=0.5, color='#f1c40f', label='PV genutzt')
                        ax.fill_between(x, pv.values[:end_idx] * 1000, pv_used_plot * 1000, 
                                       alpha=0.3, color='#e67e22', label='PV abgeregelt')
                    else:
                        ax.fill_between(x, pv.values[:end_idx] * 1000, alpha=0.5, color='#f1c40f', label='PV-Erzeugung')
                    ax.plot(x, net_load[:end_idx] * 1000, color='#2ecc71', linewidth=1, label='Nettolast')
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    
                    ax.set_xlabel('Zeitschritt (15 min)')
                    ax.set_ylabel('Leistung (kW)')
                    ax.set_title('Last und PV-Erzeugung')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
        else:
            st.session_state.net_load_b = load
            st.session_state.pv_surplus_b = None
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.session_state.get('load_profile') is not None:
            if st.button("Weiter →", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()


def show_mode_b_step2():
    """Modus B - Schritt 2: Lastanalyse (mit PV-Berücksichtigung)."""
    
    st.markdown("### Schritt 2: Lastanalyse")
    
    # Bruttolast und Nettolast
    load_brutto = st.session_state.load_profile
    has_pv = st.session_state.get('has_pv_b', False)
    pv_profile = st.session_state.get('pv_profile_b')
    
    if has_pv and pv_profile is not None:
        load = st.session_state.get('net_load_b', load_brutto)
        st.info("📊 Analyse basiert auf **Nettolast** (Bruttolast - PV-Erzeugung)")
    else:
        load = load_brutto
    
    # Spitzenlast ermitteln
    peak_load = load.max()
    peak_idx = load.idxmax()
    mean_load = load.mean()
    min_load = load.min()
    
    st.session_state.peak_load = peak_load
    
    st.markdown("""
    <div class="success-box">
        <h3>🔍 Analyseergebnis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if has_pv and pv_profile is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Max. Bruttolast", f"{load_brutto.max()*1000:.0f} kW")
        with col2:
            st.metric("Max. Nettolast", f"{peak_load*1000:.0f} kW", 
                     f"{(peak_load - load_brutto.max())*1000:.0f} kW")
        with col3:
            st.metric("Mittlere Last", f"{mean_load*1000:.0f} kW")
        with col4:
            st.metric("Minimale Last", f"{min_load*1000:.0f} kW")
        with col5:
            st.metric("Lastspitze am", str(peak_idx)[:16] if hasattr(peak_idx, '__str__') else "n/a")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Maximale Last", f"{peak_load*1000:.0f} kW")
        with col2:
            st.metric("Mittlere Last", f"{mean_load*1000:.0f} kW")
        with col3:
            st.metric("Minimale Last", f"{min_load*1000:.0f} kW")
        with col4:
            st.metric("Lastspitze am", str(peak_idx)[:16] if hasattr(peak_idx, '__str__') else "n/a")
    
    st.markdown("---")
    
    # Histogramm der Lastverteilung
    st.markdown("#### Lastverteilung")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramm
    ax1.hist(load * 1000, bins=50, color='#3498db', alpha=0.7, edgecolor='white', label='Nettolast' if has_pv else 'Last')
    if has_pv and pv_profile is not None:
        ax1.hist(load_brutto * 1000, bins=50, color='#e74c3c', alpha=0.3, edgecolor='white', label='Bruttolast')
    ax1.axvline(x=peak_load * 1000, color='red', linestyle='--', label=f'Maximum: {peak_load*1000:.0f} kW')
    ax1.axvline(x=mean_load * 1000, color='green', linestyle='--', label=f'Mittelwert: {mean_load*1000:.0f} kW')
    ax1.set_xlabel('Leistung (kW)')
    ax1.set_ylabel('Häufigkeit')
    ax1.set_title('Verteilung der Lastwerte')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dauerlinie
    sorted_load = np.sort(load.values)[::-1]
    hours = np.arange(len(sorted_load)) * 0.25  # 15-min zu Stunden
    ax2.fill_between(hours, sorted_load * 1000, alpha=0.5, color='#3498db')
    ax2.plot(hours, sorted_load * 1000, color='#2980b9', label='Nettolast' if has_pv else 'Last')
    if has_pv and pv_profile is not None:
        sorted_brutto = np.sort(load_brutto.values)[::-1]
        ax2.plot(hours, sorted_brutto * 1000, color='#e74c3c', alpha=0.5, linestyle='--', label='Bruttolast')
    ax2.axhline(y=peak_load * 1000, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Stunden pro Jahr')
    ax2.set_ylabel('Leistung (kW)')
    ax2.set_title('Jahresdauerlinie')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        if st.button("Weiter →", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()


def show_mode_b_step3():
    """Modus B - Schritt 3: Parameter eingeben."""
    
    st.markdown("### Schritt 3: Zielwerte und Kosten festlegen")
    
    peak_load = st.session_state.peak_load
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Ziel-Leistungsgrenze")
        
        st.info(f"Aktuelle Spitzenlast: {peak_load*1000:.0f} kW ({peak_load:.2f} MW)")
        
        target_limit = st.number_input(
            "Maximaler Netzbezug (kW)",
            min_value=float(st.session_state.load_profile.mean() * 1000),
            max_value=float(peak_load * 1000),
            value=float(peak_load * 0.8 * 1000),
            step=10.0,
            help="Auf welche Leistung soll der Bezug begrenzt werden?"
        ) / 1000  # Umrechnung in MW
        
        st.session_state.target_limit = target_limit
        
        reduction = (peak_load - target_limit) * 1000
        reduction_pct = (peak_load - target_limit) / peak_load * 100
        
        st.success(f"Angestrebte Reduktion: {reduction:.0f} kW ({reduction_pct:.1f}%)")
    
    with col2:
        st.markdown("#### 💰 Stromkosten")
        
        energy_price = st.number_input(
            "Strombezugspreis (€/MWh)",
            min_value=0.0,
            max_value=500.0,
            value=150.0,
            step=5.0,
            help="Arbeitspreis für den Strombezug"
        )
        
        power_price = st.number_input(
            "Leistungspreis (€/kW/Jahr)",
            min_value=0.0,
            max_value=500.0,
            value=120.0,
            step=5.0,
            help="Jährlicher Preis für die bezogene Spitzenleistung"
        )
        
        st.session_state.energy_price = energy_price
        st.session_state.power_price = power_price
        
        # Aktuelle Kosten berechnen
        annual_peak_cost = peak_load * 1000 * power_price
        st.info(f"Aktuelle jährliche Leistungskosten: {annual_peak_cost:,.0f} €")
    
    st.markdown("---")
    
    st.markdown("#### 🔋 Speicherparameter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        efficiency = st.slider("Wirkungsgrad (%)", 70, 98, 88) / 100
    with col2:
        soc_min = st.number_input("Min. SoC (%)", 0, 30, 10) / 100
    with col3:
        soc_max = st.number_input("Max. SoC (%)", 70, 100, 90) / 100
    
    st.session_state.technical_params = {
        'efficiency': efficiency,
        'soc_min': soc_min,
        'soc_max': soc_max,
        'cycle_life': 6000,
        'calendar_life': 15,
    }
    
    st.markdown("---")
    
    st.markdown("#### 💶 Investitionsparameter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        capex_energy = st.number_input("CAPEX Energie (€/kWh)", 50, 1000, 250, 10)
    with col2:
        capex_power = st.number_input("CAPEX Leistung (€/kW)", 0, 500, 80, 10)
    with col3:
        project_lifetime = st.number_input("Projektlaufzeit (Jahre)", 5, 25, 15)
    
    st.session_state.economic_params = {
        'capex_energy': capex_energy,
        'capex_power': capex_power,
        'opex_rate': 2.0,
        'discount_rate': 6.0,
        'project_lifetime': project_lifetime,
    }
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col3:
        if st.button("Weiter →", type="primary", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()


def show_mode_b_step4():
    """Modus B - Schritt 4: Speicherberechnung."""
    
    st.markdown("### Schritt 4: Speicherdimensionierung für Peak Shaving")
    
    # Nettolast verwenden wenn PV vorhanden
    has_pv = st.session_state.get('has_pv_b', False)
    pv_surplus = st.session_state.get('pv_surplus_b')
    
    if has_pv and st.session_state.get('net_load_b') is not None:
        load = st.session_state.net_load_b
        st.info("📊 Berechnung basiert auf Nettolast (nach Abzug der PV-Erzeugung)")
    else:
        load = st.session_state.load_profile
        pv_surplus = None
    
    target = st.session_state.target_limit
    peak = st.session_state.peak_load
    tech = st.session_state.technical_params
    econ = st.session_state.economic_params
    power_price = st.session_state.power_price
    
    info_text = f"""
    <div class="info-box">
        <strong>Berechnungsgrundlage:</strong><br>
        Spitzenlast: {peak*1000:.0f} kW → Ziel: {target*1000:.0f} kW<br>
        Erforderliche Reduktion: {(peak-target)*1000:.0f} kW
    """
    if has_pv and pv_surplus is not None:
        info_text += f"<br>PV-Überschuss wird zusätzlich im Speicher geladen."
    info_text += "</div>"
    
    st.markdown(info_text, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Berechnungsmodus wählen
    st.markdown("#### 🎯 Berechnungsmodus")
    calc_mode = st.radio(
        "Wie soll die Speichergröße bestimmt werden?",
        ["Optimale Größe für Zielgrenze", "Optimale Größe für Wunsch-Amortisation"],
        horizontal=True,
        key="calc_mode_b",
        help="Wählen Sie, ob der Speicher für die gewählte Zielgrenze oder für eine bestimmte Amortisationszeit optimiert werden soll."
    )
    
    target_payback = None
    if calc_mode == "Optimale Größe für Wunsch-Amortisation":
        col1, col2 = st.columns(2)
        with col1:
            target_payback = st.number_input(
                "Gewünschte Amortisationszeit (Jahre)",
                min_value=1,
                max_value=25,
                value=10,
                step=1,
                help="Der Speicher wird so dimensioniert, dass er sich innerhalb dieser Zeit amortisiert."
            )
        with col2:
            st.info(f"🎯 Es wird die größtmögliche Speicherkonfiguration gesucht, "
                   f"die sich innerhalb von {target_payback} Jahren amortisiert.")
    
    st.markdown("---")
    
    # Berechnung starten
    if calc_mode == "Optimale Größe für Zielgrenze":
        button_text = "🔍 Speichergröße berechnen"
    else:
        button_text = f"🔍 Optimale Größe für {target_payback} Jahre Amortisation berechnen"
    
    if st.button(button_text, type="primary", use_container_width=True):
        
        if calc_mode == "Optimale Größe für Zielgrenze":
            # Neue Logik: Einmalige PyPSA-Optimierung mit erweiterbarer Kapazität
            st.info("⏳ Optimiere Speichergröße...")
            progress_bar = st.progress(0.3, text="PyPSA-Optimierung läuft...")
            
            # Maximale Größe schätzen
            max_power = (peak - target) / tech['efficiency'] * 1.5
            max_capacity = max_power * 8  # Max E/P = 8h
            
            result = optimize_peak_shaving_storage(
                load_profile=load,
                target_limit=target,
                efficiency=tech['efficiency'],
                soc_min=tech['soc_min'],
                soc_max=tech['soc_max'],
                max_power_mw=max_power,
                max_capacity_mwh=max_capacity,
                pv_surplus=pv_surplus,
                capex_power=econ['capex_power'],
                capex_energy=econ['capex_energy'],
                interest_rate=econ['discount_rate'] / 100,  # % zu Dezimal
                lifetime_years=econ['project_lifetime'],
                opex_rate=econ['opex_rate'] / 100,  # % zu Dezimal
            )
            
            progress_bar.progress(1.0, text="✅ Fertig!")
            
            if result is None:
                st.error("❌ PyPSA-Optimierung fehlgeschlagen. Bitte Parameter prüfen.")
                return
            
            best_capacity = result['optimal_capacity_mwh']
            required_power = result['optimal_power_mw']
            best_result = result
            
            st.session_state.optimal_storage_size = best_capacity
            st.session_state.optimal_power = required_power
            st.session_state.peak_shaving_results = best_result
            
            st.success("✅ Berechnung abgeschlossen!")
            
            # Ergebnisse anzeigen
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Optimale Kapazität", f"{best_capacity*1000:.0f} kWh")
            with col2:
                st.metric("Optimale Leistung", f"{required_power*1000:.0f} kW")
            with col3:
                ep_ratio = best_capacity / required_power if required_power > 0 else 0
                st.metric("E/P-Verhältnis", f"{ep_ratio:.1f} h")
            with col4:
                if best_result['target_achieved']:
                    st.metric("Status", "✅ Ziel erreicht")
                else:
                    st.metric("Nicht gekappt", f"{best_result['unmet_peaks_count']} Zeitpunkte")
        
        else:
            # Neue Logik: Speicher für Wunsch-Amortisation
            st.info("⏳ Die Optimierung kann einige Minuten dauern. Fortschritt wird unten angezeigt.")
            
            # Stop-Button über Session State
            if 'stop_optimization' not in st.session_state:
                st.session_state.stop_optimization = False
            
            stop_col1, stop_col2 = st.columns([3, 1])
            with stop_col2:
                if st.button("⏹️ Abbrechen", type="secondary", use_container_width=True):
                    st.session_state.stop_optimization = True
                    st.warning("Optimierung wird abgebrochen...")
                    st.rerun()
            
            # Verschiedene Speichergrößen durchprobieren
            results_list = []
            
            # Leistung variieren von 10% bis 100% der max. Reduktion
            max_reduction = peak - target
            
            # Reduzierte Schrittzahl für schnellere Berechnung
            power_steps = list(range(10, 101, 10))  # 10 Schritte statt 19
            ep_ratios = np.arange(0.5, 4.1, 0.5)    # 8 Schritte statt 15
            
            total_iterations = len(power_steps) * len(ep_ratios)
            progress_bar = st.progress(0, text="Starte Optimierung...")
            status_text = st.empty()
            
            current_iteration = 0
            
            for power_pct in power_steps:
                # Stop-Check
                if st.session_state.get('stop_optimization', False):
                    st.session_state.stop_optimization = False
                    st.warning("⚠️ Optimierung abgebrochen!")
                    break
                
                storage_power = max_reduction * power_pct / 100 / tech['efficiency']
                
                for ep_ratio in ep_ratios:
                    current_iteration += 1
                    progress = current_iteration / total_iterations
                    progress_bar.progress(progress, text=f"Berechne... {current_iteration}/{total_iterations}")
                    status_text.text(f"Teste: {storage_power*1000:.0f} kW, E/P={ep_ratio:.1f}h")
                    
                    storage_capacity = storage_power * ep_ratio
                    
                    result = calculate_peak_shaving(
                        load_profile=load,
                        target_limit=target,
                        storage_power_mw=storage_power,
                        storage_capacity_mwh=storage_capacity,
                        efficiency=tech['efficiency'],
                        soc_min=tech['soc_min'],
                        soc_max=tech['soc_max'],
                        pv_surplus=pv_surplus
                    )
                    
                    # Wirtschaftlichkeit berechnen
                    reduction_kw = result['peak_reduction_mw'] * 1000
                    savings_per_year = reduction_kw * power_price
                    
                    capex = (storage_capacity * 1000 * econ['capex_energy'] + 
                            storage_power * 1000 * econ['capex_power'])
                    
                    if savings_per_year > 0:
                        payback = capex / savings_per_year
                    else:
                        payback = float('inf')
                    
                    results_list.append({
                        'power_kw': storage_power * 1000,
                        'capacity_kwh': storage_capacity * 1000,
                        'ep_ratio': ep_ratio,
                        'reduction_kw': reduction_kw,
                        'savings_eur': savings_per_year,
                        'capex_eur': capex,
                        'payback_years': payback,
                        'target_achieved': result['target_achieved'],
                        'unmet_peaks': result['unmet_peaks_count'],
                        'result': result
                    })
            
            progress_bar.progress(1.0, text="✅ Fertig!")
            status_text.empty()
            
            # Filtern nach Amortisationsziel
            valid_configs = [r for r in results_list 
                            if r['payback_years'] <= target_payback and r['reduction_kw'] > 0]
            
            if valid_configs:
                # Beste Konfiguration: Maximale Reduktion innerhalb Amortisationsziel
                best_config = max(valid_configs, key=lambda x: x['reduction_kw'])
                
                st.success(f"✅ Optimale Konfiguration für {target_payback} Jahre Amortisation gefunden!")
                
                # Ergebnisse anzeigen
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Kapazität", f"{best_config['capacity_kwh']:.0f} kWh")
                with col2:
                    st.metric("Leistung", f"{best_config['power_kw']:.0f} kW")
                with col3:
                    st.metric("Peak-Reduktion", f"{best_config['reduction_kw']:.0f} kW")
                with col4:
                    st.metric("Amortisation", f"{best_config['payback_years']:.1f} Jahre")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CAPEX", f"{best_config['capex_eur']:,.0f} €")
                with col2:
                    st.metric("Einsparung/Jahr", f"{best_config['savings_eur']:,.0f} €")
                with col3:
                    st.metric("E/P-Verhältnis", f"{best_config['ep_ratio']:.1f} h")
                
                # In Session State speichern
                st.session_state.optimal_storage_size = best_config['capacity_kwh'] / 1000
                st.session_state.optimal_power = best_config['power_kw'] / 1000
                st.session_state.peak_shaving_results = best_config['result']
                best_result = best_config['result']
                best_capacity = best_config['capacity_kwh'] / 1000
                required_power = best_config['power_kw'] / 1000
                
                # Vergleichstabelle anzeigen
                with st.expander("📊 Alle Konfigurationen innerhalb Amortisationsziel"):
                    df_valid = pd.DataFrame([{
                        'Kapazität (kWh)': r['capacity_kwh'],
                        'Leistung (kW)': r['power_kw'],
                        'E/P (h)': r['ep_ratio'],
                        'Reduktion (kW)': r['reduction_kw'],
                        'Einsparung (€/a)': r['savings_eur'],
                        'CAPEX (€)': r['capex_eur'],
                        'Amortisation (a)': r['payback_years']
                    } for r in sorted(valid_configs, key=lambda x: -x['reduction_kw'])[:20]])
                    
                    st.dataframe(df_valid, use_container_width=True)
            
            else:
                st.error(f"❌ Keine Konfiguration erfüllt das Amortisationsziel von {target_payback} Jahren!")
                
                # Beste gefundene Konfiguration anzeigen
                valid_any = [r for r in results_list if r['reduction_kw'] > 0]
                if valid_any:
                    best_found = min(valid_any, key=lambda x: x['payback_years'])
                    
                    st.warning(f"Beste gefundene Amortisation: **{best_found['payback_years']:.1f} Jahre**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Kapazität", f"{best_found['capacity_kwh']:.0f} kWh")
                    with col2:
                        st.metric("Leistung", f"{best_found['power_kw']:.0f} kW")
                    with col3:
                        st.metric("Peak-Reduktion", f"{best_found['reduction_kw']:.0f} kW")
                    
                    st.info("💡 **Tipps zur Verbesserung:**\n"
                           "- Höhere Zielgrenze wählen (geringere Reduktion, aber bessere Wirtschaftlichkeit)\n"
                           "- Höheren Leistungspreis verhandeln\n"
                           "- Günstigere Speicherkosten durch andere Technologie")
                    
                    # Trotzdem speichern für Visualisierung
                    st.session_state.optimal_storage_size = best_found['capacity_kwh'] / 1000
                    st.session_state.optimal_power = best_found['power_kw'] / 1000
                    st.session_state.peak_shaving_results = best_found['result']
                    best_result = best_found['result']
                    best_capacity = best_found['capacity_kwh'] / 1000
                    required_power = best_found['power_kw'] / 1000
                else:
                    st.error("Keine gültige Konfiguration gefunden. Bitte Parameter prüfen.")
                    best_result = None
        
        # Visualisierung (für beide Modi)
        if 'best_result' in dir() and best_result is not None:
            st.markdown("---")
            st.markdown("#### Beispielwoche mit Peak Shaving")
            
            # Eine Woche mit hoher Last finden
            weekly_peaks = [load.iloc[i*672:(i+1)*672].max() for i in range(len(load)//672)]
            peak_week = np.argmax(weekly_peaks)
            start_idx = peak_week * 672
            end_idx = min(start_idx + 672, len(load))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            x = range(end_idx - start_idx)
            
            # Original vs. gekappt
            ax1.plot(x, load.iloc[start_idx:end_idx] * 1000, 
                    label='Original', alpha=0.7, linewidth=1)
            ax1.plot(x, best_result['grid_power_profile'][start_idx:end_idx] * 1000,
                    label='Mit Speicher', linewidth=1)
            ax1.axhline(y=target * 1000, color='red', linestyle='--', 
                       label=f'Zielgrenze: {target*1000:.0f} kW')
            ax1.set_ylabel('Netzbezug (kW)')
            ax1.set_title('Lastprofil mit Peak Shaving')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Speicher-SoC
            ax2.fill_between(x, 
                            best_result['soc_profile'][start_idx:end_idx] / best_capacity * 100,
                            alpha=0.5, color='#3498db')
            ax2.plot(x, best_result['soc_profile'][start_idx:end_idx] / best_capacity * 100,
                    color='#2980b9')
            ax2.set_ylabel('Ladezustand (%)')
            ax2.set_xlabel('Zeitschritt (15 min)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col3:
        if st.session_state.peak_shaving_results is not None:
            if st.button("Weiter →", type="primary", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()


def show_mode_b_step5():
    """Modus B - Schritt 5: Wirtschaftlichkeit und Ergebnisse."""
    
    st.markdown("### Schritt 5: Wirtschaftlichkeitsanalyse")
    
    capacity = st.session_state.optimal_storage_size
    power = st.session_state.optimal_power
    peak = st.session_state.peak_load
    target = st.session_state.target_limit
    result = st.session_state.peak_shaving_results
    tech = st.session_state.technical_params
    econ = st.session_state.economic_params
    power_price = st.session_state.power_price
    
    # Berechnungen
    capex = capacity * 1000 * econ['capex_energy'] + power * 1000 * econ['capex_power']
    opex = capex * econ['opex_rate'] / 100
    
    # Einsparung durch Leistungsreduktion
    old_peak_cost = peak * 1000 * power_price
    new_peak_cost = result['new_peak_mw'] * 1000 * power_price
    annual_savings = old_peak_cost - new_peak_cost
    
    # Wirtschaftlichkeit
    net_savings = annual_savings - opex
    payback = capex / net_savings if net_savings > 0 else float('inf')
    
    # NPV
    npv = -capex
    for y in range(1, econ['project_lifetime'] + 1):
        npv += net_savings / ((1 + econ['discount_rate']/100) ** y)
    
    st.markdown("---")
    
    st.markdown("## 📊 Ergebniszusammenfassung")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.0f} kWh</div>
            <div class="metric-label">Speicherkapazität</div>
        </div>
        """.format(capacity * 1000), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.0f} kW</div>
            <div class="metric-label">Speicherleistung</div>
        </div>
        """.format(power * 1000), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:,.0f} €</div>
            <div class="metric-label">Jährliche Einsparung</div>
        </div>
        """.format(annual_savings), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f} Jahre</div>
            <div class="metric-label">Amortisationszeit</div>
        </div>
        """.format(payback), unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Kostenübersicht")
        
        cost_data = {
            "Position": [
                "Bisherige Leistungskosten",
                "Neue Leistungskosten",
                "Brutto-Einsparung",
                "Betriebskosten Speicher",
                "Netto-Einsparung",
                "",
                "Investitionskosten",
                "Kapitalwert (NPV)",
                "Amortisationszeit",
            ],
            "Wert": [
                f"{old_peak_cost:,.0f} €/Jahr",
                f"{new_peak_cost:,.0f} €/Jahr",
                f"{annual_savings:,.0f} €/Jahr",
                f"-{opex:,.0f} €/Jahr",
                f"{net_savings:,.0f} €/Jahr",
                "",
                f"{capex:,.0f} €",
                f"{npv:,.0f} €",
                f"{payback:.1f} Jahre",
            ]
        }
        st.table(pd.DataFrame(cost_data))
    
    with col2:
        st.markdown("### 📉 Leistungsreduktion")
        
        peak_data = {
            "Kennzahl": [
                "Bisherige Spitzenlast",
                "Zielgrenze",
                "Erreichte Spitzenlast",
                "Reduktion",
            ],
            "Wert": [
                f"{peak*1000:.0f} kW",
                f"{target*1000:.0f} kW",
                f"{result['new_peak_mw']*1000:.0f} kW",
                f"{(peak - result['new_peak_mw'])*1000:.0f} kW ({(peak - result['new_peak_mw'])/peak*100:.1f}%)",
            ]
        }
        st.table(pd.DataFrame(peak_data))
        
        if result['target_achieved']:
            st.success("✅ Zielgrenze wird eingehalten!")
        else:
            st.warning(f"⚠️ Zielgrenze wird an {result['unmet_peaks_count']} Zeitpunkten überschritten")
    
    st.markdown("---")
    
    # Export
    st.markdown("### 💾 Ergebnisse exportieren")
    
    export_data = {
        "projekt": {
            "datum": datetime.now().isoformat(),
            "modus": "B - Peak Shaving",
        },
        "lastanalyse": {
            "spitzenlast_kw": peak * 1000,
            "mittlere_last_kw": st.session_state.load_profile.mean() * 1000,
            "zielgrenze_kw": target * 1000,
        },
        "speicher": {
            "kapazitaet_kwh": capacity * 1000,
            "leistung_kw": power * 1000,
            "wirkungsgrad": tech['efficiency'],
        },
        "wirtschaftlichkeit": {
            "investitionskosten_eur": capex,
            "jaehrliche_einsparung_eur": annual_savings,
            "netto_einsparung_eur": net_savings,
            "amortisation_jahre": payback,
            "npv_eur": npv,
        },
    }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    st.download_button(
        "📥 Ergebnisse als JSON herunterladen",
        json_str,
        "peak_shaving_ergebnis.json",
        "application/json",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()
    with col3:
        if st.button("🏠 Neues Projekt", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# =============================================================================
# Modus C: NVP-Überbauung
# =============================================================================

def simulate_nvp_storage(wind_profile, pv_profile, p_nvp_mw, p_wind_inst_mw, p_pv_inst_mw,
                         storage_power_mw, storage_capacity_mwh, eta_charge=0.95, 
                         eta_discharge=0.95, soc_min=0.0, soc_max=1.0,
                         da_prices=None, price_threshold=None, discharge_strategy="immediate"):
    """
    Simuliert den Speicherbetrieb bei NVP-Überbauung mit PyPSA.
    
    KORREKTES MODELL:
    - EE-Erzeugung als negative Last (muss abgenommen werden)
    - Netzeinspeisung begrenzt auf NVP (mit Erlös)
    - Speicher (Store mit Links für Effizienz)
    - Abregelung als Slack-Senke
    
    Args:
        da_prices: Day-Ahead-Preise (€/MWh), optional
        price_threshold: Nicht verwendet (PyPSA optimiert automatisch)
        discharge_strategy: Nicht verwendet (PyPSA optimiert automatisch)
    
    Returns:
        dict mit Zeitreihen und Kennzahlen
    """
    n = len(wind_profile) if wind_profile is not None else len(pv_profile)
    dt = 0.25  # 15 Minuten
    
    # Erzeugung berechnen
    p_wind = wind_profile.values * p_wind_inst_mw if wind_profile is not None else np.zeros(n)
    p_pv = pv_profile.values * p_pv_inst_mw if pv_profile is not None else np.zeros(n)
    p_ee = p_wind + p_pv
    
    # Preise vorbereiten
    if da_prices is not None and len(da_prices) >= n:
        prices = da_prices.values[:n] if hasattr(da_prices, 'values') else np.array(da_prices[:n])
        avg_price = np.mean(prices)
    else:
        prices = None
        avg_price = 80
    
    # Nutzbare Kapazität
    usable_capacity = storage_capacity_mwh * (soc_max - soc_min)
    initial_soc = usable_capacity * 0.5
    
    # === PyPSA Netzwerk ===
    network = pypsa.Network()
    network.set_snapshots(range(n))
    
    # Zeitauflösung
    network.snapshot_weightings.loc[:, "generators"] = dt
    network.snapshot_weightings.loc[:, "stores"] = dt
    network.snapshot_weightings.loc[:, "links"] = dt
    
    # === Hauptbus ===
    network.add("Bus", "main")
    
    # === EE-Erzeugung als negative Last (MUSS abgenommen werden) ===
    network.add("Load", "ee_generation",
                bus="main",
                p_set=-p_ee)  # Negativ = Einspeisung
    
    # === Netzeinspeisung (begrenzt auf NVP, mit Erlös) ===
    network.add("Bus", "grid")
    
    # Generator am Grid-Bus als Senke
    network.add("Generator", "grid_sink",
                bus="grid",
                p_nom=p_nvp_mw * 2,
                p_min_pu=-1,
                p_max_pu=0,
                marginal_cost=0)
    
    if prices is not None:
        network.add("Link", "grid_feed",
                    bus0="main",
                    bus1="grid",
                    p_nom=p_nvp_mw,
                    efficiency=1.0,
                    marginal_cost=0)
        network.links_t.marginal_cost = pd.DataFrame(
            {"grid_feed": -prices},
            index=network.snapshots
        )
    else:
        network.add("Link", "grid_feed",
                    bus0="main",
                    bus1="grid",
                    p_nom=p_nvp_mw,
                    efficiency=1.0,
                    marginal_cost=-avg_price)
    
    # === Speicher ===
    network.add("Bus", "storage")
    
    network.add("Store", "battery",
                bus="storage",
                e_nom=usable_capacity,
                e_initial=initial_soc,
                e_cyclic=False,
                standing_loss=0)
    
    # Laden: main -> storage
    network.add("Link", "charge",
                bus0="main",
                bus1="storage",
                p_nom=storage_power_mw,
                efficiency=eta_charge,
                marginal_cost=0.001)
    
    # Entladen: storage -> main
    network.add("Link", "discharge",
                bus0="storage",
                bus1="main",
                p_nom=storage_power_mw,
                efficiency=eta_discharge,
                marginal_cost=0)
    
    # === Abregelung (Senke ohne Erlös) ===
    network.add("Bus", "curtail")
    
    network.add("Generator", "curtail_sink",
                bus="curtail",
                p_nom=p_ee.max() * 2,
                p_min_pu=-1,
                p_max_pu=0,
                marginal_cost=0)
    
    network.add("Link", "curtailment",
                bus0="main",
                bus1="curtail",
                p_nom=p_ee.max(),
                efficiency=1.0,
                marginal_cost=0.01)
    
    # === Optimieren ===
    try:
        status = network.optimize(solver_name="highs", solver_options={"threads": 1})
        optimization_ok = status[0] == "ok"
    except Exception as e:
        st.error(f"❌ PyPSA-Optimierungsfehler: {e}")
        return None
    
    if not optimization_ok:
        st.error(f"❌ Optimierung nicht erfolgreich: {status}")
        return None
    
    # === Ergebnisse extrahieren ===
    p_grid = network.links_t.p0["grid_feed"].values
    p_charge = network.links_t.p0["charge"].values
    p_discharge_out = network.links_t.p1["discharge"].values
    p_curtail = network.links_t.p0["curtailment"].values
    soc = network.stores_t.e["battery"].values
    
    # Überschuss (Referenz)
    p_surplus = np.maximum(p_ee - p_nvp_mw, 0)
    
    # SOC mit soc_min Offset
    soc_absolute = soc + soc_min * storage_capacity_mwh
    
    # Erlös berechnen
    if prices is not None:
        discharge_revenue = p_grid * dt * prices
        total_discharge_revenue = discharge_revenue.sum()
        avg_discharge_price = total_discharge_revenue / (p_grid.sum() * dt) if p_grid.sum() > 0 else None
    else:
        discharge_revenue = np.zeros(n)
        total_discharge_revenue = 0
        avg_discharge_price = None
    
    # Kennzahlen berechnen
    total_generation = p_ee.sum() * dt
    total_surplus = p_surplus.sum() * dt
    total_curtailment = p_curtail.sum() * dt
    total_grid_feed = p_grid.sum() * dt
    total_charged = p_charge.sum() * dt
    total_discharged = p_discharge_out.sum() * dt
    
    # Netzauslastung
    grid_utilization = total_grid_feed / (p_nvp_mw * n * dt) if p_nvp_mw > 0 else 0
    
    # Erfassungsgrad
    capture_rate = min(total_charged / total_surplus, 1.0) if total_surplus > 0 else 1.0
    
    # Zyklen
    cycles = total_discharged / storage_capacity_mwh if storage_capacity_mwh > 0 else 0
    
    return {
        'p_ee': p_ee,
        'p_grid': p_grid,
        'p_charge': p_charge,
        'p_discharge': p_discharge_out,
        'p_curtail': p_curtail,
        'p_surplus': p_surplus,
        'soc': soc_absolute,
        'prices': prices,
        'discharge_revenue': discharge_revenue,
        'total_generation_mwh': total_generation,
        'total_surplus_mwh': total_surplus,
        'total_curtailment_mwh': total_curtailment,
        'total_grid_feed_mwh': total_grid_feed,
        'total_charged_mwh': total_charged,
        'total_discharged_mwh': total_discharged,
        'total_discharge_revenue': total_discharge_revenue,
        'avg_discharge_price': avg_discharge_price,
        'grid_utilization': grid_utilization,
        'capture_rate': capture_rate,
        'cycles': cycles,
        'optimization_status': 'optimal',
        'objective_value': network.objective,
    }


def run_nvp_parameter_study(wind_profile, pv_profile, p_nvp_mw, p_wind_inst_mw, p_pv_inst_mw,
                            power_range, duration_range, eta_charge=0.95, eta_discharge=0.95,
                            da_prices=None, price_threshold=None, discharge_strategy="immediate"):
    """
    Führt eine Parameterstudie für verschiedene Speicherkonfigurationen durch.
    
    Args:
        power_range: Liste von Speicherleistungen (relativ zu P_NVP, z.B. [0.1, 0.2, ...])
        duration_range: Liste von Speicherdauern in Stunden
        da_prices: Day-Ahead-Preise (optional)
        price_threshold: Preis-Schwellwert oder Perzentil
        discharge_strategy: Entladestrategie
        
    Returns:
        DataFrame mit Ergebnissen für jede Kombination
    """
    results = []
    
    for power_ratio in power_range:
        for duration in duration_range:
            storage_power = p_nvp_mw * power_ratio
            storage_capacity = storage_power * duration
            
            sim = simulate_nvp_storage(
                wind_profile=wind_profile,
                pv_profile=pv_profile,
                p_nvp_mw=p_nvp_mw,
                p_wind_inst_mw=p_wind_inst_mw,
                p_pv_inst_mw=p_pv_inst_mw,
                storage_power_mw=storage_power,
                storage_capacity_mwh=storage_capacity,
                eta_charge=eta_charge,
                eta_discharge=eta_discharge,
                da_prices=da_prices,
                price_threshold=price_threshold,
                discharge_strategy=discharge_strategy,
            )
            
            result_entry = {
                'power_ratio': power_ratio,
                'duration_h': duration,
                'storage_power_mw': storage_power,
                'storage_capacity_mwh': storage_capacity,
                'capture_rate': sim['capture_rate'],
                'grid_utilization': sim['grid_utilization'],
                'curtailment_mwh': sim['total_curtailment_mwh'],
                'surplus_mwh': sim['total_surplus_mwh'],
                'cycles': sim['cycles'],
                'discharged_mwh': sim['total_discharged_mwh'],
            }
            
            # Erlösdaten hinzufügen wenn Preise vorhanden
            if da_prices is not None:
                result_entry['discharge_revenue'] = sim['total_discharge_revenue']
                result_entry['avg_discharge_price'] = sim['avg_discharge_price']
            
            results.append(result_entry)
    
    return pd.DataFrame(results)


def optimize_nvp_storage(wind_profile, pv_profile, p_nvp_mw, p_wind_inst_mw, p_pv_inst_mw,
                         max_storage_power_mw, max_storage_capacity_mwh,
                         eta_charge=0.95, eta_discharge=0.95, soc_min=0.0, soc_max=1.0,
                         da_prices=None, capex_power=80, capex_energy=250,
                         interest_rate=0.05, lifetime_years=15,
                         opex_rate=0.02, feed_in_value=80):
    """
    Optimiert Speicherleistung UND -kapazität für NVP-Überbauung.
    
    EINFACHES KORREKTES MODELL:
    - Ein Hauptbus
    - EE-Erzeugung muss verbraucht werden (als negative Last)
    - Netzeinspeisung begrenzt auf NVP (Link mit Erlös)
    - Speicher (Store mit Links für Effizienz)
    - Abregelung (Link ohne Erlös)
    
    Wirtschaftliche Parameter:
    - CAPEX: Investitionskosten (annualisiert)
    - OPEX: Betriebskosten (% von CAPEX pro Jahr)
    - Wert vermiedener Abregelung: feed_in_value (€/MWh)
    
    Args:
        wind_profile: Normiertes Windprofil (0-1)
        pv_profile: Normiertes PV-Profil (0-1)
        p_nvp_mw: Netzverknüpfungspunkt-Leistung (MW)
        p_wind_inst_mw: Installierte Windleistung (MW)
        p_pv_inst_mw: Installierte PV-Leistung (MW)
        max_storage_power_mw: Max. Speicherleistung für Optimierung (MW)
        max_storage_capacity_mwh: Max. Speicherkapazität für Optimierung (MWh)
        da_prices: Day-Ahead-Preise (€/MWh), optional
        capex_power: CAPEX Leistung (€/kW)
        capex_energy: CAPEX Energie (€/kWh)
        interest_rate: Zinssatz für Annuität (z.B. 0.05 = 5%)
        lifetime_years: Lebensdauer für Annuität (Jahre)
        opex_rate: Betriebskosten als Anteil von CAPEX (z.B. 0.02 = 2%)
        feed_in_value: Marktwert der Einspeisung / Wert vermiedener Abregelung (€/MWh)
    
    Returns:
        dict mit optimaler Konfiguration und Ergebnissen
    """
    n = len(wind_profile) if wind_profile is not None else len(pv_profile)
    dt = 0.25  # 15 Minuten
    
    # Erzeugung berechnen
    p_wind = wind_profile.values * p_wind_inst_mw if wind_profile is not None else np.zeros(n)
    p_pv = pv_profile.values * p_pv_inst_mw if pv_profile is not None else np.zeros(n)
    p_ee = p_wind + p_pv
    
    # Preise vorbereiten
    if da_prices is not None and len(da_prices) >= n:
        prices = da_prices.values[:n] if hasattr(da_prices, 'values') else np.array(da_prices[:n])
        # Bei DA-Preisen: Diese direkt verwenden
        use_price_profile = True
    else:
        prices = None
        # Ohne DA-Preise: Fester Marktwert verwenden
        use_price_profile = False
    
    # Überschuss berechnen (Referenz für "ohne Speicher")
    p_surplus = np.maximum(p_ee - p_nvp_mw, 0)
    total_surplus_ref = p_surplus.sum() * dt
    
    # Referenz ohne Speicher
    p_grid_no_storage = np.minimum(p_ee, p_nvp_mw)
    total_grid_no_storage = p_grid_no_storage.sum() * dt
    p_curtail_no_storage = p_ee - p_grid_no_storage
    total_curtail_no_storage = p_curtail_no_storage.sum() * dt
    
    # Wenn kein Überschuss, brauchen wir keinen Speicher
    if total_surplus_ref < 0.1:
        return {
            'optimal_capacity_mwh': 0,
            'optimal_power_mw': 0,
            'ep_ratio': 0,
            'p_ee': p_ee,
            'p_grid': p_grid_no_storage,
            'p_charge': np.zeros(n),
            'p_discharge': np.zeros(n),
            'p_curtail': p_curtail_no_storage,
            'p_surplus': p_surplus,
            'soc': np.zeros(n),
            'prices': prices,
            'discharge_revenue': np.zeros(n),
            'total_generation_mwh': p_ee.sum() * dt,
            'total_surplus_mwh': total_surplus_ref,
            'total_curtailment_mwh': total_curtail_no_storage,
            'total_grid_feed_mwh': total_grid_no_storage,
            'total_charged_mwh': 0,
            'total_discharged_mwh': 0,
            'total_discharge_revenue': 0,
            'avg_discharge_price': None,
            'grid_utilization': total_grid_no_storage / (p_nvp_mw * n * dt),
            'capture_rate': 0,
            'cycles': 0,
            'optimization_status': 'no_surplus',
            'objective_value': 0,
            'ref_grid_feed_mwh': total_grid_no_storage,
            'ref_curtailment_mwh': total_curtail_no_storage,
        }
    
    # === Annuität berechnen (KORREKT!) ===
    annuity_factor = calculate_annuity_factor(interest_rate, lifetime_years)
    
    # Jährliche Kapitalkosten pro Einheit (CAPEX annualisiert + OPEX)
    annual_capex_energy = capex_energy * 1000 * annuity_factor  # €/MWh/Jahr
    annual_opex_energy = capex_energy * 1000 * opex_rate        # €/MWh/Jahr
    annual_cost_energy = annual_capex_energy + annual_opex_energy  # Gesamtkosten Energie
    
    annual_capex_power = capex_power * 1000 * annuity_factor    # €/MW/Jahr
    annual_opex_power = capex_power * 1000 * opex_rate          # €/MW/Jahr
    annual_cost_power = annual_capex_power + annual_opex_power  # Gesamtkosten Leistung
    
    # Nutzbare Kapazität (max)
    max_usable_capacity = max_storage_capacity_mwh * (soc_max - soc_min)
    
    # === PyPSA Netzwerk ===
    network = pypsa.Network()
    network.set_snapshots(range(n))
    
    # Zeitauflösung setzen
    network.snapshot_weightings.loc[:, "generators"] = dt
    network.snapshot_weightings.loc[:, "stores"] = dt
    network.snapshot_weightings.loc[:, "links"] = dt
    
    # === Hauptbus ===
    network.add("Bus", "main")
    
    # === EE-Erzeugung als negative Last (MUSS abgenommen werden) ===
    network.add("Load", "ee_generation",
                bus="main",
                p_set=-p_ee)  # Negativ = Einspeisung in den Bus
    
    # === Netzeinspeisung (Link zu Senke, begrenzt auf NVP, mit Erlös) ===
    network.add("Bus", "grid")
    
    # Generator am Grid-Bus der unbegrenzt aufnehmen kann (= Netz)
    network.add("Generator", "grid_sink",
                bus="grid",
                p_nom=p_nvp_mw * 2,  # Großzügig
                p_min_pu=-1,  # Kann "negative" Leistung = aufnehmen
                p_max_pu=0,   # Nimmt nur auf, erzeugt nicht
                marginal_cost=0)
    
    # Link vom Hauptbus zum Grid (begrenzt auf NVP)
    if use_price_profile and prices is not None:
        network.add("Link", "grid_feed",
                    bus0="main",
                    bus1="grid",
                    p_nom=p_nvp_mw,
                    efficiency=1.0,
                    marginal_cost=0)
        # Zeitvariable negative Kosten = Erlös (DA-Preise)
        network.links_t.marginal_cost = pd.DataFrame(
            {"grid_feed": -prices},
            index=network.snapshots
        )
    else:
        # Ohne DA-Preise: Fester Marktwert der Einspeisung
        network.add("Link", "grid_feed",
                    bus0="main",
                    bus1="grid",
                    p_nom=p_nvp_mw,
                    efficiency=1.0,
                    marginal_cost=-feed_in_value)  # Wert der vermiedenen Abregelung
    
    # === Speicher ===
    network.add("Bus", "storage")
    
    # Store
    network.add("Store", "battery",
                bus="storage",
                e_nom=0,
                e_nom_extendable=True,
                e_nom_min=0,
                e_nom_max=max_usable_capacity,
                e_cyclic=False,
                e_initial=0,
                standing_loss=0,
                capital_cost=annual_cost_energy)
    
    # Laden: main -> storage
    network.add("Link", "charge",
                bus0="main",
                bus1="storage",
                p_nom=0,
                p_nom_extendable=True,
                p_nom_min=0,
                p_nom_max=max_storage_power_mw,
                efficiency=eta_charge,
                marginal_cost=0.001,
                capital_cost=annual_cost_power / 2)
    
    # Entladen: storage -> main
    network.add("Link", "discharge",
                bus0="storage",
                bus1="main",
                p_nom=0,
                p_nom_extendable=True,
                p_nom_min=0,
                p_nom_max=max_storage_power_mw,
                efficiency=eta_discharge,
                marginal_cost=0,
                capital_cost=annual_cost_power / 2)
    
    # === Abregelung (Link zu Senke ohne Erlös) ===
    network.add("Bus", "curtail")
    
    network.add("Generator", "curtail_sink",
                bus="curtail",
                p_nom=p_ee.max() * 2,
                p_min_pu=-1,
                p_max_pu=0,
                marginal_cost=0)
    
    network.add("Link", "curtailment",
                bus0="main",
                bus1="curtail",
                p_nom=p_ee.max(),
                efficiency=1.0,
                marginal_cost=0.01)  # Kleine Kosten = vermeiden
    
    # === Optimieren ===
    try:
        status = network.optimize(solver_name="highs", solver_options={"threads": 1})
        optimization_ok = status[0] == "ok"
    except Exception as e:
        st.error(f"PyPSA Fehler: {e}")
        return None
    
    if not optimization_ok:
        st.error(f"Optimierung nicht erfolgreich: {status}")
        return None
    
    # === Ergebnisse extrahieren ===
    optimal_capacity = network.stores.loc["battery", "e_nom_opt"]
    optimal_power_charge = network.links.loc["charge", "p_nom_opt"]
    optimal_power_discharge = network.links.loc["discharge", "p_nom_opt"]
    optimal_power = max(optimal_power_charge, optimal_power_discharge)
    
    # Kapazität zurückrechnen (inkl. SOC-Grenzen)
    total_capacity = optimal_capacity / (soc_max - soc_min) if (soc_max - soc_min) > 0 else optimal_capacity
    
    # Zeitreihen
    p_grid = network.links_t.p0["grid_feed"].values
    p_charge = network.links_t.p0["charge"].values
    p_discharge_out = network.links_t.p1["discharge"].values
    p_curtail = network.links_t.p0["curtailment"].values
    
    # SOC
    if optimal_capacity > 0:
        soc = network.stores_t.e["battery"].values
        soc_absolute = soc + soc_min * total_capacity
    else:
        soc_absolute = np.zeros(n)
    
    # Kennzahlen
    total_generation = p_ee.sum() * dt
    total_grid_feed = p_grid.sum() * dt
    total_curtailment = p_curtail.sum() * dt
    total_charged = p_charge.sum() * dt
    total_discharged = p_discharge_out.sum() * dt
    
    # Erlös
    if use_price_profile and prices is not None:
        revenue_profile = p_grid * dt * prices
        total_revenue = revenue_profile.sum()
        avg_discharge_price = total_revenue / total_grid_feed if total_grid_feed > 0 else None
    else:
        revenue_profile = p_grid * dt * feed_in_value
        total_revenue = revenue_profile.sum()
        avg_discharge_price = feed_in_value
    
    # Netzauslastung
    max_possible_feed = p_nvp_mw * n * dt
    grid_utilization = total_grid_feed / max_possible_feed if max_possible_feed > 0 else 0
    
    # Erfassungsgrad (wie viel vom Überschuss wird gespeichert)
    capture_rate = min(total_charged / total_surplus_ref, 1.0) if total_surplus_ref > 0 else 0
    
    # Zyklen
    cycles = total_discharged / total_capacity if total_capacity > 0 else 0
    
    # E/P-Verhältnis
    ep_ratio = total_capacity / optimal_power if optimal_power > 0 else 0
    
    return {
        'optimal_capacity_mwh': total_capacity,
        'optimal_power_mw': optimal_power,
        'ep_ratio': ep_ratio,
        'p_ee': p_ee,
        'p_grid': p_grid,
        'p_charge': p_charge,
        'p_discharge': p_discharge_out,
        'p_curtail': p_curtail,
        'p_surplus': p_surplus,
        'soc': soc_absolute,
        'prices': prices,
        'discharge_revenue': revenue_profile,
        'total_generation_mwh': total_generation,
        'total_surplus_mwh': total_surplus_ref,
        'total_curtailment_mwh': total_curtailment,
        'total_grid_feed_mwh': total_grid_feed,
        'total_charged_mwh': total_charged,
        'total_discharged_mwh': total_discharged,
        'total_discharge_revenue': total_revenue,
        'avg_discharge_price': avg_discharge_price,
        'grid_utilization': grid_utilization,
        'capture_rate': capture_rate,
        'cycles': cycles,
        'ref_grid_feed_mwh': total_grid_no_storage,
        'ref_curtailment_mwh': total_curtail_no_storage,
        'optimization_status': 'optimal',
        'objective_value': network.objective,
        # Wirtschaftliche Parameter
        'annuity_factor': annuity_factor,
        'annual_cost_energy': annual_cost_energy,
        'annual_cost_power': annual_cost_power,
        'annual_capex_energy': annual_capex_energy,
        'annual_opex_energy': annual_opex_energy,
        'feed_in_value': feed_in_value,
        'opex_rate': opex_rate,
    }


def show_mode_c():
    """Zeigt den Ablauf für Modus C: NVP-Überbauung."""
    
    # Header mit Zurück-Button
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Zurück"):
            st.session_state.current_mode = None
            st.rerun()
    with col_title:
        st.markdown("## ⚡ Modus C: NVP-Überbauung")
    
    # Fortschrittsanzeige
    steps = ["Konfiguration", "Daten laden", "Überschussanalyse", "Parameterstudie", "Ergebnisse"]
    current = st.session_state.current_step
    
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current:
                st.success(f"✓ {step}")
            elif i + 1 == current:
                st.info(f"→ {step}")
            else:
                st.text(f"○ {step}")
    
    st.markdown("---")
    
    if current == 1:
        show_mode_c_step1()
    elif current == 2:
        show_mode_c_step2()
    elif current == 3:
        show_mode_c_step3()
    elif current == 4:
        show_mode_c_step4()
    elif current == 5:
        show_mode_c_step5()


def show_mode_c_step1():
    """Modus C - Schritt 1: NVP-Konfiguration."""
    
    st.markdown("### Schritt 1: Konfiguration des Netzverknüpfungspunktes")
    
    st.markdown("""
    <div class="info-box">
        <strong>Was ist NVP-Überbauung?</strong><br>
        Bei der Überbauung wird mehr Erzeugungsleistung (Wind + PV) installiert als 
        der Netzverknüpfungspunkt (NVP) einspeisen kann. Der entstehende Überschuss 
        wird durch einen Batteriespeicher zeitversetzt eingespeist.<br><br>
        <strong>Überbauungsfaktor γ = (P_Wind + P_PV) / P_NVP</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔌 Netzverknüpfungspunkt")
        
        # Einheit wählen - Wert wird automatisch in session_state gespeichert durch key
        unit_nvp = st.radio("Einheit für Leistungen:", ["MW", "kW"], horizontal=True, key="unit_nvp_radio")
        unit_factor = 1 if unit_nvp == "MW" else 1000
        
        p_nvp = st.number_input(
            f"NVP-Anschlussleistung ({unit_nvp})",
            min_value=0.1 if unit_nvp == "MW" else 1.0,
            max_value=10000.0,
            value=10.0 if unit_nvp == "MW" else 1000.0,
            step=1.0,
            help="Maximale Einspeiseleistung am Netzverknüpfungspunkt"
        )
        
        # In MW umrechnen und separat speichern
        p_nvp_mw = p_nvp if unit_nvp == "MW" else p_nvp / 1000
        st.session_state.p_nvp_mw = p_nvp_mw
        st.session_state.unit_display = unit_nvp  # Separater Key für Anzeige
    
    with col2:
        st.markdown("#### 🌬️ Installierte Erzeugungsleistung")
        
        p_wind_inst = st.number_input(
            f"Installierte Windleistung ({unit_nvp})",
            min_value=0.0,
            max_value=50000.0,
            value=15.0 if unit_nvp == "MW" else 1500.0,
            step=1.0,
            help="Nennleistung der Windenergieanlagen"
        )
        
        p_pv_inst = st.number_input(
            f"Installierte PV-Leistung ({unit_nvp})",
            min_value=0.0,
            max_value=50000.0,
            value=10.0 if unit_nvp == "MW" else 1000.0,
            step=1.0,
            help="Installierte Leistung der PV-Anlage"
        )
        
        # In MW umrechnen
        p_wind_inst_mw = p_wind_inst if unit_nvp == "MW" else p_wind_inst / 1000
        p_pv_inst_mw = p_pv_inst if unit_nvp == "MW" else p_pv_inst / 1000
        
        st.session_state.p_wind_inst_mw = p_wind_inst_mw
        st.session_state.p_pv_inst_mw = p_pv_inst_mw
    
    st.markdown("---")
    
    # Überbauungsfaktor anzeigen
    p_ee_total = p_wind_inst_mw + p_pv_inst_mw
    overbuild_factor = p_ee_total / p_nvp_mw if p_nvp_mw > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gesamte EE-Leistung", f"{p_ee_total:.2f} MW")
    with col2:
        st.metric("NVP-Kapazität", f"{p_nvp_mw:.2f} MW")
    with col3:
        st.metric("Überbauungsfaktor γ", f"{overbuild_factor:.1%}")
    with col4:
        wind_share = p_wind_inst_mw / p_ee_total * 100 if p_ee_total > 0 else 0
        st.metric("Windanteil", f"{wind_share:.0f}%")
    
    if overbuild_factor <= 1:
        st.warning("⚠️ Keine Überbauung: Die installierte Leistung ist kleiner oder gleich der NVP-Kapazität.")
    elif overbuild_factor > 3:
        st.warning("⚠️ Sehr hohe Überbauung (>300%): Es werden erhebliche Überschüsse entstehen.")
    else:
        st.success(f"✓ Überbauung von {overbuild_factor:.0%} konfiguriert")
    
    st.markdown("---")
    
    # Speicherparameter
    st.markdown("#### 🔋 Speicherparameter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        eta_charge = st.slider("Ladewirkungsgrad (%)", 80, 99, 95) / 100
        eta_discharge = st.slider("Entladewirkungsgrad (%)", 80, 99, 95) / 100
    
    with col2:
        cost_power = st.number_input("Leistungskosten (€/kW)", 0, 500, 80)
        cost_energy = st.number_input("Kapazitätskosten (€/kWh)", 50, 1000, 200)
    
    st.session_state.eta_charge = eta_charge
    st.session_state.eta_discharge = eta_discharge
    st.session_state.cost_power = cost_power
    st.session_state.cost_energy = cost_energy
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if p_ee_total > 0 and p_nvp_mw > 0:
            if st.button("Weiter →", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()


def show_mode_c_step2():
    """Modus C - Schritt 2: Erzeugungsprofile und Preise laden."""
    
    st.markdown("### Schritt 2: Profile und Preise hochladen")
    
    st.markdown("""
    <div class="info-box">
        <strong>📁 Benötigte Dateien:</strong>
        <ul>
            <li><strong>Windprofil</strong>: Normierte Zeitreihe (0-1) oder absolute Leistung</li>
            <li><strong>PV-Profil</strong>: Normierte Zeitreihe (0-1) oder absolute Leistung</li>
            <li><strong>Day-Ahead-Preise</strong> (optional): Für preisoptimierte Entladung</li>
        </ul>
        <p>Mindestens ein Erzeugungsprofil ist erforderlich.</p>
        <p><strong>Unterstützte Formate:</strong> CSV, Excel (.xlsx, .xls)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌬️ Windprofil")
        
        if st.session_state.p_wind_inst_mw > 0:
            wind_file = st.file_uploader(
                "Windprofil hochladen",
                type=['csv', 'xlsx', 'xls'],
                key="wind_upload_c",
                help="Zeitreihe mit Windleistung"
            )
            
            if wind_file is not None:
                df = load_csv_file(wind_file)
                if df is not None:
                    # Normieren falls nötig
                    values = df.iloc[:, 0].values
                    if values.max() > 1.5:  # Wahrscheinlich absolute Werte
                        is_normalized = st.radio(
                            "Sind die Werte normiert (0-1)?",
                            ["Nein, absolute Werte", "Ja, bereits normiert"],
                            key="wind_norm"
                        )
                        if is_normalized == "Nein, absolute Werte":
                            values = values / values.max()
                    
                    st.session_state.wind_profile_c = pd.Series(values, index=df.index)
                    st.success(f"✓ {len(df)} Datenpunkte geladen")
                    
                    with st.expander("Vorschau"):
                        st.line_chart(st.session_state.wind_profile_c.iloc[:min(1000, len(df))])
        else:
            st.info("Keine Windleistung konfiguriert")
            st.session_state.wind_profile_c = None
    
    with col2:
        st.markdown("#### ☀️ PV-Profil")
        
        if st.session_state.p_pv_inst_mw > 0:
            pv_file = st.file_uploader(
                "PV-Profil hochladen",
                type=['csv', 'xlsx', 'xls'],
                key="pv_upload_c",
                help="Zeitreihe mit PV-Leistung"
            )
            
            if pv_file is not None:
                df = load_csv_file(pv_file)
                if df is not None:
                    # Normieren falls nötig
                    values = df.iloc[:, 0].values
                    if values.max() > 1.5:  # Wahrscheinlich absolute Werte
                        is_normalized = st.radio(
                            "Sind die Werte normiert (0-1)?",
                            ["Nein, absolute Werte", "Ja, bereits normiert"],
                            key="pv_norm"
                        )
                        if is_normalized == "Nein, absolute Werte":
                            values = values / values.max()
                    
                    st.session_state.pv_profile_c = pd.Series(values, index=df.index)
                    st.success(f"✓ {len(df)} Datenpunkte geladen")
                    
                    with st.expander("Vorschau"):
                        st.line_chart(st.session_state.pv_profile_c.iloc[:min(1000, len(df))])
        else:
            st.info("Keine PV-Leistung konfiguriert")
            st.session_state.pv_profile_c = None
    
    st.markdown("---")
    
    # Day-Ahead-Preise und Entladestrategie
    st.markdown("#### 💰 Entladestrategie (optional)")
    
    st.markdown("""
    <div class="info-box">
        <strong>Preisoptimierte Entladung:</strong><br>
        Optional können Day-Ahead-Preise hochgeladen werden, um die Entladung auf 
        Zeiten mit hohen Strompreisen zu beschränken. Dies erhöht den Erlös pro entladener kWh.
    </div>
    """, unsafe_allow_html=True)
    
    use_price_optimization = st.checkbox(
        "Preisoptimierte Entladung aktivieren",
        value=False,
        key="use_price_opt_c",
        help="Speicher entlädt nur, wenn der Strompreis einen Schwellwert überschreitet"
    )
    
    if use_price_optimization:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 💶 Day-Ahead-Preise")
            price_file = st.file_uploader(
                "Day-Ahead-Preise hochladen",
                type=['csv', 'xlsx', 'xls'],
                key="price_upload_c",
                help="Zeitreihe mit Strompreisen in €/MWh"
            )
            
            if price_file is not None:
                df = load_csv_file(price_file)
                if df is not None:
                    st.session_state.da_prices_c = df.iloc[:, 0]
                    prices = df.iloc[:, 0]
                    st.success(f"✓ {len(df)} Preisdaten geladen")
                    
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Mittelwert", f"{prices.mean():.1f} €/MWh")
                    with col_s2:
                        st.metric("Minimum", f"{prices.min():.1f} €/MWh")
                    with col_s3:
                        st.metric("Maximum", f"{prices.max():.1f} €/MWh")
        
        with col2:
            st.markdown("##### ⚙️ Entladestrategie")
            
            discharge_strategy = st.radio(
                "Strategie wählen:",
                ["price_threshold", "price_percentile"],
                format_func=lambda x: "Fester Preis-Schwellwert" if x == "price_threshold" else "Tägliches Preis-Perzentil",
                key="discharge_strategy_c",
                help="Fester Schwellwert: Entladung nur wenn Preis > X €/MWh\nPerzentil: Entladung nur in den oberen X% des Tages"
            )
            
            if discharge_strategy == "price_threshold":
                price_threshold = st.number_input(
                    "Preis-Schwellwert (€/MWh)",
                    min_value=0.0,
                    max_value=500.0,
                    value=80.0,
                    step=5.0,
                    key="price_threshold_c",
                    help="Entladung nur wenn Day-Ahead-Preis über diesem Wert"
                )
                st.session_state.price_threshold_c_val = price_threshold
                st.session_state.discharge_strategy_c_val = "price_threshold"
            else:
                price_percentile = st.slider(
                    "Preis-Perzentil (%)",
                    min_value=50,
                    max_value=95,
                    value=70,
                    step=5,
                    key="price_percentile_c",
                    help="Entladung nur in den oberen X% der Tagespreise (z.B. 70 = obere 30%)"
                )
                st.session_state.price_threshold_c_val = price_percentile
                st.session_state.discharge_strategy_c_val = "price_percentile"
            
            # Visualisierung der Schwellwerte
            if st.session_state.get('da_prices_c') is not None:
                prices = st.session_state.da_prices_c
                if discharge_strategy == "price_threshold":
                    threshold = st.session_state.get('price_threshold_c_val', 80)
                    hours_above = (prices >= threshold).sum() / 4  # Stunden
                    pct_above = (prices >= threshold).mean() * 100
                    st.info(f"📊 {hours_above:.0f} Stunden ({pct_above:.1f}%) über Schwellwert")
                else:
                    percentile = st.session_state.get('price_threshold_c_val', 70)
                    threshold = np.percentile(prices, percentile)
                    st.info(f"📊 Schwellwert entspricht ~{threshold:.1f} €/MWh (Durchschnitt)")
    else:
        st.session_state.da_prices_c = None
        st.session_state.discharge_strategy_c_val = "immediate"
        st.session_state.price_threshold_c_val = None
        st.info("ℹ️ Ohne Preisoptimierung: Speicher entlädt sofort bei freier Netzkapazität")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        has_wind = st.session_state.get('wind_profile_c') is not None
        has_pv = st.session_state.get('pv_profile_c') is not None
        
        # Prüfen ob Preise benötigt aber nicht vorhanden
        needs_prices = st.session_state.get('use_price_opt_c', False)
        has_prices = st.session_state.get('da_prices_c') is not None
        
        if (has_wind or has_pv) and (not needs_prices or has_prices):
            if st.button("Weiter →", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
        else:
            if not (has_wind or has_pv):
                st.warning("⚠️ Bitte laden Sie mindestens ein Erzeugungsprofil hoch")
            elif needs_prices and not has_prices:
                st.warning("⚠️ Bitte laden Sie Day-Ahead-Preise hoch oder deaktivieren Sie die Preisoptimierung")


def show_mode_c_step3():
    """Modus C - Schritt 3: Überschussanalyse ohne Speicher."""
    
    st.markdown("### Schritt 3: Überschussanalyse (ohne Speicher)")
    
    # Daten abrufen
    wind_profile = st.session_state.get('wind_profile_c')
    pv_profile = st.session_state.get('pv_profile_c')
    p_nvp = st.session_state.p_nvp_mw
    p_wind = st.session_state.p_wind_inst_mw
    p_pv = st.session_state.p_pv_inst_mw
    
    # Erzeugung berechnen
    n = len(wind_profile) if wind_profile is not None else len(pv_profile)
    
    p_wind_ts = wind_profile.values * p_wind if wind_profile is not None else np.zeros(n)
    p_pv_ts = pv_profile.values * p_pv if pv_profile is not None else np.zeros(n)
    p_ee = p_wind_ts + p_pv_ts
    
    # Überschuss und Einspeisung ohne Speicher
    p_surplus = np.maximum(p_ee - p_nvp, 0)
    p_grid_no_storage = np.minimum(p_ee, p_nvp)
    
    dt = 0.25  # 15 Minuten
    total_generation = p_ee.sum() * dt
    total_surplus = p_surplus.sum() * dt
    total_grid = p_grid_no_storage.sum() * dt
    
    surplus_share = total_surplus / total_generation * 100 if total_generation > 0 else 0
    grid_utilization = total_grid / (p_nvp * n * dt) * 100 if p_nvp > 0 else 0
    
    st.markdown("---")
    
    # Kennzahlen anzeigen
    st.markdown("#### 📊 Ergebnisse ohne Speicher")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gesamterzeugung", f"{total_generation:,.0f} MWh")
    with col2:
        st.metric("Netzeinspeisung", f"{total_grid:,.0f} MWh")
    with col3:
        st.metric("Überschuss (Abregelung)", f"{total_surplus:,.0f} MWh", f"{surplus_share:.1f}%")
    with col4:
        st.metric("Netzauslastung", f"{grid_utilization:.1f}%")
    
    if surplus_share < 1:
        st.success("✓ Sehr geringe Abregelung - Speicher kaum erforderlich")
    elif surplus_share < 5:
        st.info("ℹ️ Moderate Abregelung - Speicher kann sinnvoll sein")
    else:
        st.warning(f"⚠️ Erhebliche Abregelung von {surplus_share:.1f}% - Speicher empfohlen")
    
    st.markdown("---")
    
    # Visualisierung
    st.markdown("#### 📈 Visualisierung")
    
    tab1, tab2, tab3 = st.tabs(["Zeitreihe", "Jahresdauerlinie", "Histogramm"])
    
    with tab1:
        # Ausschnitt wählen
        days_to_show = st.slider("Anzeigetage", 1, 30, 7, key="days_c3")
        end_idx = min(days_to_show * 96, n)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        x = range(end_idx)
        
        ax1.fill_between(x, p_wind_ts[:end_idx], alpha=0.5, label='Wind', color='#3498db')
        ax1.fill_between(x, p_wind_ts[:end_idx], p_ee[:end_idx], alpha=0.5, label='PV', color='#f1c40f')
        ax1.axhline(y=p_nvp, color='red', linestyle='--', label=f'NVP ({p_nvp:.1f} MW)')
        ax1.set_ylabel('Leistung (MW)')
        ax1.set_title('Erzeugung und NVP-Kapazität')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(x, p_grid_no_storage[:end_idx], alpha=0.5, label='Netzeinspeisung', color='#2ecc71')
        ax2.fill_between(x, p_grid_no_storage[:end_idx], p_ee[:end_idx], alpha=0.5, label='Überschuss', color='#e74c3c')
        ax2.set_ylabel('Leistung (MW)')
        ax2.set_xlabel('Zeitschritt (15 min)')
        ax2.set_title('Netzeinspeisung und Überschuss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        p_ee_sorted = np.sort(p_ee)[::-1]
        hours = np.arange(len(p_ee_sorted)) * dt
        
        ax.fill_between(hours, p_ee_sorted, alpha=0.5, color='#3498db')
        ax.axhline(y=p_nvp, color='red', linestyle='--', label=f'NVP-Kapazität ({p_nvp:.1f} MW)')
        ax.fill_between(hours, np.minimum(p_ee_sorted, p_nvp), alpha=0.3, color='#2ecc71', label='Nutzbar')
        
        ax.set_xlabel('Stunden pro Jahr')
        ax.set_ylabel('Leistung (MW)')
        ax.set_title('Jahresdauerlinie der Erzeugung')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(p_surplus[p_surplus > 0], bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Überschussleistung (MW)')
        ax.set_ylabel('Häufigkeit')
        ax.set_title('Verteilung der Überschussleistung')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    # Speichern für nächsten Schritt
    st.session_state.surplus_analysis = {
        'total_generation': total_generation,
        'total_surplus': total_surplus,
        'surplus_share': surplus_share,
        'grid_utilization': grid_utilization,
    }
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col3:
        if st.button("Weiter →", type="primary", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()


def show_mode_c_step4():
    """Modus C - Schritt 4: Parameterstudie."""
    
    st.markdown("### Schritt 4: Speicherdimensionierung")
    
    # Auswahl: Automatische Optimierung oder Parameterstudie
    st.markdown("#### 🎯 Berechnungsmodus")
    calc_mode = st.radio(
        "Wie soll die optimale Speichergröße ermittelt werden?",
        ["Automatische Optimierung (empfohlen)", "Manuelle Parameterstudie"],
        horizontal=True,
        help="Die automatische Optimierung findet die optimale Speichergröße in einem Rechenlauf. "
             "Die Parameterstudie testet alle Kombinationen und zeigt eine Heatmap."
    )
    
    st.markdown("---")
    
    if calc_mode == "Automatische Optimierung (empfohlen)":
        # === NEUE AUTOMATISCHE OPTIMIERUNG ===
        st.markdown("""
        <div class="info-box">
            <strong>Automatische Optimierung mit PyPSA</strong><br>
            Das Tool findet in einem Rechenlauf die optimale Speichergröße, die:
            <ul>
                <li>Abregelung minimiert / Netzeinspeisung maximiert</li>
                <li>Speicherkosten berücksichtigt</li>
                <li>Die NVP-Grenze einhält</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Maximale Speichergröße")
            max_power_pct = st.slider("Max. Leistung (% von NVP)", 50, 200, 100,
                                      help="Obergrenze für die Optimierung")
            max_duration = st.slider("Max. Speicherdauer (h)", 4, 24, 8,
                                    help="Maximales E/P-Verhältnis")
        
        with col2:
            st.markdown("##### Kostenparameter (aus Schritt 1)")
            # Werte aus Session State holen
            capex_energy = st.session_state.get('cost_energy', 200)
            capex_power = st.session_state.get('cost_power', 80)
            
            st.info(f"**CAPEX Energie:** {capex_energy} €/kWh\n\n"
                   f"**CAPEX Leistung:** {capex_power} €/kW")
            
            if st.checkbox("Werte anpassen?", key="adjust_capex"):
                capex_energy = st.number_input("CAPEX Energie (€/kWh)", 100, 500, capex_energy)
                capex_power = st.number_input("CAPEX Leistung (€/kW)", 50, 200, capex_power)
        
        # Annuitätsparameter
        st.markdown("##### Wirtschaftliche Parameter")
        
        col1, col2 = st.columns(2)
        with col1:
            interest_rate = st.slider("Kalkulationszinssatz (%)", 1, 10, 5, 
                                     help="Zinssatz für die Annuitätenberechnung") / 100
            lifetime_years = st.slider("Speicher-Lebensdauer (Jahre)", 10, 25, 15,
                                      help="Wirtschaftliche Nutzungsdauer")
        with col2:
            opex_rate = st.slider("OPEX (% von CAPEX/Jahr)", 0, 5, 2,
                                 help="Jährliche Betriebskosten als Anteil der Investition") / 100
            
            # Marktwert der Einspeisung / Wert vermiedener Abregelung
            da_prices = st.session_state.get('da_prices_c')
            if da_prices is not None:
                avg_market_price = da_prices.mean()
                st.info(f"Ø Day-Ahead-Preis: {avg_market_price:.1f} €/MWh (aus Upload)")
                feed_in_value = avg_market_price
            else:
                feed_in_value = st.number_input(
                    "Marktwert Einspeisung (€/MWh)",
                    min_value=20,
                    max_value=200,
                    value=80,
                    help="Wert der vermiedenen Abregelung = Erlös pro eingespeister MWh"
                )
        
        # Annuitätenfaktor und Gesamtkosten anzeigen
        anf = (interest_rate * (1 + interest_rate) ** lifetime_years) / ((1 + interest_rate) ** lifetime_years - 1)
        annual_capex = capex_energy * anf
        annual_opex = capex_energy * opex_rate
        annual_total = annual_capex + annual_opex
        
        st.caption(f"**Jährliche Kosten pro kWh:** CAPEX {annual_capex:.1f} €/kWh/a + OPEX {annual_opex:.1f} €/kWh/a = **{annual_total:.1f} €/kWh/a**")
        st.caption(f"**Marktwert Einspeisung:** {feed_in_value:.0f} €/MWh")
        
        st.markdown("---")
        
        if st.button("🚀 Optimierung starten", type="primary", use_container_width=True):
            
            p_nvp = st.session_state.p_nvp_mw
            max_power_mw = p_nvp * max_power_pct / 100
            max_capacity_mwh = max_power_mw * max_duration
            
            st.info("⏳ PyPSA-Optimierung läuft... (ca. 30-60 Sekunden)")
            progress_bar = st.progress(0.3, text="Optimiere Speichergröße...")
            
            # Preisparameter abrufen
            da_prices = st.session_state.get('da_prices_c')
            
            result = optimize_nvp_storage(
                wind_profile=st.session_state.get('wind_profile_c'),
                pv_profile=st.session_state.get('pv_profile_c'),
                p_nvp_mw=p_nvp,
                p_wind_inst_mw=st.session_state.p_wind_inst_mw,
                p_pv_inst_mw=st.session_state.p_pv_inst_mw,
                max_storage_power_mw=max_power_mw,
                max_storage_capacity_mwh=max_capacity_mwh,
                eta_charge=st.session_state.eta_charge,
                eta_discharge=st.session_state.eta_discharge,
                da_prices=da_prices,
                capex_power=capex_power,
                capex_energy=capex_energy,
                interest_rate=interest_rate,
                lifetime_years=lifetime_years,
                opex_rate=opex_rate,
                feed_in_value=feed_in_value,
            )
            
            progress_bar.progress(1.0, text="✅ Fertig!")
            
            if result is None:
                st.error("❌ Optimierung fehlgeschlagen. Bitte Parameterstudie verwenden.")
            elif result['optimal_power_mw'] < 0.001:
                # Spezialfall: Optimum ist 0 Speicher
                st.warning("⚠️ **Ergebnis: Kein Speicher wirtschaftlich sinnvoll**")
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>Warum?</strong><br>
                    Bei nur <strong>{result['total_surplus_mwh']:.0f} MWh</strong> Überschuss pro Jahr 
                    übersteigen die Speicherkosten den Nutzen.<br><br>
                    <strong>Jährliche Kapitalkosten:</strong> {result.get('annual_cost_energy', 0):.0f} €/MWh/Jahr<br>
                    <strong>Überschuss ohne Speicher:</strong> {result.get('ref_curtailment_mwh', result['total_curtailment_mwh']):.0f} MWh/Jahr<br>
                    <strong>Netzauslastung ohne Speicher:</strong> {result['grid_utilization']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                st.info("💡 **Optionen:**\n"
                       "- Niedrigere CAPEX-Annahmen testen\n"
                       "- Längere Lebensdauer oder niedrigeren Zinssatz wählen\n"
                       "- Manuelle Parameterstudie durchführen\n"
                       "- Höhere Überbauung (mehr installierte Leistung) prüfen")
                
                # Trotzdem speichern für Konsistenz
                st.session_state.optimal_nvp_result = result
            else:
                st.success("✅ Optimale Speichergröße gefunden!")
                
                # Ergebnisse anzeigen
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Optimale Leistung", f"{result['optimal_power_mw']*1000:.0f} kW",
                             delta=f"{result['optimal_power_mw']/p_nvp*100:.0f}% von NVP")
                with col2:
                    st.metric("Optimale Kapazität", f"{result['optimal_capacity_mwh']*1000:.0f} kWh")
                with col3:
                    st.metric("E/P-Verhältnis", f"{result['ep_ratio']:.1f} h")
                with col4:
                    st.metric("Erfassungsgrad", f"{result['capture_rate']*100:.1f}%")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Netzauslastung", f"{result['grid_utilization']*100:.1f}%")
                with col2:
                    st.metric("Abregelung", f"{result['total_curtailment_mwh']:.0f} MWh/Jahr")
                with col3:
                    st.metric("Zyklen/Jahr", f"{result['cycles']:.0f}")
                
                # CAPEX berechnen
                total_capex = (result['optimal_capacity_mwh'] * 1000 * capex_energy + 
                              result['optimal_power_mw'] * 1000 * capex_power)
                st.info(f"💰 Geschätzte Investitionskosten: **{total_capex:,.0f} €**")
                
                # In Session State speichern für Schritt 5
                st.session_state.optimal_nvp_result = result
                st.session_state.nvp_capex_params = {'energy': capex_energy, 'power': capex_power}
                
                # Auch als "beste Konfiguration" für Kompatibilität
                st.session_state.selected_config = {
                    'power_ratio': result['optimal_power_mw'] / p_nvp,
                    'duration_h': result['ep_ratio'],
                    'storage_power_mw': result['optimal_power_mw'],
                    'storage_capacity_mwh': result['optimal_capacity_mwh'],
                    'capture_rate': result['capture_rate'],
                    'grid_utilization': result['grid_utilization'],
                }
        
        st.markdown("---")
        
        # Weiter Button
        if st.session_state.get('optimal_nvp_result') is not None:
            if st.button("Weiter zu Ergebnissen →", type="primary", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()
    
    else:
        # === MANUELLE PARAMETERSTUDIE (wie bisher) ===
        st.markdown("""
        <div class="info-box">
            Das Tool variiert systematisch Speicherleistung und -kapazität und berechnet 
            für jede Kombination den Erfassungsgrad (wie viel Überschuss genutzt wird) 
            und die resultierende Netzauslastung.
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter für Studie
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Speicherleistung")
            power_min = st.slider("Min. Leistung (% von NVP)", 0, 50, 10)
            power_max = st.slider("Max. Leistung (% von NVP)", 50, 150, 100)
            power_steps = st.number_input("Anzahl Schritte (Leistung)", 3, 10, 5,
                                         help="Weniger Schritte = schnellere Berechnung")
        
        with col2:
            st.markdown("#### Speicherdauer (E/P)")
            duration_min = st.slider("Min. Dauer (h)", 1, 5, 1)
            duration_max = st.slider("Max. Dauer (h)", 5, 20, 8)
            duration_steps = st.number_input("Anzahl Schritte (Dauer)", 3, 10, 4,
                                            help="Weniger Schritte = schnellere Berechnung")
        
        total_combinations = int(power_steps) * int(duration_steps)
        st.warning(f"⚠️ {total_combinations} Kombinationen = ca. {total_combinations * 30 // 60} Minuten Rechenzeit")
        
        st.markdown("---")
        
        if st.button("🔍 Parameterstudie starten", type="primary", use_container_width=True):
            
            power_range = np.linspace(power_min/100, power_max/100, int(power_steps))
            duration_range = np.linspace(duration_min, duration_max, int(duration_steps))
            
            # Preisparameter abrufen
            da_prices = st.session_state.get('da_prices_c')
            discharge_strategy = st.session_state.get('discharge_strategy_c_val', 'immediate')
            price_threshold = st.session_state.get('price_threshold_c_val')
            
            # Fortschrittsanzeige
            progress_bar = st.progress(0, text="Starte Parameterstudie...")
            status_text = st.empty()
            
            results_list = []
            total = len(power_range) * len(duration_range)
            current = 0
            
            for power_ratio in power_range:
                for duration in duration_range:
                    current += 1
                    progress_bar.progress(current / total, text=f"Berechne {current}/{total}...")
                    status_text.text(f"Leistung: {power_ratio*100:.0f}%, Dauer: {duration:.0f}h")
                    
                    storage_power = st.session_state.p_nvp_mw * power_ratio
                    storage_capacity = storage_power * duration
                    
                    sim = simulate_nvp_storage(
                        wind_profile=st.session_state.get('wind_profile_c'),
                        pv_profile=st.session_state.get('pv_profile_c'),
                        p_nvp_mw=st.session_state.p_nvp_mw,
                        p_wind_inst_mw=st.session_state.p_wind_inst_mw,
                        p_pv_inst_mw=st.session_state.p_pv_inst_mw,
                        storage_power_mw=storage_power,
                        storage_capacity_mwh=storage_capacity,
                        eta_charge=st.session_state.eta_charge,
                        eta_discharge=st.session_state.eta_discharge,
                        da_prices=da_prices,
                        price_threshold=price_threshold,
                        discharge_strategy=discharge_strategy,
                    )
                    
                    result_entry = {
                        'power_ratio': power_ratio,
                        'duration_h': duration,
                        'storage_power_mw': storage_power,
                        'storage_capacity_mwh': storage_capacity,
                        'capture_rate': sim['capture_rate'],
                        'grid_utilization': sim['grid_utilization'],
                        'curtailment_mwh': sim['total_curtailment_mwh'],
                        'surplus_mwh': sim['total_surplus_mwh'],
                        'cycles': sim['cycles'],
                        'discharged_mwh': sim['total_discharged_mwh'],
                    }
                    
                    if da_prices is not None:
                        result_entry['discharge_revenue'] = sim.get('total_discharge_revenue', 0)
                        result_entry['avg_discharge_price'] = sim.get('avg_discharge_price')
                    
                    results_list.append(result_entry)
            
            progress_bar.progress(1.0, text="✅ Fertig!")
            status_text.empty()
            
            results = pd.DataFrame(results_list)
            st.session_state.parameter_study_results = results
            
            st.success("✅ Parameterstudie abgeschlossen!")
            
            # Info zur Entladestrategie
            if discharge_strategy != "immediate" and da_prices is not None:
                if discharge_strategy == "price_threshold":
                    st.info(f"💰 Preisoptimierte Entladung aktiv: Schwellwert {price_threshold:.0f} €/MWh")
                else:
                    st.info(f"💰 Preisoptimierte Entladung aktiv: Obere {100-price_threshold:.0f}% der Tagespreise")
            
            # Heatmap des Erfassungsgrades
            st.markdown("#### 📊 Erfassungsgrad (Anteil genutzter Überschuss)")
            
            # Pivot-Tabelle für Heatmap
            pivot = results.pivot_table(
                values='capture_rate', 
                index='duration_h', 
                columns='power_ratio',
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            im = ax.imshow(pivot.values * 100, aspect='auto', cmap='RdYlGn', 
                          origin='lower', vmin=0, vmax=100)
            
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f'{x*100:.0f}%' for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f'{y:.0f}h' for y in pivot.index])
            
            ax.set_xlabel('Speicherleistung (% von NVP)')
            ax.set_ylabel('Speicherdauer (E/P)')
            ax.set_title('Erfassungsgrad des Überschusses (%)')
            
            plt.colorbar(im, ax=ax, label='Erfassungsgrad (%)')
            
            # Werte in Zellen anzeigen
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    text = ax.text(j, i, f'{pivot.values[i, j]*100:.0f}%',
                                  ha='center', va='center', color='black', fontsize=8)
            
            st.pyplot(fig)
            plt.close()
            
            # Tabelle mit besten Konfigurationen
            st.markdown("#### 🏆 Top 5 Konfigurationen (nach Erfassungsgrad)")
            
            top5 = results.nlargest(5, 'capture_rate')[
                ['storage_power_mw', 'storage_capacity_mwh', 'duration_h', 
                 'capture_rate', 'grid_utilization', 'cycles']
            ].copy()
            
            top5.columns = ['Leistung (MW)', 'Kapazität (MWh)', 'Dauer (h)', 
                           'Erfassungsgrad', 'Netzauslastung', 'Zyklen/Jahr']
            top5['Erfassungsgrad'] = top5['Erfassungsgrad'].apply(lambda x: f'{x*100:.1f}%')
            top5['Netzauslastung'] = top5['Netzauslastung'].apply(lambda x: f'{x*100:.1f}%')
            top5['Zyklen/Jahr'] = top5['Zyklen/Jahr'].apply(lambda x: f'{x:.0f}')
            
            st.dataframe(top5, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col3:
        # Weiter wenn entweder Parameterstudie ODER Optimierung durchgeführt
        has_results = (st.session_state.get('parameter_study_results') is not None or 
                      st.session_state.get('optimal_nvp_result') is not None)
        if has_results:
            if st.button("Weiter →", type="primary", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()


def show_mode_c_step5():
    """Modus C - Schritt 5: Ergebnisse und Wirtschaftlichkeit."""
    
    st.markdown("### Schritt 5: Ergebnisse und Wirtschaftlichkeit")
    
    surplus_analysis = st.session_state.surplus_analysis
    
    # Prüfen welche Ergebnisse vorhanden sind
    has_optimization_result = st.session_state.get('optimal_nvp_result') is not None
    has_parameter_study = st.session_state.get('parameter_study_results') is not None
    
    if not has_optimization_result and not has_parameter_study:
        st.warning("⚠️ Keine Ergebnisse vorhanden. Bitte zuerst Schritt 4 durchführen.")
        if st.button("← Zurück zu Schritt 4"):
            st.session_state.current_step = 4
            st.rerun()
        return
    
    # === FALL 1: Automatische Optimierung wurde verwendet ===
    if has_optimization_result and not has_parameter_study:
        opt_result = st.session_state.optimal_nvp_result
        
        # Wenn Optimierung 0 ergab, speziellen Hinweis zeigen
        if opt_result['optimal_power_mw'] < 0.001:
            st.warning("⚠️ **Die Optimierung ergab: Kein Speicher wirtschaftlich sinnvoll**")
            st.markdown(f"""
            Bei nur **{opt_result['total_surplus_mwh']:.0f} MWh** Überschuss pro Jahr 
            übersteigen die Speicherkosten den Nutzen.
            
            **Ohne Speicher:**
            - Netzeinspeisung: {opt_result.get('ref_grid_feed_mwh', 0):,.0f} MWh/Jahr
            - Abregelung: {opt_result.get('ref_curtailment_mwh', 0):,.0f} MWh/Jahr
            - Netzauslastung: {opt_result['grid_utilization']*100:.1f}%
            """)
            
            if st.button("← Zurück zu Schritt 4 (andere Parameter testen)"):
                st.session_state.current_step = 4
                st.rerun()
            return
        
        # Ergebnisse aus Optimierung anzeigen
        st.markdown("#### 📊 Optimale Speicherkonfiguration (aus automatischer Optimierung)")
        
        unit = st.session_state.get('unit_display', 'MW')
        cost_power = st.session_state.get('cost_power', 80)
        cost_energy = st.session_state.get('cost_energy', 200)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if unit == "kW":
                st.metric("Speicherleistung", f"{opt_result['optimal_power_mw']*1000:.0f} kW")
                st.metric("Speicherkapazität", f"{opt_result['optimal_capacity_mwh']*1000:.0f} kWh")
            else:
                st.metric("Speicherleistung", f"{opt_result['optimal_power_mw']:.2f} MW")
                st.metric("Speicherkapazität", f"{opt_result['optimal_capacity_mwh']:.1f} MWh")
        
        with col2:
            st.metric("Speicherdauer (E/P)", f"{opt_result['ep_ratio']:.1f} h")
            power_ratio = opt_result['optimal_power_mw'] / st.session_state.p_nvp_mw * 100
            st.metric("Leistungsverhältnis", f"{power_ratio:.0f}% von NVP")
        
        with col3:
            st.metric("Erfassungsgrad", f"{opt_result['capture_rate']*100:.1f}%")
            st.metric("Netzauslastung", f"{opt_result['grid_utilization']*100:.1f}%")
        
        with col4:
            st.metric("Zyklen pro Jahr", f"{opt_result['cycles']:.0f}")
            capex = (opt_result['optimal_capacity_mwh'] * 1000 * cost_energy + 
                    opt_result['optimal_power_mw'] * 1000 * cost_power)
            st.metric("Investitionskosten", f"{capex:,.0f} €")
        
        st.markdown("---")
        
        # Detaillierte Simulation - Ergebnisse aus Optimierung verwenden
        st.markdown("#### 📈 Detaillierte Ergebnisse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Ohne Speicher")
            ref_grid = opt_result.get('ref_grid_feed_mwh', surplus_analysis['total_generation'] - surplus_analysis['total_surplus'])
            ref_curtail = opt_result.get('ref_curtailment_mwh', surplus_analysis['total_surplus'])
            st.write(f"- Netzeinspeisung: {ref_grid:,.0f} MWh")
            st.write(f"- Abregelung: {ref_curtail:,.0f} MWh ({ref_curtail/surplus_analysis['total_generation']*100:.1f}%)")
            st.write(f"- Netzauslastung: {surplus_analysis['grid_utilization']:.1f}%")
        
        with col2:
            st.markdown("##### Mit Speicher")
            st.write(f"- Netzeinspeisung: {opt_result['total_grid_feed_mwh']:,.0f} MWh")
            curtail_pct = opt_result['total_curtailment_mwh'] / opt_result['total_generation_mwh'] * 100 if opt_result['total_generation_mwh'] > 0 else 0
            st.write(f"- Abregelung: {opt_result['total_curtailment_mwh']:,.0f} MWh ({curtail_pct:.1f}%)")
            st.write(f"- Netzauslastung: {opt_result['grid_utilization']*100:.1f}%")
        
        # Verbesserung
        improvement = opt_result['total_grid_feed_mwh'] - ref_grid
        if improvement > 0:
            st.success(f"✓ Zusätzliche Einspeisung durch Speicher: {improvement:,.0f} MWh/Jahr")
        else:
            st.warning(f"⚠️ Keine Verbesserung durch Speicher")
        
        st.markdown("---")
        
        # Wirtschaftlichkeitsrechnung
        st.markdown("#### 💰 Wirtschaftlichkeitsrechnung")
        
        # Parameter aus Optimierung
        feed_in_value = opt_result.get('feed_in_value', 80)
        annual_cost_energy = opt_result.get('annual_cost_energy', 0)
        annual_cost_power = opt_result.get('annual_cost_power', 0)
        opex_rate = opt_result.get('opex_rate', 0.02)
        
        # Berechnung
        additional_revenue = improvement * feed_in_value  # Zusätzlicher Erlös durch vermiedene Abregelung
        annual_storage_cost = (opt_result['optimal_capacity_mwh'] * annual_cost_energy / 1000 + 
                              opt_result['optimal_power_mw'] * annual_cost_power / 1000)  # €/Jahr
        
        net_benefit = additional_revenue - annual_storage_cost
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Erlöse (pro Jahr)**")
            st.write(f"- Zusätzliche Einspeisung: {improvement:,.0f} MWh")
            st.write(f"- Marktwert: {feed_in_value:.0f} €/MWh")
            st.metric("Zusätzlicher Erlös", f"{additional_revenue:,.0f} €/a")
        
        with col2:
            st.markdown("**Kosten (pro Jahr)**")
            capex_share = annual_cost_energy / (1 + opex_rate) if opex_rate > 0 else annual_cost_energy
            opex_share = annual_cost_energy * opex_rate / (1 + opex_rate) if opex_rate > 0 else 0
            st.write(f"- CAPEX (annualisiert): {opt_result.get('annual_capex_energy', capex_share):.0f} €/MWh/a")
            st.write(f"- OPEX ({opex_rate*100:.0f}% v. CAPEX): {opt_result.get('annual_opex_energy', opex_share):.0f} €/MWh/a")
            st.metric("Jährliche Speicherkosten", f"{annual_storage_cost:,.0f} €/a")
        
        with col3:
            st.markdown("**Ergebnis**")
            if net_benefit > 0:
                st.metric("Jährlicher Nettonutzen", f"+{net_benefit:,.0f} €/a", delta="wirtschaftlich")
            else:
                st.metric("Jährlicher Nettonutzen", f"{net_benefit:,.0f} €/a", delta="unwirtschaftlich", delta_color="inverse")
            
            # Amortisation
            capex_total = (opt_result['optimal_capacity_mwh'] * 1000 * cost_energy + 
                          opt_result['optimal_power_mw'] * 1000 * cost_power)
            if additional_revenue > 0:
                simple_payback = capex_total / additional_revenue
                st.write(f"Einfache Amortisation: {simple_payback:.1f} Jahre")
        
        st.markdown("---")
        
        # Zeitreihen visualisieren
        st.markdown("#### 📊 Zeitreihen-Visualisierung")
        
        n_total = len(opt_result['p_ee'])
        n_days_total = n_total // 96  # Gesamte Tage
        
        # Anzeigemodus wählen
        col1, col2 = st.columns(2)
        
        with col1:
            view_mode = st.radio(
                "Ansicht",
                ["Ganzes Jahr (Übersicht)", "Detailansicht (Woche)"],
                horizontal=True,
                key="view_mode_c5"
            )
        
        with col2:
            if view_mode == "Detailansicht (Woche)":
                # Automatisch interessanteste Woche finden
                if st.button("🔍 Interessanteste Woche finden", help="Findet die Woche mit höchster Speicheraktivität"):
                    # Finde Woche mit meister Speicheraktivität
                    p_charge = opt_result['p_charge']
                    weekly_activity = []
                    for week_start in range(0, n_total - 7*96, 96):  # Pro Tag
                        week_end = min(week_start + 7*96, n_total)
                        activity = np.sum(p_charge[week_start:week_end])
                        weekly_activity.append((week_start // 96, activity))
                    
                    if weekly_activity:
                        best_day = max(weekly_activity, key=lambda x: x[1])[0]
                        st.session_state.start_day_c5 = best_day
                        st.rerun()
        
        if view_mode == "Ganzes Jahr (Übersicht)":
            # Ganzes Jahr anzeigen - aggregiert auf Tageswerte für bessere Übersicht
            st.info("📅 Darstellung: Tägliche Mittelwerte über das gesamte Jahr")
            
            # Auf Tageswerte aggregieren
            n_days = n_total // 96
            daily_ee = np.array([opt_result['p_ee'][i*96:(i+1)*96].mean() for i in range(n_days)])
            daily_grid = np.array([opt_result['p_grid'][i*96:(i+1)*96].mean() for i in range(n_days)])
            daily_charge = np.array([opt_result['p_charge'][i*96:(i+1)*96].sum() * 0.25 for i in range(n_days)])  # MWh/Tag
            daily_discharge = np.array([opt_result['p_discharge'][i*96:(i+1)*96].sum() * 0.25 for i in range(n_days)])
            daily_soc_max = np.array([opt_result['soc'][i*96:(i+1)*96].max() for i in range(n_days)])
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            ax1, ax2, ax3 = axes
            
            x = range(n_days)
            months = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
            
            # Erzeugung und Netzeinspeisung (Tagesmittel)
            ax1.fill_between(x, daily_ee, alpha=0.3, label='Ø Erzeugung', color='#3498db')
            ax1.plot(x, daily_grid, label='Ø Netzeinspeisung', color='#2ecc71', linewidth=1)
            ax1.axhline(y=st.session_state.p_nvp_mw, color='red', linestyle='--', label='NVP-Kapazität', alpha=0.7)
            ax1.set_ylabel('Leistung (MW)')
            ax1.set_title('Erzeugung und Netzeinspeisung (Tagesmittel)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Speicherenergie pro Tag
            ax2.bar(x, daily_charge, alpha=0.7, label='Geladen (MWh/Tag)', color='#e74c3c', width=1)
            ax2.bar(x, -daily_discharge, alpha=0.7, label='Entladen (MWh/Tag)', color='#2ecc71', width=1)
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.set_ylabel('Energie (MWh/Tag)')
            ax2.set_title('Tägliche Speichernutzung')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # SOC Maximum pro Tag
            ax3.fill_between(x, daily_soc_max, alpha=0.5, color='#9b59b6')
            ax3.plot(x, daily_soc_max, color='#9b59b6', linewidth=1)
            ax3.set_ylabel('Max. SOC (MWh)')
            ax3.set_xlabel('Tag des Jahres')
            ax3.set_title('Maximaler Speicherfüllstand pro Tag')
            ax3.grid(True, alpha=0.3)
            
            # X-Achse: Monate markieren
            month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            ax3.set_xticks(month_starts)
            ax3.set_xticklabels(months)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        else:
            # Detailansicht: Wochenweise
            col1, col2 = st.columns([3, 1])
            
            with col1:
                start_day = st.slider(
                    "Starttag im Jahr",
                    min_value=1,
                    max_value=max(1, n_days_total - 7),
                    value=st.session_state.get('start_day_c5', 150),  # Default: ca. Juni
                    key="start_day_slider_c5",
                    help="Tag 1 = 1. Januar, Tag 180 ≈ Ende Juni"
                )
            
            with col2:
                days_to_show = st.selectbox(
                    "Anzeigetage",
                    [3, 7, 14, 30],
                    index=1,
                    key="days_c5_opt"
                )
            
            start_idx = (start_day - 1) * 96
            end_idx = min(start_idx + days_to_show * 96, n_total)
            
            # Monat berechnen für Titel
            month_names = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 
                          'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']
            approx_month = min(11, start_day // 30)
            st.caption(f"📅 Angezeigt: Tag {start_day} - {start_day + days_to_show} (ca. {month_names[approx_month]})")
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            ax1, ax2, ax3 = axes
            
            x = range(end_idx - start_idx)
            
            # Erzeugung und Netzeinspeisung
            ax1.fill_between(x, opt_result['p_ee'][start_idx:end_idx], alpha=0.3, label='Erzeugung', color='#3498db')
            ax1.plot(x, opt_result['p_grid'][start_idx:end_idx], label='Netzeinspeisung', color='#2ecc71', linewidth=1.5)
            ax1.axhline(y=st.session_state.p_nvp_mw, color='red', linestyle='--', label='NVP-Kapazität')
            ax1.set_ylabel('Leistung (MW)')
            ax1.set_title('Erzeugung und Netzeinspeisung')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Speicherleistung
            ax2.fill_between(x, opt_result['p_charge'][start_idx:end_idx], alpha=0.5, label='Laden', color='#e74c3c')
            ax2.fill_between(x, -opt_result['p_discharge'][start_idx:end_idx], alpha=0.5, label='Entladen', color='#2ecc71')
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.set_ylabel('Speicherleistung (MW)')
            ax2.set_title('Speicherbetrieb')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # SOC
            ax3.plot(x, opt_result['soc'][start_idx:end_idx], color='#9b59b6', linewidth=1.5)
            ax3.set_ylabel('SOC (MWh)')
            ax3.set_xlabel(f'Zeitschritt (15 min) - Start: Tag {start_day}')
            ax3.set_title('Speicherfüllstand')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Optional: Parameterstudie nachträglich durchführen
        st.markdown("---")
        st.info("💡 **Tipp:** Möchten Sie verschiedene Speichergrößen vergleichen? "
               "Gehen Sie zurück zu Schritt 4 und wählen Sie 'Manuelle Parameterstudie'.")
        
        return
    
    # === FALL 2: Parameterstudie wurde verwendet ===
    results = st.session_state.parameter_study_results
    
    # Optimale Konfiguration auswählen
    st.markdown("#### 🎯 Speicherkonfiguration auswählen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Slider für Ziel-Erfassungsgrad
        target_capture = st.slider(
            "Ziel-Erfassungsgrad (%)",
            min_value=50,
            max_value=100,
            value=90,
            help="Wie viel Prozent des Überschusses soll der Speicher erfassen?"
        ) / 100
    
    with col2:
        # Finde kostenoptimale Konfiguration für diesen Erfassungsgrad
        cost_power = st.session_state.cost_power
        cost_energy = st.session_state.cost_energy
        
        results['capex'] = (results['storage_power_mw'] * 1000 * cost_power + 
                           results['storage_capacity_mwh'] * 1000 * cost_energy)
        
        # Filtere Konfigurationen mit mindestens Ziel-Erfassungsgrad
        valid = results[results['capture_rate'] >= target_capture]
        
        if len(valid) > 0:
            optimal = valid.loc[valid['capex'].idxmin()]
            st.success(f"✓ Kostenoptimale Konfiguration für ≥{target_capture*100:.0f}% Erfassungsgrad gefunden")
        else:
            optimal = results.loc[results['capture_rate'].idxmax()]
            st.warning(f"⚠️ Ziel nicht erreichbar. Zeige beste verfügbare Konfiguration.")
    
    st.markdown("---")
    
    # Ergebnisse anzeigen
    st.markdown("#### 📊 Optimale Speicherkonfiguration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    unit = st.session_state.get('unit_display', 'MW')
    
    with col1:
        if unit == "kW":
            st.metric("Speicherleistung", f"{optimal['storage_power_mw']*1000:.0f} kW")
            st.metric("Speicherkapazität", f"{optimal['storage_capacity_mwh']*1000:.0f} kWh")
        else:
            st.metric("Speicherleistung", f"{optimal['storage_power_mw']:.2f} MW")
            st.metric("Speicherkapazität", f"{optimal['storage_capacity_mwh']:.1f} MWh")
    
    with col2:
        st.metric("Speicherdauer (E/P)", f"{optimal['duration_h']:.1f} h")
        st.metric("Leistungsverhältnis", f"{optimal['power_ratio']*100:.0f}% von NVP")
    
    with col3:
        st.metric("Erfassungsgrad", f"{optimal['capture_rate']*100:.1f}%")
        st.metric("Netzauslastung", f"{optimal['grid_utilization']*100:.1f}%")
    
    with col4:
        st.metric("Zyklen pro Jahr", f"{optimal['cycles']:.0f}")
        st.metric("Investitionskosten", f"{optimal['capex']:,.0f} €")
    
    st.markdown("---")
    
    # Simulation mit optimaler Konfiguration
    st.markdown("#### 📈 Detaillierte Simulation")
    
    # Preisparameter abrufen
    da_prices = st.session_state.get('da_prices_c')
    discharge_strategy = st.session_state.get('discharge_strategy_c_val', 'immediate')
    price_threshold = st.session_state.get('price_threshold_c_val')
    
    sim = simulate_nvp_storage(
        wind_profile=st.session_state.get('wind_profile_c'),
        pv_profile=st.session_state.get('pv_profile_c'),
        p_nvp_mw=st.session_state.p_nvp_mw,
        p_wind_inst_mw=st.session_state.p_wind_inst_mw,
        p_pv_inst_mw=st.session_state.p_pv_inst_mw,
        storage_power_mw=optimal['storage_power_mw'],
        storage_capacity_mwh=optimal['storage_capacity_mwh'],
        eta_charge=st.session_state.eta_charge,
        eta_discharge=st.session_state.eta_discharge,
        da_prices=da_prices,
        price_threshold=price_threshold,
        discharge_strategy=discharge_strategy,
    )
    
    # Vergleichstabelle
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Ohne Speicher")
        st.write(f"- Netzeinspeisung: {surplus_analysis['total_generation'] - surplus_analysis['total_surplus']:,.0f} MWh")
        st.write(f"- Abregelung: {surplus_analysis['total_surplus']:,.0f} MWh ({surplus_analysis['surplus_share']:.1f}%)")
        st.write(f"- Netzauslastung: {surplus_analysis['grid_utilization']:.1f}%")
    
    with col2:
        st.markdown("##### Mit Speicher")
        st.write(f"- Netzeinspeisung: {sim['total_grid_feed_mwh']:,.0f} MWh")
        st.write(f"- Abregelung: {sim['total_curtailment_mwh']:,.0f} MWh ({sim['total_curtailment_mwh']/sim['total_generation_mwh']*100:.1f}%)")
        st.write(f"- Netzauslastung: {sim['grid_utilization']*100:.1f}%")
        
        # Erlöse anzeigen wenn Preisoptimierung aktiv
        if da_prices is not None and sim['total_discharge_revenue'] > 0:
            st.write(f"- **Entladeerlös: {sim['total_discharge_revenue']:,.0f} €/Jahr**")
            if sim['avg_discharge_price'] is not None:
                st.write(f"- Ø Entladepreis: {sim['avg_discharge_price']:.1f} €/MWh")
    
    # Verbesserung
    improvement = sim['total_grid_feed_mwh'] - (surplus_analysis['total_generation'] - surplus_analysis['total_surplus'])
    st.success(f"✓ Zusätzliche Einspeisung durch Speicher: {improvement:,.0f} MWh/Jahr")
    
    # Entladestrategie-Info anzeigen
    if discharge_strategy != "immediate" and da_prices is not None:
        if discharge_strategy == "price_threshold":
            st.info(f"💰 Preisoptimierte Entladung: Nur bei Preisen ≥ {price_threshold:.0f} €/MWh")
        else:
            st.info(f"💰 Preisoptimierte Entladung: Nur im oberen {100-price_threshold:.0f}% Preisbereich des Tages")
    
    st.markdown("---")
    
    # Beispielwoche visualisieren
    st.markdown("#### 📊 Beispielwoche mit Speicherbetrieb")
    
    days_to_show = st.slider("Anzeigetage", 1, 14, 7, key="days_c5")
    end_idx = min(days_to_show * 96, len(sim['p_ee']))
    
    fig, axes = plt.subplots(4 if da_prices is not None else 3, 1, figsize=(14, 12 if da_prices is not None else 10), sharex=True)
    
    if da_prices is not None:
        ax1, ax2, ax3, ax4 = axes
    else:
        ax1, ax2, ax3 = axes
    
    x = range(end_idx)
    
    # Erzeugung und Netzeinspeisung
    ax1.fill_between(x, sim['p_ee'][:end_idx], alpha=0.3, label='Erzeugung', color='#3498db')
    ax1.plot(x, sim['p_grid'][:end_idx], label='Netzeinspeisung', color='#2ecc71', linewidth=1.5)
    ax1.axhline(y=st.session_state.p_nvp_mw, color='red', linestyle='--', label='NVP-Kapazität')
    ax1.set_ylabel('Leistung (MW)')
    ax1.set_title('Erzeugung und Netzeinspeisung')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speicherleistung
    ax2.fill_between(x, sim['p_charge'][:end_idx], alpha=0.5, label='Laden', color='#e74c3c')
    ax2.fill_between(x, -sim['p_discharge'][:end_idx], alpha=0.5, label='Entladen', color='#2ecc71')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Speicherleistung (MW)')
    ax2.set_title('Speicherbetrieb')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ladezustand
    soc_percent = sim['soc'][:end_idx] / optimal['storage_capacity_mwh'] * 100
    ax3.fill_between(x, soc_percent, alpha=0.5, color='#9b59b6')
    ax3.plot(x, soc_percent, color='#9b59b6', linewidth=1)
    ax3.set_ylabel('Ladezustand (%)')
    ax3.set_title('Speicher-Ladezustand')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # Preisplot hinzufügen wenn Preise vorhanden
    if da_prices is not None:
        prices_plot = sim['prices'][:end_idx] if sim['prices'] is not None else da_prices.values[:end_idx]
        ax4.plot(x, prices_plot, color='#f39c12', linewidth=1, label='Day-Ahead-Preis')
        
        # Schwellwert anzeigen
        if discharge_strategy == "price_threshold" and price_threshold is not None:
            ax4.axhline(y=price_threshold, color='#e74c3c', linestyle='--', 
                       label=f'Schwellwert ({price_threshold:.0f} €/MWh)')
        
        # Entladezeitpunkte markieren
        discharge_mask = sim['p_discharge'][:end_idx] > 0
        if discharge_mask.any():
            ax4.fill_between(x, 0, prices_plot, where=discharge_mask, 
                           alpha=0.3, color='#2ecc71', label='Entladung')
        
        ax4.set_ylabel('Preis (€/MWh)')
        ax4.set_xlabel('Zeitschritt (15 min)')
        ax4.set_title('Day-Ahead-Preise und Entladezeitpunkte')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax3.set_xlabel('Zeitschritt (15 min)')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Export
    st.markdown("#### 💾 Ergebnisse exportieren")
    
    export_data = {
        "konfiguration": {
            "nvp_leistung_mw": st.session_state.p_nvp_mw,
            "wind_installiert_mw": st.session_state.p_wind_inst_mw,
            "pv_installiert_mw": st.session_state.p_pv_inst_mw,
            "ueberbauungsfaktor": (st.session_state.p_wind_inst_mw + st.session_state.p_pv_inst_mw) / st.session_state.p_nvp_mw,
        },
        "speicher": {
            "leistung_mw": optimal['storage_power_mw'],
            "kapazitaet_mwh": optimal['storage_capacity_mwh'],
            "dauer_h": optimal['duration_h'],
            "investitionskosten_eur": optimal['capex'],
        },
        "ergebnisse": {
            "erfassungsgrad": optimal['capture_rate'],
            "netzauslastung": sim['grid_utilization'],
            "zyklen_pro_jahr": optimal['cycles'],
            "zusaetzliche_einspeisung_mwh": improvement,
        },
    }
    
    # Erlösdaten hinzufügen wenn Preise vorhanden
    if da_prices is not None:
        export_data["entladestrategie"] = {
            "strategie": discharge_strategy,
            "schwellwert": price_threshold,
        }
        export_data["erloese"] = {
            "entladeerlos_eur": sim['total_discharge_revenue'],
            "durchschnittlicher_entladepreis_eur_mwh": sim['avg_discharge_price'],
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Ergebnisse als JSON",
            json_str,
            "nvp_ueberbauung_ergebnis.json",
            "application/json",
            use_container_width=True
        )
    
    with col2:
        csv_buffer = io.StringIO()
        results.to_csv(csv_buffer, index=False, sep=';', decimal=',')
        st.download_button(
            "📥 Parameterstudie als CSV",
            csv_buffer.getvalue(),
            "parameterstudie_ergebnisse.csv",
            "text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Zurück", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()
    with col3:
        if st.button("🏠 Neues Projekt", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# =============================================================================
# Hauptprogramm
# =============================================================================
def main():
    """Hauptfunktion der Anwendung."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔋 Batteriespeicher-Tool")
        st.markdown("---")
        
        if st.session_state.current_mode:
            mode_names = {
                'A': 'A - Wirtschaftlichkeit',
                'B': 'B - Peak Shaving',
                'C': 'C - NVP-Überbauung',
            }
            st.markdown(f"**Aktiver Modus:** {mode_names.get(st.session_state.current_mode, '')}")
            st.markdown(f"**Schritt:** {st.session_state.current_step}")
            st.markdown("---")
        
        st.markdown("#### ℹ️ Hilfe")
        st.markdown("""
        Bei Fragen oder Problemen:
        - Prüfen Sie das Dateiformat (CSV)
        - Achten Sie auf die Einheiten
        - Starten Sie bei Bedarf neu
        """)
        
        if st.session_state.current_mode:
            if st.button("🔄 Neu starten"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Hauptbereich
    if st.session_state.current_mode is None:
        show_start_page()
    elif st.session_state.current_mode == 'A':
        show_mode_a()
    elif st.session_state.current_mode == 'B':
        show_mode_b()
    elif st.session_state.current_mode == 'C':
        show_mode_c()


if __name__ == "__main__":
    main()
