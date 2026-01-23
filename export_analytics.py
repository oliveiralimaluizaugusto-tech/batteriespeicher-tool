"""
Export und Analyse-Modul für das Batteriespeicher-Optimierungstool.

Dieses Modul stellt Funktionen bereit für:
1. CSV-Export der Parameterstudie (Rohdaten)
2. Zeitreihen-Export (15-min oder stündlich)
3. Überschuss-Histogramm mit Leistungsklassen
4. Saisonale Auswertung (Winter/Sommer)

Version: 1.0.0
Autor: Batteriespeicher-Tool
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import json
import hashlib

# Tool-Version für Metadaten
TOOL_VERSION = "3.1.0"

# =============================================================================
# KONFIGURATION
# =============================================================================

class ExportConfig:
    """Konfigurationsklasse für Export-Optionen."""
    
    def __init__(
        self,
        export_parameter_study: bool = True,
        export_timeseries_run_ids: Optional[List[int]] = None,
        export_histogram: bool = True,
        export_seasonal: bool = True,
        histogram_bin_width_mw: Optional[float] = None,  # None = adaptiv
        seasonal_definition: str = "germany",  # "germany" oder "custom"
        output_dir: str = "exports"
    ):
        """
        Initialisiert die Export-Konfiguration.
        
        Args:
            export_parameter_study: Parameterstudie als CSV exportieren
            export_timeseries_run_ids: Liste der Run-IDs für Zeitreihen-Export (None = alle)
            export_histogram: Überschuss-Histogramme exportieren
            export_seasonal: Saisonale Auswertung exportieren
            histogram_bin_width_mw: Binbreite für Histogramm (None = adaptiv)
            seasonal_definition: Saisondefinition ("germany" oder "custom")
            output_dir: Ausgabeverzeichnis
        """
        self.export_parameter_study = export_parameter_study
        self.export_timeseries_run_ids = export_timeseries_run_ids
        self.export_histogram = export_histogram
        self.export_seasonal = export_seasonal
        self.histogram_bin_width_mw = histogram_bin_width_mw
        self.seasonal_definition = seasonal_definition
        self.output_dir = output_dir


# Saisondefinitionen
SEASON_DEFINITIONS = {
    "germany": {
        "Winter": [11, 12, 1, 2],      # Nov-Feb
        "Sommer": [5, 6, 7, 8],         # May-Aug
        "Übergang": [3, 4, 9, 10]       # Mar-Apr, Sep-Oct
    },
    "meteorological": {
        "Winter": [12, 1, 2],
        "Frühling": [3, 4, 5],
        "Sommer": [6, 7, 8],
        "Herbst": [9, 10, 11]
    }
}


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def create_export_directory(base_path: str = "exports") -> Path:
    """
    Erstellt das Export-Verzeichnis falls nicht vorhanden.
    
    Args:
        base_path: Pfad zum Export-Verzeichnis
        
    Returns:
        Path-Objekt zum erstellten Verzeichnis
    """
    export_dir = Path(base_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def generate_metadata() -> Dict[str, Any]:
    """
    Generiert Metadaten für Reproduzierbarkeit.
    
    Returns:
        Dictionary mit Metadaten
    """
    metadata = {
        "tool_version": TOOL_VERSION,
        "export_timestamp": datetime.now().isoformat(),
        "export_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    
    # Git-Commit falls verfügbar
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            metadata["git_commit"] = result.stdout.strip()[:8]
    except:
        metadata["git_commit"] = "not_available"
    
    return metadata


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """
    Sichere Division mit Fallback bei Division durch Null.
    
    Args:
        numerator: Zähler
        denominator: Nenner
        default: Rückgabewert bei Division durch Null
        
    Returns:
        Ergebnis der Division oder default
    """
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def detect_time_resolution_from_index(index: pd.DatetimeIndex) -> Dict[str, Any]:
    """
    Erkennt die Zeitauflösung aus einem DatetimeIndex.
    
    Args:
        index: Pandas DatetimeIndex
        
    Returns:
        Dictionary mit Auflösungsinformationen
    """
    if len(index) < 2:
        return {"dt": 0.25, "resolution": "15-Minuten", "steps_per_day": 96}
    
    # Median der Zeitdifferenzen
    time_diff = pd.Series(index).diff().median()
    minutes = time_diff.total_seconds() / 60
    
    if minutes <= 20:
        return {"dt": 0.25, "resolution": "15-Minuten", "steps_per_day": 96}
    elif minutes <= 70:
        return {"dt": 1.0, "resolution": "Stündlich", "steps_per_day": 24}
    else:
        return {"dt": minutes / 60, "resolution": f"{minutes:.0f}-Minuten", "steps_per_day": int(24 * 60 / minutes)}


# =============================================================================
# 1. PARAMETERSTUDIE EXPORT
# =============================================================================

def export_parameter_study(
    study_results: pd.DataFrame,
    scenario_params: Dict[str, Any],
    output_dir: str = "exports",
    filename: str = "parameter_study_results.csv"
) -> Path:
    """
    Exportiert die Parameterstudie als CSV mit allen relevanten Spalten.
    
    Args:
        study_results: DataFrame mit Parameterstudie-Ergebnissen
        scenario_params: Dictionary mit Szenario-Parametern (nvp_mw, wind_mw, etc.)
        output_dir: Ausgabeverzeichnis
        filename: Name der Ausgabedatei
        
    Returns:
        Pfad zur erstellten Datei
        
    Spalten im Export:
        run_id, scenario, nvp_mw, wind_mw, pv_mw,
        E_MWh, P_MW, EP_h, eta_roundtrip, soc_min, soc_max,
        curtailment_without_MWh, curtailment_with_MWh,
        surplus_without_MWh, captured_MWh, discharged_MWh,
        capture_rate, full_load_hours_nvp, hours_congested, cycles_estimate,
        objective_value, notes
    """
    export_dir = create_export_directory(output_dir)
    
    # Metadaten generieren
    metadata = generate_metadata()
    
    # Export-DataFrame erstellen
    export_df = pd.DataFrame()
    
    # Run-ID
    export_df["run_id"] = range(1, len(study_results) + 1)
    
    # Szenario-Name (falls vorhanden)
    export_df["scenario"] = scenario_params.get("scenario_name", "default")
    
    # Anlagenparameter aus Szenario
    export_df["nvp_mw"] = scenario_params.get("nvp_mw", np.nan)
    export_df["wind_mw"] = scenario_params.get("wind_mw", np.nan)
    export_df["pv_mw"] = scenario_params.get("pv_mw", np.nan)
    
    # Speicherparameter - Mapping von vorhandenen Spalten
    column_mapping = {
        "E_MWh": ["storage_capacity_mwh", "capacity_mwh", "E_MWh"],
        "P_MW": ["storage_power_mw", "power_mw", "P_MW"],
        "EP_h": ["duration_h", "ep_ratio", "EP_h"],
        "eta_roundtrip": ["efficiency", "eta_roundtrip", "eta"],
        "soc_min": ["soc_min"],
        "soc_max": ["soc_max"],
        "curtailment_without_MWh": ["ref_curtailment_mwh", "curtailment_without_MWh", "surplus_mwh"],
        "curtailment_with_MWh": ["curtailment_mwh", "total_curtailment_mwh"],
        "surplus_without_MWh": ["surplus_mwh", "total_surplus_mwh"],
        "captured_MWh": ["captured_mwh", "avoided_curtailment_mwh"],
        "discharged_MWh": ["discharged_mwh", "total_discharged_mwh"],
        "capture_rate": ["capture_rate"],
        "cycles_estimate": ["cycles", "cycles_estimate"],
        "grid_utilization": ["grid_utilization"],
    }
    
    for target_col, source_cols in column_mapping.items():
        value_found = False
        for source_col in source_cols:
            if source_col in study_results.columns:
                export_df[target_col] = study_results[source_col].values
                value_found = True
                break
        if not value_found:
            export_df[target_col] = np.nan
    
    # Berechnete Felder
    
    # capture_rate berechnen falls nicht vorhanden
    if "capture_rate" not in study_results.columns or export_df["capture_rate"].isna().all():
        curtail_without = export_df["curtailment_without_MWh"].values
        curtail_with = export_df["curtailment_with_MWh"].values
        
        capture_rates = []
        for cw, cwith in zip(curtail_without, curtail_with):
            if pd.notna(cw) and cw > 0:
                capture_rates.append(1 - safe_divide(cwith, cw, 0))
            else:
                capture_rates.append(np.nan)
        export_df["capture_rate"] = capture_rates
    
    # Volllaststunden NVP
    nvp_mw = scenario_params.get("nvp_mw", 0)
    if "grid_feed_mwh" in study_results.columns and nvp_mw > 0:
        export_df["full_load_hours_nvp"] = study_results["grid_feed_mwh"] / nvp_mw
    elif "total_grid_feed_mwh" in study_results.columns and nvp_mw > 0:
        export_df["full_load_hours_nvp"] = study_results["total_grid_feed_mwh"] / nvp_mw
    else:
        export_df["full_load_hours_nvp"] = np.nan
    
    # Stunden mit Überlastung (surplus > 0)
    export_df["hours_congested"] = np.nan  # Wird aus Zeitreihen berechnet falls verfügbar
    
    # Zielfunktionswert (falls vorhanden)
    if "objective_value" in study_results.columns:
        export_df["objective_value"] = study_results["objective_value"]
    elif "npv" in study_results.columns:
        export_df["objective_value"] = study_results["npv"]
    else:
        export_df["objective_value"] = np.nan
    
    # Notizen
    export_df["notes"] = ""
    
    # Metadaten als Kommentar am Anfang
    output_path = export_dir / filename
    
    # Header mit Metadaten schreiben
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Parameterstudie Export\n")
        f.write(f"# Tool-Version: {metadata['tool_version']}\n")
        f.write(f"# Export-Zeitpunkt: {metadata['export_timestamp']}\n")
        f.write(f"# Git-Commit: {metadata.get('git_commit', 'n/a')}\n")
        f.write(f"#\n")
    
    # Daten anhängen
    export_df.to_csv(output_path, mode='a', index=False, encoding='utf-8')
    
    return output_path


# =============================================================================
# 2. ZEITREIHEN EXPORT
# =============================================================================

def export_timeseries(
    run_id: int,
    timeseries_data: Dict[str, np.ndarray],
    scenario_params: Dict[str, Any],
    timestamps: Optional[pd.DatetimeIndex] = None,
    output_dir: str = "exports"
) -> Path:
    """
    Exportiert Zeitreihen für einen ausgewählten Run.
    
    Args:
        run_id: ID des Runs
        timeseries_data: Dictionary mit Zeitreihen-Arrays
        scenario_params: Szenario-Parameter
        timestamps: Optionaler DatetimeIndex für Zeitstempel
        output_dir: Ausgabeverzeichnis
        
    Returns:
        Pfad zur erstellten Datei
        
    Exportspalten:
        timestamp, load_MW, wind_MW, pv_MW,
        generation_total_MW, nvp_export_MW,
        curtailment_MW, surplus_MW,
        battery_charge_MW, battery_discharge_MW,
        soc_MWh, soc_pu
    """
    export_dir = create_export_directory(output_dir)
    metadata = generate_metadata()
    
    n = len(next(iter(timeseries_data.values())))
    
    # Zeitstempel generieren falls nicht vorhanden
    if timestamps is None:
        # Versuche Auflösung zu erkennen
        if n >= 34000:  # 15-min
            freq = "15min"
            dt = 0.25
        else:  # Stündlich
            freq = "h"
            dt = 1.0
        timestamps = pd.date_range(start="2024-01-01", periods=n, freq=freq)
    else:
        time_res = detect_time_resolution_from_index(timestamps)
        dt = time_res["dt"]
    
    # Export-DataFrame erstellen
    export_df = pd.DataFrame()
    export_df["timestamp"] = timestamps
    
    # Spalten-Mapping
    column_mapping = {
        "load_MW": ["p_load", "load", "demand"],
        "wind_MW": ["p_wind", "wind"],
        "pv_MW": ["p_pv", "pv", "solar"],
        "generation_total_MW": ["p_ee", "generation", "p_gen"],
        "nvp_export_MW": ["p_grid", "grid_feed", "export"],
        "curtailment_MW": ["p_curtail", "curtailment", "curtail"],
        "surplus_MW": ["p_surplus", "surplus", "excess"],
        "battery_charge_MW": ["p_charge", "charge", "battery_charge"],
        "battery_discharge_MW": ["p_discharge", "discharge", "battery_discharge"],
        "soc_MWh": ["soc", "state_of_charge", "e_stored"],
    }
    
    for target_col, source_keys in column_mapping.items():
        value_found = False
        for source_key in source_keys:
            if source_key in timeseries_data:
                data = timeseries_data[source_key]
                # Länge anpassen falls nötig
                if len(data) == n:
                    export_df[target_col] = data
                elif len(data) == n + 1:  # SOC hat oft n+1 Werte
                    export_df[target_col] = data[:-1]
                else:
                    export_df[target_col] = np.nan
                value_found = True
                break
        if not value_found:
            export_df[target_col] = np.nan
    
    # SOC in pu berechnen falls Kapazität bekannt
    storage_capacity = scenario_params.get("storage_capacity_mwh", None)
    if storage_capacity and not export_df["soc_MWh"].isna().all():
        export_df["soc_pu"] = export_df["soc_MWh"] / storage_capacity
    else:
        export_df["soc_pu"] = np.nan
    
    # Optionale Spalten
    for opt_col in ["market_dispatch_MW", "redispatch_MW"]:
        if opt_col.lower().replace("_mw", "") in timeseries_data:
            export_df[opt_col] = timeseries_data[opt_col.lower().replace("_mw", "")]
        else:
            export_df[opt_col] = np.nan
    
    # Datei schreiben
    filename = f"timeseries_run_{run_id:03d}.csv"
    output_path = export_dir / filename
    
    # Header mit Metadaten
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Zeitreihen-Export Run {run_id}\n")
        f.write(f"# Tool-Version: {metadata['tool_version']}\n")
        f.write(f"# Export-Zeitpunkt: {metadata['export_timestamp']}\n")
        f.write(f"# Zeitauflösung: {dt} h\n")
        f.write(f"# Datenpunkte: {n}\n")
        f.write(f"#\n")
    
    export_df.to_csv(output_path, mode='a', index=False, encoding='utf-8')
    
    return output_path


# =============================================================================
# 3. ÜBERSCHUSS-HISTOGRAMM
# =============================================================================

def calculate_surplus_histogram(
    surplus_mw: np.ndarray,
    dt_hours: float = 0.25,
    bin_width_mw: Optional[float] = None,
    min_bin_width: float = 1.0
) -> pd.DataFrame:
    """
    Berechnet das Überschuss-Histogramm mit Leistungsklassen.
    
    Args:
        surplus_mw: Array mit Überschussleistung in MW
        dt_hours: Zeitauflösung in Stunden
        bin_width_mw: Binbreite in MW (None = adaptiv)
        min_bin_width: Minimale Binbreite in MW
        
    Returns:
        DataFrame mit Histogramm-Daten
    """
    # Nur positive Überschüsse
    positive_surplus = surplus_mw[surplus_mw > 0]
    
    if len(positive_surplus) == 0:
        return pd.DataFrame({
            "bin_lower_MW": [0],
            "bin_upper_MW": [0],
            "hours_count": [0],
            "energy_MWh": [0],
            "frequency_percent": [0]
        })
    
    max_surplus = np.max(positive_surplus)
    
    # Binbreite bestimmen
    if bin_width_mw is None:
        # Adaptiv: max_surplus / 20, mindestens min_bin_width
        bin_width_mw = max(max_surplus / 20, min_bin_width)
    
    # Bins erstellen
    n_bins = int(np.ceil(max_surplus / bin_width_mw))
    bins = np.arange(0, (n_bins + 1) * bin_width_mw, bin_width_mw)
    
    # Histogramm berechnen
    counts, bin_edges = np.histogram(positive_surplus, bins=bins)
    
    # Stunden berechnen (Anzahl * dt)
    hours = counts * dt_hours
    
    # Energie pro Bin (Mittelpunkt * Stunden)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    energy = []
    for i, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (positive_surplus >= lower) & (positive_surplus < upper)
        if np.any(mask):
            energy.append(np.sum(positive_surplus[mask]) * dt_hours)
        else:
            energy.append(0)
    
    # DataFrame erstellen
    hist_df = pd.DataFrame({
        "bin_lower_MW": bin_edges[:-1],
        "bin_upper_MW": bin_edges[1:],
        "hours_count": hours,
        "energy_MWh": energy,
        "frequency_percent": counts / len(surplus_mw) * 100
    })
    
    return hist_df


def export_surplus_histogram(
    run_id: int,
    surplus_mw: np.ndarray,
    dt_hours: float = 0.25,
    bin_width_mw: Optional[float] = None,
    output_dir: str = "exports",
    create_plot: bool = True
) -> Tuple[Path, Optional[Path]]:
    """
    Exportiert das Überschuss-Histogramm als CSV und optional als Plot.
    
    Args:
        run_id: ID des Runs
        surplus_mw: Array mit Überschussleistung
        dt_hours: Zeitauflösung
        bin_width_mw: Binbreite (None = adaptiv)
        output_dir: Ausgabeverzeichnis
        create_plot: Plot als PNG erstellen
        
    Returns:
        Tuple aus (CSV-Pfad, PNG-Pfad oder None)
    """
    export_dir = create_export_directory(output_dir)
    metadata = generate_metadata()
    
    # Histogramm berechnen
    hist_df = calculate_surplus_histogram(surplus_mw, dt_hours, bin_width_mw)
    
    # CSV exportieren
    csv_filename = f"surplus_histogram_run_{run_id:03d}.csv"
    csv_path = export_dir / csv_filename
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(f"# Überschuss-Histogramm Run {run_id}\n")
        f.write(f"# Tool-Version: {metadata['tool_version']}\n")
        f.write(f"# Export-Zeitpunkt: {metadata['export_timestamp']}\n")
        f.write(f"# Zeitauflösung: {dt_hours} h\n")
        f.write(f"# Gesamt-Überschussstunden: {hist_df['hours_count'].sum():.1f} h\n")
        f.write(f"# Gesamt-Überschussenergie: {hist_df['energy_MWh'].sum():.1f} MWh\n")
        f.write(f"#\n")
    
    hist_df.to_csv(csv_path, mode='a', index=False, encoding='utf-8')
    
    # Optional: Plot erstellen
    png_path = None
    if create_plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Balkendiagramm
            bar_width = hist_df["bin_upper_MW"].iloc[0] - hist_df["bin_lower_MW"].iloc[0]
            ax.bar(
                hist_df["bin_lower_MW"] + bar_width/2,
                hist_df["hours_count"],
                width=bar_width * 0.9,
                color='#e74c3c',
                alpha=0.7,
                edgecolor='white'
            )
            
            ax.set_xlabel("Überschussleistung (MW)")
            ax.set_ylabel("Stunden pro Jahr")
            ax.set_title(f"Überschuss-Histogramm (Run {run_id})")
            ax.grid(True, alpha=0.3)
            
            # Statistiken einblenden
            total_hours = hist_df["hours_count"].sum()
            total_energy = hist_df["energy_MWh"].sum()
            max_surplus = surplus_mw.max()
            
            stats_text = (f"Gesamt: {total_hours:.0f} h\n"
                         f"Energie: {total_energy:.0f} MWh\n"
                         f"Max: {max_surplus:.1f} MW")
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            png_filename = f"surplus_histogram_run_{run_id:03d}.png"
            png_path = export_dir / png_filename
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            pass  # Matplotlib nicht verfügbar
    
    return csv_path, png_path


# =============================================================================
# 4. SAISONALE AUSWERTUNG
# =============================================================================

def get_season(month: int, definition: str = "germany") -> str:
    """
    Bestimmt die Saison für einen gegebenen Monat.
    
    Args:
        month: Monat (1-12)
        definition: Saisondefinition ("germany" oder "meteorological")
        
    Returns:
        Name der Saison
    """
    seasons = SEASON_DEFINITIONS.get(definition, SEASON_DEFINITIONS["germany"])
    
    for season_name, months in seasons.items():
        if month in months:
            return season_name
    
    return "Unbekannt"


def calculate_seasonal_statistics(
    timestamps: pd.DatetimeIndex,
    surplus_mw: np.ndarray,
    curtailment_mw: np.ndarray,
    dt_hours: float = 0.25,
    captured_mw: Optional[np.ndarray] = None,
    season_definition: str = "germany"
) -> pd.DataFrame:
    """
    Berechnet saisonale Statistiken für Überschuss und Abregelung.
    
    Args:
        timestamps: DatetimeIndex mit Zeitstempeln
        surplus_mw: Überschussleistung in MW
        curtailment_mw: Abregelungsleistung in MW
        dt_hours: Zeitauflösung
        captured_mw: Optional - erfasste Leistung durch Speicher
        season_definition: Saisondefinition
        
    Returns:
        DataFrame mit saisonalen Kennzahlen
    """
    # DataFrame für Berechnung
    df = pd.DataFrame({
        "timestamp": timestamps,
        "surplus_mw": surplus_mw,
        "curtailment_mw": curtailment_mw,
    })
    
    if captured_mw is not None:
        df["captured_mw"] = captured_mw
    
    # Monat und Saison zuweisen
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["season"] = df["month"].apply(lambda m: get_season(m, season_definition))
    
    # Aggregation pro Saison
    results = []
    
    for season in SEASON_DEFINITIONS.get(season_definition, SEASON_DEFINITIONS["germany"]).keys():
        season_data = df[df["season"] == season]
        
        if len(season_data) == 0:
            continue
        
        surplus_energy = season_data["surplus_mw"].sum() * dt_hours
        curtailment_energy = season_data["curtailment_mw"].sum() * dt_hours
        hours_surplus = (season_data["surplus_mw"] > 0).sum() * dt_hours
        max_surplus = season_data["surplus_mw"].max()
        p95_surplus = np.percentile(season_data["surplus_mw"], 95)
        
        season_results = {
            "season": season,
            "surplus_energy_MWh": surplus_energy,
            "curtailment_energy_MWh": curtailment_energy,
            "hours_surplus": hours_surplus,
            "max_surplus_MW": max_surplus,
            "p95_surplus_MW": p95_surplus,
        }
        
        # Capture Rate wenn möglich
        if captured_mw is not None and surplus_energy > 0:
            captured_energy = season_data["captured_mw"].sum() * dt_hours
            season_results["captured_energy_MWh"] = captured_energy
            season_results["capture_rate_season"] = safe_divide(captured_energy, surplus_energy)
        else:
            season_results["captured_energy_MWh"] = np.nan
            season_results["capture_rate_season"] = np.nan
        
        results.append(season_results)
    
    return pd.DataFrame(results)


def export_seasonal_summary(
    run_id: int,
    timestamps: pd.DatetimeIndex,
    surplus_mw: np.ndarray,
    curtailment_mw: np.ndarray,
    dt_hours: float = 0.25,
    captured_mw: Optional[np.ndarray] = None,
    season_definition: str = "germany",
    output_dir: str = "exports"
) -> Path:
    """
    Exportiert die saisonale Zusammenfassung als CSV.
    
    Args:
        run_id: ID des Runs
        timestamps: DatetimeIndex
        surplus_mw: Überschussleistung
        curtailment_mw: Abregelungsleistung
        dt_hours: Zeitauflösung
        captured_mw: Optional - erfasste Leistung
        season_definition: Saisondefinition
        output_dir: Ausgabeverzeichnis
        
    Returns:
        Pfad zur erstellten Datei
    """
    export_dir = create_export_directory(output_dir)
    metadata = generate_metadata()
    
    # Saisonale Statistiken berechnen
    seasonal_df = calculate_seasonal_statistics(
        timestamps, surplus_mw, curtailment_mw, dt_hours, captured_mw, season_definition
    )
    
    # Datei schreiben
    filename = f"seasonal_summary_run_{run_id:03d}.csv"
    output_path = export_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Saisonale Auswertung Run {run_id}\n")
        f.write(f"# Tool-Version: {metadata['tool_version']}\n")
        f.write(f"# Export-Zeitpunkt: {metadata['export_timestamp']}\n")
        f.write(f"# Saisondefinition: {season_definition}\n")
        f.write(f"# Zeitauflösung: {dt_hours} h\n")
        f.write(f"#\n")
        f.write(f"# Saisondefinition ({season_definition}):\n")
        for season, months in SEASON_DEFINITIONS.get(season_definition, {}).items():
            month_names = {1:'Jan', 2:'Feb', 3:'Mär', 4:'Apr', 5:'Mai', 6:'Jun',
                         7:'Jul', 8:'Aug', 9:'Sep', 10:'Okt', 11:'Nov', 12:'Dez'}
            months_str = ", ".join([month_names[m] for m in months])
            f.write(f"#   {season}: {months_str}\n")
        f.write(f"#\n")
    
    seasonal_df.to_csv(output_path, mode='a', index=False, encoding='utf-8')
    
    return output_path


# =============================================================================
# HAUPTEXPORT-FUNKTION
# =============================================================================

def run_full_export(
    study_results: pd.DataFrame,
    timeseries_dict: Dict[int, Dict[str, np.ndarray]],
    scenario_params: Dict[str, Any],
    timestamps: Optional[pd.DatetimeIndex] = None,
    config: Optional[ExportConfig] = None
) -> Dict[str, List[Path]]:
    """
    Führt den vollständigen Export basierend auf der Konfiguration durch.
    
    Args:
        study_results: DataFrame mit Parameterstudie-Ergebnissen
        timeseries_dict: Dictionary {run_id: {key: array}} mit Zeitreihen
        scenario_params: Szenario-Parameter
        timestamps: Optionaler DatetimeIndex
        config: Export-Konfiguration (None = Defaults)
        
    Returns:
        Dictionary mit Listen der erstellten Dateipfade pro Export-Typ
    """
    if config is None:
        config = ExportConfig()
    
    export_dir = create_export_directory(config.output_dir)
    
    # Ergebnisse sammeln
    exported_files = {
        "parameter_study": [],
        "timeseries": [],
        "histograms": [],
        "seasonal": []
    }
    
    # Zeitauflösung bestimmen
    if timestamps is not None:
        time_res = detect_time_resolution_from_index(timestamps)
        dt_hours = time_res["dt"]
    else:
        # Fallback: aus Anzahl Datenpunkte
        if timeseries_dict:
            first_ts = next(iter(timeseries_dict.values()))
            first_array = next(iter(first_ts.values()))
            n = len(first_array)
            dt_hours = 0.25 if n >= 34000 else 1.0
        else:
            dt_hours = 0.25
    
    # 1. Parameterstudie exportieren
    if config.export_parameter_study and len(study_results) > 0:
        path = export_parameter_study(
            study_results, scenario_params, config.output_dir
        )
        exported_files["parameter_study"].append(path)
    
    # 2. Zeitreihen exportieren
    run_ids = config.export_timeseries_run_ids
    if run_ids is None:
        run_ids = list(timeseries_dict.keys())
    
    for run_id in run_ids:
        if run_id in timeseries_dict:
            path = export_timeseries(
                run_id,
                timeseries_dict[run_id],
                scenario_params,
                timestamps,
                config.output_dir
            )
            exported_files["timeseries"].append(path)
            
            # 3. Histogramm exportieren
            if config.export_histogram:
                surplus_key = None
                for key in ["p_surplus", "surplus", "curtailment", "p_curtail"]:
                    if key in timeseries_dict[run_id]:
                        surplus_key = key
                        break
                
                if surplus_key:
                    csv_path, png_path = export_surplus_histogram(
                        run_id,
                        timeseries_dict[run_id][surplus_key],
                        dt_hours,
                        config.histogram_bin_width_mw,
                        config.output_dir,
                        create_plot=True
                    )
                    exported_files["histograms"].append(csv_path)
                    if png_path:
                        exported_files["histograms"].append(png_path)
            
            # 4. Saisonale Auswertung exportieren
            if config.export_seasonal and timestamps is not None:
                surplus_key = None
                curtail_key = None
                
                for key in ["p_surplus", "surplus"]:
                    if key in timeseries_dict[run_id]:
                        surplus_key = key
                        break
                
                for key in ["p_curtail", "curtailment"]:
                    if key in timeseries_dict[run_id]:
                        curtail_key = key
                        break
                
                if surplus_key and curtail_key:
                    # Captured berechnen falls möglich
                    captured = None
                    if "p_charge" in timeseries_dict[run_id]:
                        captured = timeseries_dict[run_id]["p_charge"]
                    
                    path = export_seasonal_summary(
                        run_id,
                        timestamps,
                        timeseries_dict[run_id][surplus_key],
                        timeseries_dict[run_id][curtail_key],
                        dt_hours,
                        captured,
                        config.seasonal_definition,
                        config.output_dir
                    )
                    exported_files["seasonal"].append(path)
    
    # Zusammenfassung erstellen
    summary_path = export_dir / "export_summary.json"
    summary = {
        "metadata": generate_metadata(),
        "config": {
            "export_parameter_study": config.export_parameter_study,
            "export_timeseries_run_ids": config.export_timeseries_run_ids,
            "export_histogram": config.export_histogram,
            "export_seasonal": config.export_seasonal,
        },
        "files_created": {
            k: [str(p) for p in v] for k, v in exported_files.items()
        },
        "total_files": sum(len(v) for v in exported_files.values())
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return exported_files


# =============================================================================
# STREAMLIT INTEGRATION
# =============================================================================

def create_export_ui_section(
    study_results: Optional[pd.DataFrame] = None,
    timeseries_dict: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    scenario_params: Optional[Dict[str, Any]] = None,
    timestamps: Optional[pd.DatetimeIndex] = None
) -> None:
    """
    Erstellt die Streamlit-UI für den Export-Bereich.
    
    Diese Funktion sollte in der Hauptanwendung aufgerufen werden.
    """
    import streamlit as st
    
    st.markdown("### 📥 Datenexport")
    
    st.markdown("""
    <div class="info-box">
        <strong>Export-Optionen:</strong> Exportieren Sie Rohdaten, Zeitreihen, 
        Histogramme und saisonale Auswertungen für externe Analyse.
    </div>
    """, unsafe_allow_html=True)
    
    # Export-Optionen
    col1, col2 = st.columns(2)
    
    with col1:
        export_param_study = st.checkbox(
            "Parameterstudie (CSV)", 
            value=True,
            help="Alle Parameterkombinationen mit Kennzahlen"
        )
        export_histograms = st.checkbox(
            "Überschuss-Histogramme",
            value=True,
            help="Verteilung der Überschussleistung"
        )
    
    with col2:
        export_timeseries = st.checkbox(
            "Zeitreihen",
            value=False,
            help="Vollständige Zeitreihen für ausgewählte Runs"
        )
        export_seasonal = st.checkbox(
            "Saisonale Auswertung",
            value=True,
            help="Winter/Sommer-Vergleich"
        )
    
    # Erweiterte Optionen
    with st.expander("Erweiterte Export-Optionen"):
        col1, col2 = st.columns(2)
        
        with col1:
            histogram_bin_width = st.number_input(
                "Histogramm-Binbreite (MW)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                help="0 = automatisch (adaptiv)"
            )
            if histogram_bin_width == 0:
                histogram_bin_width = None
        
        with col2:
            season_def = st.selectbox(
                "Saisondefinition",
                ["germany", "meteorological"],
                index=0,
                help="Deutschland: Winter Nov-Feb, Sommer Mai-Aug"
            )
        
        # Run-Auswahl für Zeitreihen
        if export_timeseries and timeseries_dict:
            available_runs = list(timeseries_dict.keys())
            selected_runs = st.multiselect(
                "Runs für Zeitreihen-Export",
                available_runs,
                default=available_runs[:3] if len(available_runs) > 3 else available_runs,
                help="Wählen Sie die Runs für den Zeitreihen-Export"
            )
        else:
            selected_runs = None
    
    # Export-Button
    if st.button("📥 Export starten", type="primary", use_container_width=True):
        if study_results is None or len(study_results) == 0:
            st.warning("Keine Daten zum Exportieren vorhanden.")
            return
        
        with st.spinner("Exportiere Daten..."):
            config = ExportConfig(
                export_parameter_study=export_param_study,
                export_timeseries_run_ids=selected_runs,
                export_histogram=export_histograms,
                export_seasonal=export_seasonal,
                histogram_bin_width_mw=histogram_bin_width,
                seasonal_definition=season_def
            )
            
            try:
                exported = run_full_export(
                    study_results=study_results,
                    timeseries_dict=timeseries_dict or {},
                    scenario_params=scenario_params or {},
                    timestamps=timestamps,
                    config=config
                )
                
                # Erfolg anzeigen
                total_files = sum(len(v) for v in exported.values())
                st.success(f"✅ {total_files} Dateien exportiert nach /exports/")
                
                # Dateien auflisten
                with st.expander("Exportierte Dateien"):
                    for category, files in exported.items():
                        if files:
                            st.markdown(f"**{category}:**")
                            for f in files:
                                st.text(f"  - {f.name}")
                
            except Exception as e:
                st.error(f"Fehler beim Export: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
