# 🔋 Batteriespeicher-Optimierungstool

**Entwicklung eines Python-basierten Tools zur Bestimmung optimaler Betriebsstrategien und Dimensionierung von Großbatteriespeichern**

Bachelorarbeit an der TH Köln - Fakultät für Anlagen, Energie- und Maschinensysteme

---

## 🎯 Funktionen

Das Tool bietet drei Analysemodi für verschiedene Anwendungsfälle:

| Modus | Beschreibung | Anwendung |
|-------|--------------|-----------|
| **A - Wirtschaftlichkeit** | Erlös- und Amortisationsberechnung | Großspeicher mit Netzdienstleistungen |
| **B - Peak Shaving** | Lastspitzenkappung mit PyPSA | Industriekunden mit hohen Leistungspreisen |
| **C - NVP-Überbauung** | Speicherdimensionierung bei EE-Überbauung | Wind-/PV-Parks mit begrenztem Netzanschluss |

---

## 📋 Changelog v3.1 (Januar 2026)

### NEU: Erweiterte Export- und Analysefunktionen

**1. CSV-Export der Parameterstudie (Rohdaten)**
- Vollständige Rohdaten statt nur Heatmap
- Spalten: run_id, scenario, nvp_mw, wind_mw, pv_mw, E_MWh, P_MW, EP_h, eta_roundtrip, soc_min, soc_max, curtailment_without_MWh, curtailment_with_MWh, capture_rate, cycles_estimate, etc.
- Export nach `/exports/parameter_study_results.csv`

**2. Zeitreihen-Export (15-min oder stündlich)**
- Automatische Erkennung der Zeitauflösung
- Spalten: timestamp, wind_MW, pv_MW, generation_total_MW, nvp_export_MW, curtailment_MW, battery_charge_MW, battery_discharge_MW, soc_MWh, soc_pu
- Export nach `/exports/timeseries_run_<id>.csv`

**3. Überschuss-Histogramm mit Leistungsklassen**
- Zeigt "Anzahl Stunden pro Leistungsintervall"
- Adaptive Binbreite (max_surplus/20, min 1 MW) oder benutzerdefiniert
- Export als CSV und PNG

**4. Saisonale Auswertung (Winter/Sommer)**
- Definition für Deutschland: Winter (Nov-Feb), Sommer (Mai-Aug), Übergang (Mär-Apr, Sep-Okt)
- Kennzahlen: surplus_energy_MWh, curtailment_energy_MWh, hours_surplus, max_surplus_MW, p95_surplus_MW, capture_rate_season

### Automatische Zeitauflösungserkennung

Das Tool erkennt nun automatisch die Zeitauflösung der hochgeladenen Profile:
- **35.040 Datenpunkte** → 15-Minuten-Auflösung
- **8.760 Datenpunkte** → Stündliche Auflösung

### Modus A - Verbesserte Wirtschaftlichkeitsrechnung

**Realistische Erlösberechnung:**
- FCR-Erlös Default erhöht auf 160.000 €/MW/Jahr (vorher: 120.000)
- aFRR-Erlös Default erhöht auf 70.000 €/MW/Jahr (vorher: 50.000)
- Arbitrage-Erlös Default erhöht auf 35.000 €/MWh/Jahr (vorher: 15.000)

**Verbesserte Arbitrage-Berechnung:**
- 1.5-2 Zyklen pro Tag statt max. 1
- Intraday-Aufschlag von 35% berücksichtigt
- Default-Spread von 65 €/MWh (vorher: 30 €/MWh)

**Neue Visualisierung:**
- Erlösaufteilung als Tortendiagramm
- Detaillierte Erlöstabelle mit Brutto/Netto
- Marktvergleich (konservativ/durchschnitt/optimistisch)

---

## 🚀 Live-Demo

**[➡️ App auf Streamlit Cloud öffnen](https://batteriespeicher-tool.streamlit.app)**

---

## 💻 Lokale Installation

### Voraussetzungen
- Python 3.9 oder höher
- pip (Python Package Manager)

### Installation

```bash
# Repository klonen
git clone https://github.com/DEIN-USERNAME/batteriespeicher-tool.git
cd batteriespeicher-tool

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Die App öffnet sich automatisch unter `http://localhost:8501`

---

## 📊 Benötigte Eingabedaten

### Modus A - Wirtschaftlichkeit
- Optional: Day-Ahead-Preisprofil (CSV)
- Optional: FCR-Preise (CSV)
- Oder: Verwendung von Benchmark-Werten

### Modus B - Peak Shaving
- **Lastprofil** (CSV/Excel): Zeitreihe mit Leistungswerten in kW oder MW
- Format: Spalte 1 = Zeitstempel, Spalte 2 = Leistung
- Auflösung: 15-Minuten oder stündlich (wird automatisch erkannt)

### Modus C - NVP-Überbauung
- **Wind-/PV-Profile** (CSV): Normierte Erzeugungsprofile (0-1)
- Format: Spalte 1 = Zeitstempel, Spalte 2 = normierte Leistung
- Auflösung: 15-Minuten (35.040) oder stündlich (8.760) - wird automatisch erkannt

---

## 🔧 Technische Details

### Optimierung
- **Framework:** PyPSA (Python for Power System Analysis)
- **Solver:** HiGHS (Open-Source LP/MIP Solver)
- **Methode:** Lineare Optimierung mit erweiterbaren Komponenten

### Wirtschaftlichkeitsrechnung
- Annuitätenmethode für CAPEX
- Berücksichtigung von OPEX (% von CAPEX)
- Konfigurierbare Zinssätze und Lebensdauern
- NPV, IRR, Amortisationszeit, LCOS

---

## 📁 Projektstruktur

```
batteriespeicher-tool/
├── app.py                # Hauptanwendung (Streamlit)
├── export_analytics.py   # Export- und Analysemodul
├── requirements.txt      # Python-Abhängigkeiten
├── exports/              # Exportierte Dateien (automatisch erstellt)
│   ├── parameter_study_results.csv
│   ├── timeseries_run_001.csv
│   ├── surplus_histogram_run_001.csv
│   ├── surplus_histogram_run_001.png
│   └── seasonal_summary_run_001.csv
└── README.md             # Diese Datei
```

---

## 📚 Referenzen

- [PyPSA Dokumentation](https://pypsa.readthedocs.io/)
- [Streamlit Dokumentation](https://docs.streamlit.io/)
- [HiGHS Solver](https://highs.dev/)
- Frontier Economics: Batteriespeicher-Marktstudien
- Aurora Energy Research: European Battery Storage Outlook

---

## 👤 Autor

**Luiz Lima**  
TH Köln - Fakultät für Anlagen, Energie- und Maschinensysteme  
Studiengang: Erneuerbare Energien

