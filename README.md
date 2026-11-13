# Batteriespeicher-Optimierungstool

**Entwicklung eines Python-basierten Tools zur Bestimmung optimaler Betriebsstrategien und Dimensionierung von Großbatteriespeichern**

Bachelorarbeit an der TH Köln - Fakultät für Anlagen, Energie- und Maschinensysteme

---

## Funktionen

Das Tool bietet drei Analysemodi für verschiedene Anwendungsfälle:

| Modus | Beschreibung | Anwendung |
|-------|--------------|-----------|
| **A - Wirtschaftlichkeit** | Erlös- und Amortisationsberechnung | Großspeicher mit Netzdienstleistungen |
| **B - Peak Shaving** | Lastspitzenkappung mit PyPSA | Industriekunden mit hohen Leistungspreisen |
| **C - NVP-Überbauung** | Speicherdimensionierung bei EE-Überbauung | Wind-/PV-Parks mit begrenztem Netzanschluss |

---



**[App auf Streamlit Cloud öffnen](https://batteriespeicher-tool.streamlit.app)**

---

## Lokale Installation

### Voraussetzungen
- Python 3.9 oder höher
- pip (Python Package Manager)

### Installation

```bash
# Repository klonen
git clone https://github.com/oliveiralimaluizaugusto-tech/batteriespeicher-tool.git
cd batteriespeicher-tool

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Die App öffnet sich automatisch unter `http://localhost:8501`

---

## Benötigte Eingabedaten

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

## Technische Details

### Optimierung
- **Framework:** PyPSA (Python for Power System Analysis)
- **Solver:** HiGHS (Open-Source LP/MIP Solver)
- **Methode:** Lineare Optimierung mit erweiterbaren Komponenten

### Wirtschaftlichkeitsrechnung
- Annuitätenmethode für CAPEX
- Berücksichtigung von OPEX (% von CAPEX)
- Konfigurierbare Zinssätze und Lebensdauern
- NPV, IRR, Amortisationszeit, LCOS

### Modellannahmen (`speicher_modell.py`)
- **Einheitliche Zielfunktion in €/Jahr:** Investition als Annuität + OPEX, Betriebskosten und
  Erlöse mit der tatsächlichen Zeitschrittlänge gewichtet (15-min- und Stundendaten liefern
  dasselbe Ergebnis). Zeitreihen, die kein volles Jahr abdecken, werden linear hochgerechnet.
- **Ein Wechselrichter:** Lade- und Entladeleistung sind gekoppelt (AC-seitige Nennleistung).
- **SOC-Grenzen:** Kosten beziehen sich auf die Nennkapazität, nutzbar ist `soc_max − soc_min`.
- **Zyklischer Speicher:** Endfüllstand = Anfangsfüllstand (keine geschenkte Anfangsenergie).
- **Modus A:** Arbitrage als optimaler Fahrplan gegen die Preiszeitreihe (perfekte Voraussicht,
  obere Schranke), optional mit Zyklenlimit. FCR/aFRR/Arbitrage teilen sich Leistung und Kapazität.
- **Modus B:** Zielgrenze als harte Nebenbedingung (kostenminimaler Speicher) oder
  wirtschaftliches Optimum mit der Jahresspitze als Variable (Leistungspreis in der Zielfunktion).
  Verlustenergie wird mit dem Arbeitspreis bewertet.
- **Modus C:** Der Speicher lädt nur aus eigener Erzeugung (kein Netzbezug). Bewertet wird der
  Mehrerlös gegenüber dem Betrieb ohne Speicher – also die tatsächlich zusätzlich eingespeiste
  Energie nach Verlusten.
- **Nicht modelliert:** Degradation/Ersatzinvestitionen, Netzentgelte und Umlagen, Steuern,
  Prognosefehler. Ergebnisse sind Auslegungsrechnungen, keine Erlösprognosen.

---

## Projektstruktur

```
batteriespeicher-tool/
├── app.py                # Hauptanwendung (Streamlit-Oberfläche)
├── speicher_modell.py    # Rechenmodelle (PyPSA), ohne Streamlit – separat testbar
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

## Referenzen

- [PyPSA Dokumentation](https://pypsa.readthedocs.io/)
- [Streamlit Dokumentation](https://docs.streamlit.io/)
- [HiGHS Solver](https://highs.dev/)
- Frontier Economics: Batteriespeicher-Marktstudien
- Aurora Energy Research: European Battery Storage Outlook

---

## Autor

**Luiz Lima**  
TH Köln - Fakultät für Anlagen, Energie- und Maschinensysteme  
Studiengang: Erneuerbare Energien

