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

### Modus B - Peak Shaving
- **Lastprofil** (CSV/Excel): Zeitreihe mit Leistungswerten in kW oder MW
- Format: Spalte 1 = Zeitstempel, Spalte 2 = Leistung
- Auflösung: 15-Minuten empfohlen

### Modus C - NVP-Überbauung
- **Wind-/PV-Profile** (CSV): Normierte Erzeugungsprofile (0-1)
- Format: Spalte 1 = Zeitstempel, Spalte 2 = normierte Leistung
- Auflösung: 15-Minuten, 1 Jahr (35.040 Zeitschritte)

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

---

## 📁 Projektstruktur

```
batteriespeicher-tool/
├── app.py              # Hauptanwendung (Streamlit)
├── requirements.txt    # Python-Abhängigkeiten
└── README.md           # Diese Datei
```

---

## 📚 Referenzen

- [PyPSA Dokumentation](https://pypsa.readthedocs.io/)
- [Streamlit Dokumentation](https://docs.streamlit.io/)
- [HiGHS Solver](https://highs.dev/)

---

## 👤 Autor

**Luiz Lima**  
TH Köln - Fakultät für Anlagen, Energie- und Maschinensysteme  
Studiengang: Erneuerbare Energien

