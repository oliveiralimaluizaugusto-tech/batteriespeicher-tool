"""
Speichermodelle (PyPSA) für das Batteriespeicher-Optimierungstool
=================================================================

Dieses Modul enthält die reinen Rechenmodelle – ohne Streamlit-Abhängigkeit –
damit sie unabhängig von der Oberfläche getestet werden können.

Konventionen (gelten für alle Modelle):
- Leistungen in MW, Energien in MWh, Preise in €/MWh, CAPEX in €/kW bzw. €/kWh.
- ``dt`` ist die Zeitschrittlänge in Stunden (0.25 = 15 min, 1.0 = stündlich)
  und muss vom Aufrufer übergeben werden (kein Raten aus der Datenpunktzahl).
- Speicherleistung P ist die AC-seitige Lade- UND Entladeleistung
  (ein Wechselrichter, deshalb sind beide Richtungen gekoppelt).
- Speicherkapazität E ist die Nennkapazität. Nutzbar ist E * (soc_max - soc_min).
- Der Speicher wird zyklisch gerechnet (Endfüllstand = Anfangsfüllstand),
  damit keine "geschenkte" Anfangsenergie das Ergebnis verfälscht.
- Die Zielfunktion ist immer in €/Jahr: Investitionen als Annuität + OPEX,
  Betriebskosten/-erlöse werden auf ein Jahr hochgerechnet, falls die
  Zeitreihe kein volles Jahr abdeckt (``year_fraction``).
"""

import logging

import numpy as np
import pypsa

HOURS_PER_YEAR = 8760.0

# PyPSA/linopy loggen sehr ausführlich – für die App reichen Warnungen.
for _name in ("pypsa", "linopy", "highspy"):
    logging.getLogger(_name).setLevel(logging.ERROR)


class OptimizationError(RuntimeError):
    """Optimierung nicht lösbar (z.B. Zielgrenze mit den Grenzen unerreichbar)."""


# =============================================================================
# Finanzmathematik
# =============================================================================
def annuity_factor(rate, years):
    """Annuitätenfaktor ANF = i(1+i)^n / ((1+i)^n - 1)."""
    if years <= 0:
        return 1.0
    if not rate:
        return 1.0 / years
    q = (1 + rate) ** years
    return rate * q / (q - 1)


def present_value_factor(rate, years):
    """Rentenbarwertfaktor = 1 / ANF (Barwert einer konstanten Zahlungsreihe)."""
    return 1.0 / annuity_factor(rate, years)


def npv_constant(capex, annual_net_cashflow, rate, years):
    """Kapitalwert bei konstantem jährlichem Netto-Cashflow."""
    return -capex + annual_net_cashflow * present_value_factor(rate, years)


def irr(cashflows, low=-0.99, high=10.0, tol=1e-7):
    """
    Interner Zinsfuß per Bisektion. Gibt None zurück, wenn kein
    Vorzeichenwechsel des Kapitalwerts im Suchbereich existiert.
    """
    def npv(rate):
        return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))

    f_low, f_high = npv(low), npv(high)
    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low * f_high > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol or (high - low) < tol:
            return mid
        if f_low * f_mid <= 0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2


def annual_storage_cost(power_mw, capacity_mwh, capex_power, capex_energy,
                        interest_rate, lifetime_years, opex_rate):
    """
    Jährliche Speicherkosten (Annuität + OPEX) in €/Jahr.

    capex_power in €/kW, capex_energy in €/kWh, opex_rate als Anteil von CAPEX.
    """
    capex = total_capex(power_mw, capacity_mwh, capex_power, capex_energy)
    anf = annuity_factor(interest_rate, lifetime_years)
    return {
        "capex": capex,
        "annuity": capex * anf,
        "opex": capex * opex_rate,
        "total": capex * (anf + opex_rate),
    }


def total_capex(power_mw, capacity_mwh, capex_power, capex_energy):
    """CAPEX in € aus Leistung (MW, €/kW) und Kapazität (MWh, €/kWh)."""
    return capacity_mwh * 1000 * capex_energy + power_mw * 1000 * capex_power


# =============================================================================
# Hilfsfunktionen für Zeitreihen
# =============================================================================
def year_fraction(n, dt):
    """Anteil eines Jahres, den die Zeitreihe abdeckt (1.0 = volles Jahr)."""
    return n * dt / HOURS_PER_YEAR


def _as_array(series):
    if series is None:
        return None
    values = series.values if hasattr(series, "values") else np.asarray(series)
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError(
            f"Zeitreihe enthält {int(np.isnan(values).sum())} fehlende Werte (NaN). "
            "Bitte Datei prüfen oder Lücken vorher füllen."
        )
    return values


def align_to_length(values, n):
    """
    Bringt eine Zeitreihe (z.B. stündliche Preise) auf n Zeitschritte.

    - gleiche Länge: unverändert
    - ganzzahlig gröber (z.B. 8760 -> 35040): Werte werden wiederholt
    - ganzzahlig feiner (z.B. 35040 -> 8760): Mittelwertbildung
    - sonst: None (passt nicht zusammen)
    """
    if values is None:
        return None
    values = np.asarray(values.values if hasattr(values, "values") else values, dtype=float)
    m = len(values)
    if m == n:
        return values
    if m == 0:
        return None
    if n % m == 0:
        return np.repeat(values, n // m)
    if m % n == 0:
        return values.reshape(n, m // n).mean(axis=1)
    return None


def _solve(network, extra_functionality=None):
    try:
        status, condition = network.optimize(
            solver_name="highs",
            solver_options={"threads": 1, "log_to_console": False, "output_flag": False},
            extra_functionality=extra_functionality,
            progress=False,
        )
    except Exception as exc:  # Solver-/Modellfehler in eine klare Meldung übersetzen
        raise OptimizationError(f"PyPSA-Optimierungsfehler: {exc}") from exc
    if status != "ok" or condition != "optimal":
        raise OptimizationError(
            f"Optimierung nicht lösbar (Status: {status}, {condition}). "
            "Meist sind die Vorgaben mit den gesetzten Grenzen nicht erreichbar."
        )


def _new_network(n, dt):
    """Leeres Netzwerk; Zielfunktion wird auf €/Jahr gewichtet."""
    network = pypsa.Network()
    network.set_snapshots(range(n))
    yf = year_fraction(n, dt)
    # objective: Stunden je Zeitschritt, hochgerechnet auf ein Jahr.
    # generators/stores: tatsächliche Stunden je Zeitschritt (Energiebilanz).
    network.snapshot_weightings.loc[:, "objective"] = dt / yf
    network.snapshot_weightings.loc[:, "generators"] = dt
    network.snapshot_weightings.loc[:, "stores"] = dt
    return network


def _add_battery(network, bus, eta_charge, eta_discharge, soc_min, soc_max,
                 power_mw=None, capacity_mwh=None,
                 annual_cost_power=0.0, annual_cost_energy=0.0,
                 max_power_mw=None, max_capacity_mwh=None,
                 throughput_cost=0.01):
    """
    Fügt einen Batteriespeicher hinzu (Store + Lade-/Entlade-Link).

    power_mw/capacity_mwh gesetzt  -> feste Größe
    power_mw/capacity_mwh = None   -> Größe wird optimiert (extendable)

    annual_cost_power  in €/MW/Jahr  (AC-Leistung)
    annual_cost_energy in €/MWh/Jahr (Nennkapazität)

    Gibt eine extra_functionality-Funktion zurück (oder None), die Lade- und
    Entladeleistung koppelt, wenn die Größe optimiert wird.
    """
    usable = soc_max - soc_min
    if usable <= 0:
        raise ValueError("soc_max muss größer als soc_min sein.")
    extendable = power_mw is None or capacity_mwh is None

    network.add("Bus", "battery_bus")

    if extendable:
        network.add(
            "Store", "battery", bus="battery_bus",
            e_nom_extendable=True, e_cyclic=True,
            e_nom_max=(max_capacity_mwh * usable) if max_capacity_mwh else np.inf,
            # e_nom ist die NUTZBARE Kapazität -> Kosten je nutzbarer MWh
            capital_cost=annual_cost_energy / usable,
        )
        # Ladeleistung = AC-Nennleistung des Wechselrichters (trägt die Kosten)
        network.add(
            "Link", "charge", bus0=bus, bus1="battery_bus",
            p_nom_extendable=True, efficiency=eta_charge,
            p_nom_max=max_power_mw if max_power_mw else np.inf,
            capital_cost=annual_cost_power,
        )
        # Entlade-Link: p_nom zählt batterieseitig, AC-Abgabe = eta_d * p_nom
        network.add(
            "Link", "discharge", bus0="battery_bus", bus1=bus,
            p_nom_extendable=True, efficiency=eta_discharge,
            marginal_cost=throughput_cost,
        )

        def couple_power(n, snapshots):
            p_nom = n.model.variables["Link-p_nom"]
            n.model.add_constraints(
                p_nom.loc["discharge"] * eta_discharge - p_nom.loc["charge"] == 0,
                name="Link-charge_discharge_coupling",
            )

        return couple_power

    network.add(
        "Store", "battery", bus="battery_bus",
        e_nom=capacity_mwh * usable, e_cyclic=True,
    )
    network.add(
        "Link", "charge", bus0=bus, bus1="battery_bus",
        p_nom=power_mw, efficiency=eta_charge,
    )
    network.add(
        "Link", "discharge", bus0="battery_bus", bus1=bus,
        p_nom=power_mw / eta_discharge, efficiency=eta_discharge,
        marginal_cost=throughput_cost,
    )
    return None


def _battery_results(network, dt, eta_discharge, soc_min, soc_max,
                     power_mw=None, capacity_mwh=None):
    """Liest Speichergröße und -zeitreihen aus einem gelösten Netzwerk."""
    usable = soc_max - soc_min
    if capacity_mwh is None:
        capacity_mwh = float(network.stores.loc["battery", "e_nom_opt"]) / usable
    if power_mw is None:
        power_mw = float(network.links.loc["charge", "p_nom_opt"])
    # Numerisches Rauschen des Solvers entfernen
    if power_mw < 1e-6 or capacity_mwh < 1e-6:
        power_mw, capacity_mwh = 0.0, 0.0

    p_charge = np.clip(network.links_t.p0["charge"].values, 0, None)        # AC-Bezug
    # p1 ist in PyPSA negativ, wenn der Link in bus1 einspeist
    p_discharge = np.clip(-network.links_t.p1["discharge"].values, 0, None)  # AC-Abgabe
    soc_usable = network.stores_t.e["battery"].values
    soc = soc_usable + soc_min * capacity_mwh

    charged = p_charge.sum() * dt
    discharged = p_discharge.sum() * dt
    cycles = (discharged / eta_discharge) / capacity_mwh if capacity_mwh > 0 else 0.0

    return {
        "power_mw": power_mw,
        "capacity_mwh": capacity_mwh,
        "ep_ratio": capacity_mwh / power_mw if power_mw > 0 else 0.0,
        "p_charge": p_charge,
        "p_discharge": p_discharge,
        "soc": soc,
        "charged_mwh": charged,
        "discharged_mwh": discharged,
        "losses_mwh": charged - discharged,
        "cycles": cycles,
    }


# =============================================================================
# Modus B: Peak Shaving
# =============================================================================
def _peak_shaving_inputs(load_profile, pv_surplus):
    load = _as_array(load_profile)
    # Negative Nettolast (PV-Überschuss) ist kein Netzbezug: Der Überschuss wird
    # separat als kostenlose Ladequelle modelliert, nicht doppelt gezählt.
    demand = np.clip(load, 0, None)
    surplus = None
    if pv_surplus is not None:
        surplus = np.clip(_as_array(pv_surplus), 0, None)
        if len(surplus) != len(load) or surplus.max() <= 0:
            surplus = None
    return demand, surplus


def _peak_shaving_result(network, demand, surplus, dt, eta_discharge, soc_min, soc_max,
                         energy_price, power_mw=None, capacity_mwh=None, target_limit=None):
    n = len(demand)
    yf = year_fraction(n, dt)
    bat = _battery_results(network, dt, eta_discharge, soc_min, soc_max, power_mw, capacity_mwh)

    grid = np.clip(network.generators_t.p["grid"].values, 0, None)
    original_peak = float(demand.max())
    new_peak = float(grid.max())

    # Mehr-/Minderbezug aus dem Netz gegenüber dem Betrieb ohne Speicher
    # (Verluste erhöhen den Bezug, genutzter PV-Überschuss senkt ihn).
    extra_grid_energy = (grid.sum() - demand.sum()) * dt / yf  # MWh/Jahr

    result = {
        "optimal_power_mw": bat["power_mw"],
        "optimal_capacity_mwh": bat["capacity_mwh"],
        "ep_ratio": bat["ep_ratio"],
        "original_peak_mw": original_peak,
        "new_peak_mw": new_peak,
        "peak_reduction_mw": original_peak - new_peak,
        "grid_power_profile": grid,
        "battery_power_profile": bat["p_discharge"] - bat["p_charge"],
        "soc_profile": bat["soc"],
        "cycles": bat["cycles"] / yf,
        "losses_mwh": bat["losses_mwh"] / yf,
        "extra_grid_energy_mwh": extra_grid_energy,
        "extra_energy_cost": extra_grid_energy * energy_price,
        "year_fraction": yf,
        "optimization_status": "optimal",
        "objective_value": float(network.objective),
    }
    if target_limit is not None:
        result["target_achieved"] = new_peak <= target_limit * (1 + 1e-6) + 1e-9
        result["unmet_peaks_count"] = int((grid > target_limit * (1 + 1e-6) + 1e-9).sum())
    else:
        result["target_achieved"] = True
        result["unmet_peaks_count"] = 0
    if surplus is not None:
        pv_used = np.clip(network.generators_t.p["pv_surplus"].values, 0, None)
        result["pv_to_battery_profile"] = pv_used
        result["pv_to_battery_mwh"] = pv_used.sum() * dt / yf
    return result


def _build_peak_network(demand, surplus, dt, energy_price):
    network = _new_network(len(demand), dt)
    network.add("Bus", "load_bus")
    network.add("Load", "demand", bus="load_bus", p_set=demand)
    if surplus is not None:
        network.add(
            "Generator", "pv_surplus", bus="load_bus",
            p_nom=surplus.max(), p_max_pu=surplus / surplus.max(), marginal_cost=0.0,
        )
    return network


def optimize_peak_shaving_storage(load_profile, dt, efficiency, soc_min, soc_max,
                                  capex_power, capex_energy, interest_rate, lifetime_years,
                                  opex_rate, power_price=0.0, energy_price=0.0,
                                  target_limit=None, pv_surplus=None,
                                  max_power_mw=None, max_capacity_mwh=None):
    """
    Dimensioniert Speicherleistung und -kapazität für Peak Shaving (ein LP).

    Zwei Betriebsarten:

    target_limit gesetzt:
        Die Zielgrenze ist eine HARTE Nebenbedingung (Netzbezug <= target_limit
        in jedem Zeitschritt). Gesucht ist der kostengünstigste Speicher, der
        sie einhält. Ist das nicht möglich, wird OptimizationError ausgelöst.

    target_limit = None:
        Wirtschaftliches Optimum. Die bezogene Jahresspitze ist selbst
        Optimierungsvariable und kostet power_price (€/kW/Jahr). Das LP wägt
        Leistungspreis-Einsparung gegen Speicherkosten ab und liefert die
        Konfiguration mit dem höchsten jährlichen Nettonutzen (ggf. 0 kW).

    Args:
        load_profile: (Netto-)Lastprofil in MW
        dt: Zeitschritt in Stunden
        efficiency: Roundtrip-Wirkungsgrad (0-1)
        capex_power / capex_energy: €/kW bzw. €/kWh
        interest_rate, opex_rate: dezimal (0.06 = 6 %)
        power_price: Leistungspreis in €/kW/Jahr
        energy_price: Arbeitspreis in €/MWh (bewertet die Speicherverluste)
        pv_surplus: optionaler PV-Überschuss in MW (kostenlose Ladequelle)
    """
    demand, surplus = _peak_shaving_inputs(load_profile, pv_surplus)
    eta = float(np.sqrt(efficiency))
    anf = annuity_factor(interest_rate, lifetime_years)
    annual_cost_power = capex_power * 1000 * (anf + opex_rate)    # €/MW/a
    annual_cost_energy = capex_energy * 1000 * (anf + opex_rate)  # €/MWh/a

    network = _build_peak_network(demand, surplus, dt, energy_price)
    if target_limit is not None:
        network.add("Generator", "grid", bus="load_bus",
                    p_nom=target_limit, marginal_cost=energy_price)
    else:
        network.add("Generator", "grid", bus="load_bus",
                    p_nom_extendable=True, p_nom_max=float(demand.max()),
                    capital_cost=power_price * 1000, marginal_cost=energy_price)

    extra = _add_battery(
        network, "load_bus", eta, eta, soc_min, soc_max,
        annual_cost_power=annual_cost_power, annual_cost_energy=annual_cost_energy,
        max_power_mw=max_power_mw, max_capacity_mwh=max_capacity_mwh,
    )
    _solve(network, extra)

    result = _peak_shaving_result(network, demand, surplus, dt, eta, soc_min, soc_max,
                                  energy_price, target_limit=target_limit)
    result.update({
        "annuity_factor": anf,
        "annual_cost_power": annual_cost_power,
        "annual_cost_energy": annual_cost_energy,
    })
    return result


def calculate_peak_shaving(load_profile, dt, storage_power_mw, storage_capacity_mwh,
                           efficiency, soc_min, soc_max, energy_price=0.0,
                           pv_surplus=None, target_limit=None):
    """
    Betriebsoptimierung für einen Speicher FESTER Größe.

    Minimiert die Jahresspitze des Netzbezugs (echte Minimierung des Maximums –
    nicht der Energie oberhalb einer Grenze). Ergebnis ist die niedrigste
    Spitze, die mit diesem Speicher erreichbar ist.
    """
    demand, surplus = _peak_shaving_inputs(load_profile, pv_surplus)
    eta = float(np.sqrt(efficiency))

    network = _build_peak_network(demand, surplus, dt, energy_price)
    # Hohe Kosten je MW Spitze -> Spitze wird vorrangig minimiert; der
    # Arbeitspreis sorgt nachrangig dafür, dass nicht unnötig gezykelt wird.
    network.add("Generator", "grid", bus="load_bus",
                p_nom_extendable=True, p_nom_max=float(demand.max()),
                capital_cost=1e7, marginal_cost=max(energy_price, 1.0))
    _add_battery(network, "load_bus", eta, eta, soc_min, soc_max,
                 power_mw=storage_power_mw, capacity_mwh=storage_capacity_mwh)
    _solve(network)

    return _peak_shaving_result(network, demand, surplus, dt, eta, soc_min, soc_max,
                                energy_price, power_mw=storage_power_mw,
                                capacity_mwh=storage_capacity_mwh, target_limit=target_limit)


def peak_shaving_economics(result, capex_power, capex_energy, opex_rate, power_price,
                           discount_rate, lifetime_years):
    """
    Wirtschaftlichkeit einer Peak-Shaving-Lösung.

    Einsparung = vermiedener Leistungspreis - Mehrkosten Strombezug (Verluste)
                 - OPEX.  Amortisation und NPV beziehen alle drei Posten ein.
    """
    power, capacity = result["optimal_power_mw"], result["optimal_capacity_mwh"]
    capex = total_capex(power, capacity, capex_power, capex_energy)
    opex = capex * opex_rate
    peak_savings = result["peak_reduction_mw"] * 1000 * power_price
    energy_cost = result.get("extra_energy_cost", 0.0)
    net_savings = peak_savings - energy_cost - opex
    payback = capex / net_savings if net_savings > 0 else float("inf")
    if capex <= 0:
        payback = 0.0
    return {
        "capex": capex,
        "opex": opex,
        "peak_savings": peak_savings,
        "energy_cost": energy_cost,
        "net_savings": net_savings,
        "payback_years": payback,
        "npv": npv_constant(capex, net_savings, discount_rate, lifetime_years),
    }


# =============================================================================
# Modus C: NVP-Überbauung
# =============================================================================
def _nvp_generation(wind_profile, pv_profile, p_wind_inst_mw, p_pv_inst_mw):
    wind = _as_array(wind_profile)
    pv = _as_array(pv_profile)
    if wind is None and pv is None:
        raise ValueError("Es wird mindestens ein Erzeugungsprofil benötigt.")
    if wind is not None and pv is not None and len(wind) != len(pv):
        raise ValueError(
            f"Wind- und PV-Profil haben unterschiedliche Längen ({len(wind)} vs. {len(pv)})."
        )
    n = len(wind) if wind is not None else len(pv)
    p_wind = wind * p_wind_inst_mw if wind is not None else np.zeros(n)
    p_pv = pv * p_pv_inst_mw if pv is not None else np.zeros(n)
    return np.clip(p_wind + p_pv, 0, None)


def _nvp_prices(da_prices, n, feed_in_value):
    """Gibt (Preisvektor, Preisprofil_vorhanden) zurück."""
    if da_prices is not None:
        prices = align_to_length(da_prices, n)
        if prices is None:
            raise ValueError(
                f"Preiszeitreihe ({len(da_prices)} Werte) passt nicht zur Erzeugung "
                f"({n} Werte). Erwartet wird dieselbe Länge oder ein ganzzahliges "
                "Verhältnis (z.B. stündliche Preise zu 15-min-Profilen)."
            )
        if np.isnan(prices).any():
            raise ValueError("Preiszeitreihe enthält fehlende Werte (NaN).")
        return prices, True
    return np.full(n, float(feed_in_value)), False


def nvp_reference(p_ee, p_nvp_mw, prices, dt):
    """
    Referenzfall ohne Speicher. Bei negativen Preisen wird – wie im
    optimierten Fall – nicht eingespeist (sonst wäre der Vergleich unfair).
    """
    p_grid = np.where(prices >= 0, np.minimum(p_ee, p_nvp_mw), 0.0)
    p_surplus = np.maximum(p_ee - p_nvp_mw, 0)
    return {
        "p_grid": p_grid,
        "p_surplus": p_surplus,
        "grid_feed_mwh": p_grid.sum() * dt,
        "surplus_mwh": p_surplus.sum() * dt,
        "curtailment_mwh": (p_ee - p_grid).sum() * dt,
        "revenue": float((p_grid * prices).sum() * dt),
    }


def _solve_nvp(p_ee, p_nvp_mw, prices, dt, eta_charge, eta_discharge, soc_min, soc_max,
               power_mw=None, capacity_mwh=None, annual_cost_power=0.0,
               annual_cost_energy=0.0, max_power_mw=None, max_capacity_mwh=None):
    n = len(p_ee)
    network = _new_network(n, dt)
    network.add("Bus", "main")

    # EE-Erzeugung: darf abgeregelt werden (Abregelung = verfügbar - erzeugt).
    # Abregeln ist kostenlos, daher gibt es keinen Anreiz, Überschüsse durch
    # gleichzeitiges Laden und Entladen zu "verheizen".
    p_max = max(float(p_ee.max()), 1e-9)
    network.add("Generator", "ee", bus="main", p_nom=p_max, p_max_pu=p_ee / p_max)

    # Netzeinspeisung am NVP: reine Senke (p <= 0), begrenzt auf P_NVP.
    # marginal_cost = Preis -> Kosten = Preis * p = -Erlös.
    network.add("Generator", "grid", bus="main", p_nom=p_nvp_mw,
                p_min_pu=-1.0, p_max_pu=0.0, marginal_cost=prices)

    extra = _add_battery(
        network, "main", eta_charge, eta_discharge, soc_min, soc_max,
        power_mw=power_mw, capacity_mwh=capacity_mwh,
        annual_cost_power=annual_cost_power, annual_cost_energy=annual_cost_energy,
        max_power_mw=max_power_mw, max_capacity_mwh=max_capacity_mwh,
    )
    _solve(network, extra)
    return network


def _nvp_result(network, p_ee, p_nvp_mw, prices, has_price_profile, dt,
                eta_discharge, soc_min, soc_max, power_mw=None, capacity_mwh=None):
    n = len(p_ee)
    yf = year_fraction(n, dt)
    bat = _battery_results(network, dt, eta_discharge, soc_min, soc_max, power_mw, capacity_mwh)
    ref = nvp_reference(p_ee, p_nvp_mw, prices, dt)

    p_grid = np.clip(-network.generators_t.p["grid"].values, 0, None)
    p_gen = np.clip(network.generators_t.p["ee"].values, 0, None)
    p_curtail = np.clip(p_ee - p_gen, 0, None)
    p_surplus = ref["p_surplus"]

    total_surplus = ref["surplus_mwh"]
    total_grid = p_grid.sum() * dt
    total_revenue = float((p_grid * prices).sum() * dt)
    # Erfassungsgrad: Anteil des Überschusses (> NVP), der in den Speicher geht
    captured = np.minimum(bat["p_charge"], p_surplus).sum() * dt
    capture_rate = min(captured / total_surplus, 1.0) if total_surplus > 0 else 0.0

    discharge_revenue = bat["p_discharge"] * prices * dt
    total_discharge_revenue = float(discharge_revenue.sum())
    avg_discharge_price = (total_discharge_revenue / bat["discharged_mwh"]
                           if bat["discharged_mwh"] > 0 else None)

    return {
        "optimal_power_mw": bat["power_mw"],
        "optimal_capacity_mwh": bat["capacity_mwh"],
        "ep_ratio": bat["ep_ratio"],
        "p_ee": p_ee,
        "p_grid": p_grid,
        "p_charge": bat["p_charge"],
        "p_discharge": bat["p_discharge"],
        "p_curtail": p_curtail,
        "p_curtailment": p_curtail,  # Alias für das Exportmodul
        "p_surplus": p_surplus,
        "soc": bat["soc"],
        "prices": prices if has_price_profile else None,
        "discharge_revenue": discharge_revenue,
        # Summen über den Datenzeitraum
        "total_generation_mwh": p_ee.sum() * dt,
        "total_surplus_mwh": total_surplus,
        "total_curtailment_mwh": p_curtail.sum() * dt,
        "total_grid_feed_mwh": total_grid,
        "total_charged_mwh": bat["charged_mwh"],
        "total_discharged_mwh": bat["discharged_mwh"],
        "total_revenue": total_revenue,
        "total_discharge_revenue": total_discharge_revenue,
        "avg_discharge_price": avg_discharge_price,
        "grid_utilization": total_grid / (p_nvp_mw * n * dt) if p_nvp_mw > 0 else 0.0,
        "capture_rate": capture_rate,
        "cycles": bat["cycles"] / yf,
        # Referenz ohne Speicher (gleicher Zeitraum)
        "ref_grid_feed_mwh": ref["grid_feed_mwh"],
        "ref_curtailment_mwh": ref["curtailment_mwh"],
        "ref_revenue": ref["revenue"],
        # Auf ein Jahr hochgerechnete Mehrwerte des Speichers
        "additional_feed_in_mwh": (total_grid - ref["grid_feed_mwh"]) / yf,
        "additional_revenue": (total_revenue - ref["revenue"]) / yf,
        "year_fraction": yf,
        "has_price_profile": has_price_profile,
        "mean_price": float(prices.mean()),
        "optimization_status": "optimal",
        "objective_value": float(network.objective),
    }


def simulate_nvp_storage(wind_profile, pv_profile, p_nvp_mw, p_wind_inst_mw, p_pv_inst_mw,
                         storage_power_mw, storage_capacity_mwh, dt,
                         eta_charge=0.95, eta_discharge=0.95, soc_min=0.0, soc_max=1.0,
                         da_prices=None, feed_in_value=80.0):
    """
    Betriebsoptimierung eines Speichers FESTER Größe hinter dem NVP.

    Der Speicher lädt nur aus der eigenen EE-Erzeugung (kein Netzbezug) und
    speist über den auf p_nvp_mw begrenzten Netzanschluss ein. Mit Preisprofil
    wird preisoptimiert eingespeist (perfekte Voraussicht = obere Schranke),
    ohne Preisprofil wird jede eingespeiste MWh mit feed_in_value bewertet.
    """
    p_ee = _nvp_generation(wind_profile, pv_profile, p_wind_inst_mw, p_pv_inst_mw)
    prices, has_profile = _nvp_prices(da_prices, len(p_ee), feed_in_value)
    network = _solve_nvp(p_ee, p_nvp_mw, prices, dt, eta_charge, eta_discharge,
                         soc_min, soc_max, power_mw=storage_power_mw,
                         capacity_mwh=storage_capacity_mwh)
    result = _nvp_result(network, p_ee, p_nvp_mw, prices, has_profile, dt, eta_discharge,
                         soc_min, soc_max, power_mw=storage_power_mw,
                         capacity_mwh=storage_capacity_mwh)
    result["feed_in_value"] = feed_in_value
    return result


def optimize_nvp_storage(wind_profile, pv_profile, p_nvp_mw, p_wind_inst_mw, p_pv_inst_mw,
                         dt, max_storage_power_mw=None, max_storage_capacity_mwh=None,
                         eta_charge=0.95, eta_discharge=0.95, soc_min=0.0, soc_max=1.0,
                         da_prices=None, capex_power=170, capex_energy=165,
                         interest_rate=0.06, lifetime_years=15, opex_rate=0.02,
                         feed_in_value=80.0):
    """
    Optimiert Speicherleistung UND -kapazität hinter dem NVP (ein LP).

    Zielfunktion (€/Jahr): Einspeiseerlös - Annuität - OPEX. Der Speicher wird
    genau so groß gebaut, wie sich die letzte kW/kWh noch selbst trägt.
    """
    p_ee = _nvp_generation(wind_profile, pv_profile, p_wind_inst_mw, p_pv_inst_mw)
    prices, has_profile = _nvp_prices(da_prices, len(p_ee), feed_in_value)

    anf = annuity_factor(interest_rate, lifetime_years)
    annual_cost_power = capex_power * 1000 * (anf + opex_rate)    # €/MW/a
    annual_cost_energy = capex_energy * 1000 * (anf + opex_rate)  # €/MWh/a

    network = _solve_nvp(p_ee, p_nvp_mw, prices, dt, eta_charge, eta_discharge,
                         soc_min, soc_max,
                         annual_cost_power=annual_cost_power,
                         annual_cost_energy=annual_cost_energy,
                         max_power_mw=max_storage_power_mw,
                         max_capacity_mwh=max_storage_capacity_mwh)
    result = _nvp_result(network, p_ee, p_nvp_mw, prices, has_profile, dt, eta_discharge,
                         soc_min, soc_max)
    result.update({
        "annuity_factor": anf,
        "annual_cost_energy": annual_cost_energy,
        "annual_cost_power": annual_cost_power,
        "annual_capex_energy": capex_energy * 1000 * anf,
        "annual_opex_energy": capex_energy * 1000 * opex_rate,
        "feed_in_value": feed_in_value,
        "opex_rate": opex_rate,
        "interest_rate": interest_rate,
        "lifetime_years": lifetime_years,
        "capex_power": capex_power,
        "capex_energy": capex_energy,
    })
    if result["optimal_power_mw"] == 0:
        result["optimization_status"] = "no_storage"
    return result


def nvp_economics(sim, capex_power, capex_energy, opex_rate, discount_rate, lifetime_years):
    """
    Wirtschaftlichkeit eines NVP-Speichers aus einem Simulations-/Optimierungs-
    ergebnis. Erlös = Mehrerlös gegenüber dem Betrieb OHNE Speicher (tatsächlich
    zusätzlich eingespeiste Energie nach Verlusten, bewertet zum jeweiligen Preis).
    """
    power, capacity = sim["optimal_power_mw"], sim["optimal_capacity_mwh"]
    capex = total_capex(power, capacity, capex_power, capex_energy)
    opex = capex * opex_rate
    anf = annuity_factor(discount_rate, lifetime_years)
    revenue = sim["additional_revenue"]
    extra_mwh = sim["additional_feed_in_mwh"]
    net = revenue - opex
    discharged_per_year = sim["total_discharged_mwh"] / sim["year_fraction"]
    annual_cost = capex * anf + opex
    # Break-Even: mittleres Preisniveau, ab dem NPV = 0 (Annahme: der Mehrerlös
    # skaliert proportional mit dem Preisniveau). Bei festem Preis entspricht das
    # exakt Jahreskosten / zusätzlich eingespeiste MWh.
    if revenue > 1e-9:
        break_even = sim["mean_price"] * annual_cost / revenue
    else:
        break_even = float("inf")
    if sim.get("has_price_profile") or extra_mwh <= 1e-9:
        avg_price = None  # mit Preisprofil stammt der Erlös auch aus Preisverschiebung
    else:
        avg_price = revenue / extra_mwh
    return {
        "capex": capex,
        "annual_opex": opex,
        "annual_capex": capex * anf,
        "annual_cost": annual_cost,
        "annual_revenue": revenue,
        "additional_feed_in": extra_mwh,
        "avg_price_used": avg_price,
        "annual_net_benefit": revenue - annual_cost,
        "npv": npv_constant(capex, net, discount_rate, lifetime_years),
        "payback_years": capex / net if net > 0 else float("inf"),
        "break_even_price": break_even,
        # Diskontierte Kosten je entladener MWh (ohne Kosten der Ladeenergie)
        "lcos": annual_cost / discharged_per_year if discharged_per_year > 1e-9 else float("inf"),
    }


# =============================================================================
# Modus A: Arbitrage am Spotmarkt
# =============================================================================
def optimize_arbitrage(prices, dt, power_mw, capacity_mwh, efficiency,
                       soc_min=0.0, soc_max=1.0, max_cycles_per_day=None):
    """
    Erlösoptimaler Fahrplan eines Speichers gegen eine Preiszeitreihe (LP).

    Perfekte Voraussicht -> obere Schranke für den Erlös an diesem Markt.
    Wirkungsgradverluste sind enthalten (es wird mehr gekauft als verkauft).

    Args:
        prices: Preise in €/MWh
        max_cycles_per_day: optionale Zyklenbegrenzung (Vollzyklen bezogen auf
            die nutzbare Kapazität), z.B. aus Garantiebedingungen

    Returns:
        dict mit Jahreserlös (€/a), Zyklen/a und Fahrplan
    """
    prices = _as_array(prices)
    n = len(prices)
    yf = year_fraction(n, dt)
    if power_mw <= 0 or capacity_mwh <= 0:
        return {"annual_revenue": 0.0, "cycles_per_year": 0.0, "cycles_per_day": 0.0,
                "p_charge": np.zeros(n), "p_discharge": np.zeros(n),
                "soc": np.zeros(n), "avg_buy_price": None, "avg_sell_price": None,
                "year_fraction": yf}

    eta = float(np.sqrt(efficiency))
    usable = capacity_mwh * (soc_max - soc_min)

    network = _new_network(n, dt)
    network.add("Bus", "market_bus")
    # Markt: p > 0 = Kauf (kostet Preis), p < 0 = Verkauf (bringt Preis)
    network.add("Generator", "market", bus="market_bus", p_nom=power_mw,
                p_min_pu=-1.0, p_max_pu=1.0, marginal_cost=prices)
    _add_battery(network, "market_bus", eta, eta, soc_min, soc_max,
                 power_mw=power_mw, capacity_mwh=capacity_mwh)

    extra = None
    if max_cycles_per_day:
        max_discharge_dc = max_cycles_per_day * usable * (n * dt / 24.0)  # MWh

        def limit_cycles(net, snapshots):
            # .loc statt .sel(Link=...): unabhängig vom Namen der Dimension
            p = net.model.variables["Link-p"].loc[:, "discharge"]
            net.model.add_constraints((p * dt).sum() <= max_discharge_dc,
                                      name="Link-cycle_limit")

        extra = limit_cycles

    _solve(network, extra)
    bat = _battery_results(network, dt, eta, soc_min, soc_max, power_mw, capacity_mwh)

    bought = bat["charged_mwh"]
    sold = bat["discharged_mwh"]
    cost = float((bat["p_charge"] * prices).sum() * dt)
    income = float((bat["p_discharge"] * prices).sum() * dt)
    cycles = (sold / eta) / usable if usable > 0 else 0.0

    return {
        "annual_revenue": (income - cost) / yf,
        "cycles_per_year": cycles / yf,
        "cycles_per_day": cycles / (n * dt / 24.0),
        "p_charge": bat["p_charge"],
        "p_discharge": bat["p_discharge"],
        "soc": bat["soc"],
        "avg_buy_price": cost / bought if bought > 0 else None,
        "avg_sell_price": income / sold if sold > 0 else None,
        "year_fraction": yf,
    }
