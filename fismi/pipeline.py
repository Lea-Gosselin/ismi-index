
from fismi.data import collect_data
from fismi.io import (
    load_config,
    load_input,
    save_input,
    save_ismi,
    save_persistence,
)
from fismi.models import (
    estimate_ismi,
    estimate_persistence,
)


def run_pipeline(country, step="all", refresh_data=False):
    """
    Run the inflation monitoring pipeline.

    Parameters
    ----------
    country : str, EA or US (UK, JP incoming)
    step : str, data, ismi, persistence or all
    refresh_data : bool, Whether to collect fresh data
    """

    country = country.upper()
    config = load_config(country)

    valid_steps = {"data", "ismi", "persistence", "all"}

    if step not in valid_steps:
        raise ValueError(f"Unknown step: {step}")

    # 1. Data
    if refresh_data or step == "data":
        prices, weights = collect_data(country)
        save_input(country, prices, weights)
    else:
        prices, weights = load_input(country)

    if step == "data":
        return {"prices": prices, "weights": weights}

    outputs = {}

    # 2. ISMI
    if step in {"ismi", "all"}:
        ismi = estimate_ismi(prices, weights, config)
        save_ismi(country, ismi["results"])
        outputs["ismi"] = ismi

    # 3. Persistence
    if step in {"persistence", "all"}:
        persistence = estimate_persistence(prices, config)
        save_persistence(country, persistence)
        outputs["persistence"] = persistence

    return outputs
