
from fismi.io import load_input, save_ismi, load_config
from fismi.models import estimate_ismi


for country in ["EA", "US"]:

    print(f"\n--- {country} ---")

    config = load_config(country)
    prices, weights = load_input(country)

    allResults = estimate_ismi(
        prices=prices,
        weights=weights,
        config=config
    )

    path = save_ismi(country, allResults["results"])

    print(allResults["results"].tail())
    print("Saved:", path)
