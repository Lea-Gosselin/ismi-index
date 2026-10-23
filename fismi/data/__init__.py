from .process import prepare_data
from .ea import collect_ea
from .us import collect_us
from fismi.io import load_config


def collect_data(country):
    """
    Collect and standardize inflation data
    """

    config = load_config(country)

    collectors = {
        "EA": collect_ea,
        "US": collect_us,
        # "UK": collect_uk,
        # "JP": collect_jp,
    }

    country = country.upper()

    if country not in collectors:
        raise NotImplementedError(
            f"No collector available for {country}"
        )

    return collectors[country](config)
