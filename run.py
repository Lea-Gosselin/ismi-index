from fismi.pipeline import run_pipeline
import argparse


def main():

    parser = argparse.ArgumentParser(
        description="Inflation Monitoring Pipeline"
    )

    parser.add_argument(
        "--country",
        choices=["EA", "US", "UK", "all"],
        default="all"
    )

    parser.add_argument(
        "--step",
        choices=["data", "ismi", "persistence", "all"],
        default="all"
    )

    parser.add_argument(
        "--refresh-data",
        action="store_true"
    )

    args = parser.parse_args()

    countries = (
        ["EA", "US", "UK"]
        if args.country == "all"
        else [args.country]
    )

    for country in countries:
        print(f"\n{'=' * 40}")
        print(f"Running pipeline: {country}")
        print(f"{'=' * 40}")

        run_pipeline(
            country=country,
            step=args.step,
            refresh_data=args.refresh_data
        )

        print(f"{country}: completed successfully")


if __name__ == "__main__":
    main()
