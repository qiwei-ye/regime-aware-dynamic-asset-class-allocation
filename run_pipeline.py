"""Single command-line entry point for the research pipeline."""

import argparse
import sys

from regime_allocation import config
from regime_allocation.data import DataUnavailableError, prepare_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data",), required=True)
    parser.parse_args(argv)
    try:
        manifest = prepare_data()
    except DataUnavailableError as error:
        print(f"Data preparation stopped: {error}", file=sys.stderr)
        return 1
    coverage = manifest["coverage"]
    print(
        f"Data preparation complete: {coverage['research_weeks']} research weeks, "
        f"{coverage['required_daily_sessions_per_asset']} sessions per asset. "
        f"See {config.RETRIEVAL_MANIFEST_PATH.relative_to(config.PROJECT_ROOT).as_posix()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
