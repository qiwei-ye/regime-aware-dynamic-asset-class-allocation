"""Single command-line entry point for the research pipeline."""

import argparse
import sys

from regime_allocation import config, evaluation
from regime_allocation.data import DataUnavailableError, prepare_data
from regime_allocation.models import ModelFitError
from regime_allocation.portfolio import AllocationError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "select", "test"), default="all")
    arguments = parser.parse_args(argv)
    try:
        if arguments.stage in ("all", "data"):
            manifest = prepare_data()
            coverage = manifest["coverage"]
            print(
                f"Data preparation complete: {coverage['research_weeks']} research weeks, "
                f"{coverage['required_daily_sessions_per_asset']} sessions per asset. "
                f"See {config.RETRIEVAL_MANIFEST_PATH.relative_to(config.PROJECT_ROOT).as_posix()}."
            )
        if arguments.stage in ("all", "select"):
            evaluation.run_selection()
        if arguments.stage in ("all", "test"):
            evaluation.run_locked_test()
    except (DataUnavailableError, ModelFitError, AllocationError, evaluation.EvaluationError, OSError) as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
