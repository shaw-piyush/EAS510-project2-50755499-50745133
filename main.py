"""Single entry point for the project.

    python main.py          # SAMPLE profile on data/sample/, offline, < 5 min on CPU
    python main.py --full   # FULL profile: reproduces every report result

Stub for now (PRD 9, Tue Nov 17): it parses the flags, prints the chosen
profile and exits 0. The sample and full pipelines are wired in on Sat Nov 21.
This file must never use the network (CLAUDE.md rule 9).
"""

import argparse
import sys

import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RouterBench router experiments")
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the FULL profile on data/raw/ instead of the sample",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    profile = config.FULL if args.full else config.SAMPLE
    print(f"Profile: {profile['name']}")
    print(f"  seeds:          {profile['seeds']}")
    print(f"  svd_components: {profile['svd_components']}")
    print(f"  results_dir:    {profile['results_dir'].relative_to(config.ROOT)}")
    print("Pipeline not wired yet (stub). Exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
