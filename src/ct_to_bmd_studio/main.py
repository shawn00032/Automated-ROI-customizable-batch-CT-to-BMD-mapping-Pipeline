from __future__ import annotations

import sys

from ct_to_bmd_studio.ui.qt_app import run_app


def main() -> None:
    refinement_dev = "--refinement-dev" in sys.argv[1:]
    fast_refinement_dev = "--fast-refinement-dev" in sys.argv[1:]
    run_app(refinement_dev=refinement_dev or fast_refinement_dev, fast_refinement_dev=fast_refinement_dev)


def refinement_dev_main() -> None:
    run_app(refinement_dev=True)


def fast_refinement_dev_main() -> None:
    run_app(refinement_dev=True, fast_refinement_dev=True)


if __name__ == "__main__":
    main()
