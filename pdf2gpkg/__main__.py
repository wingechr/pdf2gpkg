import argparse
import logging

from . import __version__, pdf2gpkg


def main():
    ap = argparse.ArgumentParser(
        prog=None, description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--loglevel",
        "-l",
        choices=["debug", "info", "warning", "error"],
        default="info",
    )
    ap.add_argument("--version", "-v", action="store_true")
    ap.add_argument("--curve-density", "-d", type=float, default=1.0)
    ap.add_argument("--page-no", "-p", type=int, default=1)
    ap.add_argument("source_pdf")
    ap.add_argument("target_dir")
    kwargs = vars(ap.parse_args())
    if kwargs.pop("version"):
        print(__version__)
    loglevel = getattr(logging, kwargs.pop("loglevel").upper())
    logging.basicConfig(
        format="[%(asctime)s %(levelname)7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=loglevel,
    )
    logging.debug(kwargs)
    pdf2gpkg(**kwargs)


if __name__ == "__main__":
    main()
