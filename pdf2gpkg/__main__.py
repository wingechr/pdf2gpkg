import argparse
import logging

from . import __version__, pdf2gpkg
from .utils import DEFAULT_REF_NAME


def main():
    ap = argparse.ArgumentParser(
        prog="pdf2gpkg",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--loglevel",
        "-l",
        choices=["debug", "info", "warning", "error"],
        default="info",
    )
    ap.add_argument("--version", "-v", action="store_true")
    ap.add_argument(
        "--curve-density",
        "-d",
        type=float,
        default=1.0,
        help="density of points when converting curves to line segments",
    )
    ap.add_argument("--page-no", "-p", type=int, default=1, help="page number in pdf")
    ap.add_argument(
        "--ref-name", "-r", default=DEFAULT_REF_NAME, help="name of reference map"
    )
    ap.add_argument("source_pdf", help="path to input pdf file")
    ap.add_argument("target_dir", help="path to output folder (will be created)")
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
