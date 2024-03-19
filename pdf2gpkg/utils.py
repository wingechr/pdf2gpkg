import json
import logging
import os
import shutil
from typing import List

import fitz  # PyMuPDF
import geopandas as gpd
import numpy as np
import shapely
import shapely.geometry

# defaults
DEFAULT_REF_NAME = "germany"
BASE_DIR = os.path.dirname(__file__)
BASE_DIR_REF = BASE_DIR + "/ref"
REF_SHP_NAME = "ref.shp.zip"
GPKG_NAME = "result.gpkg"
SRC_GEOJSON = "ref_src.geojson"
TGT_GEOJSON = "ref_tgt.geojson"


"""
Matrix = (
    xx xy x0
    yx yy y0
    0  0  1
)
"""


def combine_affine_transform(t1: List, t2: List):
    xx, xy, x0, yx, yy, y0 = t1
    m1 = np.array([[xx, xy, x0], [yx, yy, y0], [0, 0, 1]])
    xx, xy, x0, yx, yy, y0 = t2
    m2 = np.array([[xx, xy, x0], [yx, yy, y0], [0, 0, 1]])
    m3 = np.matmul(m1, m2)
    xx, xy, x0 = m3[0, :]
    yx, yy, y0 = m3[1, :]
    return xx, xy, x0, yx, yy, y0


def create_affine_transform(point_map: List):
    """
    find minimum error for equations:
    X_ = X * xx + Y * xy + x0
    Y_ = X * yx + Y * yy + y0
    """

    coeffs = []
    results = []
    for (X, Y), (X_, Y_) in point_map:
        coeffs.append([X, Y, 1])
        results.append([X_, Y_])

    # (M, N): M = len(point_map), N = 3: (x, y, abs)
    coeffs = np.array(coeffs)
    # (M, K): M = len(point_map), K = 2: (x, y)
    results = np.array(results)
    # Solve for the transformation matrix using least squares
    if point_map:
        assert coeffs.shape == (len(point_map), 3)
        assert results.shape == (len(point_map), 2)
        M, _, _, _ = np.linalg.lstsq(coeffs, results, rcond=None)
    else:  # no transform
        M = np.array([[1, 0], [0, 1], [0, 0]])
    assert M.shape == (3, 2)
    xx, xy, x0 = M[:, 0]
    yx, yy, y0 = M[:, 1]
    return xx, xy, x0, yx, yy, y0


def apply_affine_transform(affine_transform, x_y):
    xx, xy, x0, yx, yy, y0 = affine_transform
    X, Y = x_y
    X_ = X * xx + Y * xy + x0
    Y_ = X * yx + Y * yy + y0
    return X_, Y_


def get_reverse_affine_transform(affine_transform):
    xx, xy, x0, yx, yy, y0 = affine_transform
    m = np.array(
        [
            [xx, xy, x0],
            [yx, yy, y0],
            [0, 0, 1],
        ]
    )
    m_ = np.linalg.inv(m)

    xx_, xy_, x0_ = m_[0, :]
    yx_, yy_, y0_ = m_[1, :]
    assert tuple(m_[2, :]) == (0.0, 0.0, 1.0)

    return xx_, xy_, x0_, yx_, yy_, y0_


def rgb_to_hex(rgb_0_1):
    r, g, b = [int(x * 255) for x in rgb_0_1]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def points_to_line(
    affine_transform, points: List[fitz.Point]
) -> shapely.geometry.LineString:
    """create shapely line Object from list of fitz.Point."""
    return shapely.geometry.LineString(
        [apply_affine_transform(affine_transform, (p.x, p.y)) for p in points]
    )


def create_quadratic_bezier_curve(
    points: List[fitz.Point], density: float = 1
) -> List[fitz.Point]:
    p_start, p_control1, p_control2, p_end = points
    length = ((p_end.x - p_start.x) ** 2 + (p_end.y - p_start.y) ** 2) ** 0.5
    n_segments = max(1, round(density * length))  # must have at least one segment
    result = []
    for i in range(n_segments + 1):  # inclusing end points
        t1 = i / n_segments
        t2 = 1 - t1
        result.append(
            fitz.Point(
                x=t1**3 * p_start.x
                + 3 * t1**2 * t2 * p_control1.x
                + 3 * t1 * t2**2 * p_control2.x
                + t2**3 * p_end.x,
                y=t1**3 * p_start.y
                + 3 * t1**2 * t2 * p_control1.y
                + 3 * t1 * t2**2 * p_control2.y
                + t2**3 * p_end.y,
            )
        )
    return result


def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(transform, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(transform, file)


def copy_template(out_dir, ref_name=DEFAULT_REF_NAME):
    # create output dir
    os.makedirs(out_dir, exist_ok=True)
    # copy ref template
    dir_ref = BASE_DIR_REF + "/" + ref_name
    for f in os.listdir(dir_ref):
        shutil.copy(dir_ref + "/" + f, out_dir + "/" + f)


def get_text_labels_as_points(page, transform, crs):
    items = []
    for x0, y0, x1, y1, text, _block_no, _block_type in page.get_text("blocks"):
        text = text.strip()
        if not text:
            logging.info("Skip empty text node")
            continue
        # center point
        x = (x0 + x1) / 2
        y = (y0 + y1) / 2
        x_, y_ = apply_affine_transform(transform, (x, y))
        geometry = shapely.geometry.Point(x_, y_)
        items.append({"geometry": geometry, "text": text})
    gdf = gpd.GeoDataFrame(items, crs=crs)
    return gdf


def get_vector_shapes_as_lines(page, transform, crs, curve_density=1):
    items = []
    for vector_object in page.get_drawings():
        lines = []
        for item in vector_object["items"]:
            # items are tuples of ("l" | "c", Point, Point, ...)
            item_type, points = item[0], item[1:]

            if item_type == "re":
                if len(points) != 2:  # Rect, number
                    raise NotImplementedError(len(points))
                rect, scale = points
                if not isinstance(rect, fitz.Rect):
                    raise NotImplementedError(rect)
                if not isinstance(scale, (int, float)):
                    raise NotImplementedError(scale)
                # convert to quad
                item_type = "qu"
                points = [rect.quad]

            if item_type == "qu":
                if len(points) != 1:
                    raise NotImplementedError(len(points))
                quad = points[0]
                if not isinstance(quad, fitz.Quad):
                    raise NotImplementedError(quad)
                points = [quad.ll, quad.ul, quad.ur, quad.lr]
            elif item_type == "c":  # Splines
                if not len(points) == 4:
                    raise NotImplementedError(len(points))
                # create bezier curve
                points = create_quadratic_bezier_curve(points, density=curve_density)

            if not points:
                logging.info("Skipping empty shape")
                continue

            part_types = set(type(p) for p in points)
            if not all(issubclass(c, fitz.Point) for c in part_types):
                raise NotImplementedError((item_type, part_types))

            if item_type not in ("l", "c", "qu", "re"):
                raise NotImplementedError(f"item_type: {item_type}")

            points_ = points_to_line(transform, points)
            lines.append(points_)

        if not lines:
            logging.info("Skip empty multiline")
            continue

        items.append(
            {
                "geometry": shapely.geometry.MultiLineString(lines),
                "dashes": bool(
                    vector_object["dashes"] and vector_object["dashes"] != "[] 0"
                ),
                "color": (
                    rgb_to_hex(vector_object["color"])
                    if vector_object["color"]
                    else None
                ),
                "type": vector_object["type"],
            }
        )

    # split lines and areas
    if not items:
        raise logging.error("No geometry")

    gdf = gpd.GeoDataFrame(items, crs=crs)
    return gdf


def get_page(pdf_path, page_no=1):
    pdf = fitz.open(pdf_path)
    page = pdf.load_page(page_no - 1)
    return page


def get_initial_transform(pdf_path, out_dir, page_no=1):
    # get page dimension
    page = get_page(pdf_path, page_no=page_no)
    width = page.rect.width
    height = page.rect.height

    gdf_ref = gpd.read_file(out_dir + "/" + REF_SHP_NAME)
    xmin, ymin, xmax, ymax = gdf_ref.total_bounds

    # map page to bounds
    ll = [(0, height), (xmin, ymin)]
    lr = [(width, height), (xmax, ymin)]
    ur = [(width, 0), (xmax, ymax)]

    transform = create_affine_transform([ll, lr, ur])
    crs = str(gdf_ref.crs)

    return {"transform": transform, "crs": crs}


def get_corrected_transform(out_dir):
    gpkg_path = out_dir + "/" + GPKG_NAME
    transform_crs = read_json(gpkg_path + ".transform.json")
    initial_transform = transform_crs["transform"]
    crs = transform_crs["crs"]
    ref_src = read_json(out_dir + "/" + SRC_GEOJSON)
    ref_tgt = read_json(out_dir + "/" + TGT_GEOJSON)

    ref_src = [f["geometry"]["coordinates"] for f in ref_src["features"]]
    ref_tgt = [f["geometry"]["coordinates"] for f in ref_tgt["features"]]

    new_transform = create_affine_transform(list(zip(ref_src, ref_tgt)))
    # in reverse order (left side multiplication of matrix)
    combined_transform = combine_affine_transform(new_transform, initial_transform)
    return {"transform": combined_transform, "crs": crs}


def extract_page_geoemtry(pdf_path, out_dir, page_no=1, curve_density=1):
    gpkg_path = out_dir + "/" + GPKG_NAME

    transform_file = gpkg_path + ".transform.json"
    if os.path.exists(transform_file):
        logging.info("loading transformation")
        transform_crs = get_corrected_transform(out_dir)
        # scr = target, so we overwrite it
        ref = read_json(out_dir + "/" + TGT_GEOJSON)
        ref["name"] = "ref_src"
        save_json(ref, out_dir + "/" + SRC_GEOJSON)

    else:
        logging.info("initializing transformation")
        transform_crs = get_initial_transform(pdf_path, out_dir, page_no=page_no)

    transform = transform_crs["transform"]
    crs = transform_crs["crs"]

    page = get_page(pdf_path, page_no=page_no)

    gdf_labels = get_text_labels_as_points(page, transform, crs)
    gdf_shapes = get_vector_shapes_as_lines(
        page, transform, crs, curve_density=curve_density
    )

    gdf_labels.to_file(gpkg_path, layer="labels")
    gdf_shapes.to_file(gpkg_path, layer="lines")
    save_json(transform_crs, transform_file)


def pdf2gpkg(source_pdf, target_dir, page_no=1, curve_density=1.0):
    if not os.path.exists(target_dir):
        copy_template(target_dir)
    extract_page_geoemtry(
        source_pdf, target_dir, page_no=page_no, curve_density=curve_density
    )
