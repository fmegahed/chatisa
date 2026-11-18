/**
 * In-browser Python worker (Pyodide).
 *
 * A static ES-module worker, loaded by URL rather than through the bundler.
 * Pyodide and its wheels are self-hosted under /runtimes/pyodide/, so a run
 * downloads only the packages a snippet imports, from our own origin.
 *
 * Protocol, shared with the other language workers:
 *   in:  { id, code }
 *   out: { id, ok: true, result } | { id, ok: false, error }
 * result: { text?, table?: { columns, rows }, imageDataUrl? }
 *
 * One Pyodide instance is created on first use and reused. Globals are cleared
 * between runs so one snippet cannot see another's variables.
 */

let readyPromise = null;

async function getPyodide() {
  if (!readyPromise) {
    readyPromise = (async () => {
      const { loadPyodide } = await import("/runtimes/pyodide/pyodide.mjs");
      const pyodide = await loadPyodide({ indexURL: "/runtimes/pyodide/" });
      // There is no DOM in a worker, so matplotlib must render off-screen. AGG
      // makes plt.show() a no-op and lets us save a figure to PNG bytes. We also
      // silence the "FigureCanvasAgg is non-interactive" warning plt.show() then
      // emits: it is expected here, since we capture the figure ourselves.
      pyodide.runPython(
        [
          "import os, warnings",
          "os.environ['MPLBACKEND'] = 'AGG'",
          "warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')",
        ].join("\n"),
      );
      // Web access parity with R (2026-07-24): a module (surviving the
      // per-run globals clearing) that, once `requests` has been imported,
      // patches its transport to route CROSS-ORIGIN calls through our
      // SSRF-guarded /api/py-proxy. Same-origin calls stay direct; response
      // .url is restored so students see the address they asked for. The
      // patch call is a no-op until requests actually appears in sys.modules,
      // so sessions that never touch the network pay nothing.
      pyodide.runPython(
        `
import sys as _chatisa_sys, types as _chatisa_types
_chatisa_mod = _chatisa_types.ModuleType('_chatisa_net')
_chatisa_mod.__dict__['_origin'] = ${JSON.stringify(self.location.origin)}
# The function is compiled WITH the module's dict as its globals: the per-run
# clearing of user globals (and the del below) must never break it.
exec(compile('''
_patched = False
def patch():
    global _patched
    import sys
    if _patched or 'requests' not in sys.modules:
        return
    from urllib.parse import quote, urlsplit
    from requests.adapters import HTTPAdapter
    _proxy = _origin + '/api/py-proxy?url='
    _orig_send = HTTPAdapter.send
    def _send(self, request, **kwargs):
        url = request.url or ''
        parts = urlsplit(url)
        same_origin = (parts.scheme + '://' + parts.netloc) == _origin
        if parts.scheme in ('http', 'https') and parts.netloc and not same_origin:
            original = url
            request.url = _proxy + quote(original, safe='')
            response = _orig_send(self, request, **kwargs)
            response.url = original
            return response
        return _orig_send(self, request, **kwargs)
    HTTPAdapter.send = _send
    _patched = True
''', '<chatisa_net>', 'exec'), _chatisa_mod.__dict__)
_chatisa_sys.modules['_chatisa_net'] = _chatisa_mod
del _chatisa_sys, _chatisa_types, _chatisa_mod
`,
      );
      return pyodide;
    })();
  }
  return readyPromise;
}

const QUIET = { messageCallback: () => {}, errorCallback: () => {} };

/**
 * Pure-Python packages Pyodide does not build, which we host ourselves (see
 * scripts/setup-runtimes setupPypiWheels). Each lists the Pyodide-built packages
 * it needs at import time, so those load first and the wheel then installs with
 * deps off, never reaching out to PyPI. wheelDeps are other hosted wheels.
 */
const HOSTED_WHEELS = {
  seaborn: { pyodideDeps: ["numpy", "pandas", "matplotlib", "scipy"], wheelDeps: [] },
  openpyxl: { pyodideDeps: [], wheelDeps: ["et_xmlfile"] },
  et_xmlfile: { pyodideDeps: [], wheelDeps: [] },
  // House chart style (2026-07-25): the matplotlib counterparts of ggrepel and
  // ggtext. Keys are the IMPORT names, which is what neededHostedWheels matches.
  adjustText: { pyodideDeps: ["numpy", "matplotlib", "scipy"], wheelDeps: [] },
  highlight_text: { pyodideDeps: ["matplotlib"], wheelDeps: [] },
  // The Nixtla forecasting stack (v6.3.0). statsforecast and coreforecast are
  // OUR OWN wasm builds (vendor/pyodide-wasm-wheels, ABI-pinned to this
  // Pyodide); the rest are pure PyPI wheels. fugue, triad and adagio ride
  // along because statsforecast.core imports fugue unconditionally even
  // though only its cluster backends (spark, dask, ray; none installable
  // here) would ever use it.
  utilsforecast: { pyodideDeps: ["numpy", "pandas", "packaging", "narwhals"], wheelDeps: [] },
  coreforecast: { pyodideDeps: ["numpy"], wheelDeps: [] },
  triad: {
    pyodideDeps: ["numpy", "pandas", "six", "pyarrow", "fsspec"],
    wheelDeps: [],
  },
  adagio: { pyodideDeps: [], wheelDeps: ["triad"] },
  fugue: { pyodideDeps: ["numpy", "pandas"], wheelDeps: ["triad", "adagio"] },
  statsforecast: {
    pyodideDeps: [
      "numpy",
      "pandas",
      "scipy",
      "statsmodels",
      "cloudpickle",
      "tqdm",
      "threadpoolctl",
      "packaging",
      "narwhals",
    ],
    wheelDeps: ["coreforecast", "utilsforecast", "fugue"],
  },
};

let wheelManifest = null;
async function getWheelManifest() {
  if (!wheelManifest) {
    try {
      const res = await fetch("/runtimes/pyodide-wheels/wheels.json");
      wheelManifest = res.ok ? await res.json() : {};
    } catch {
      wheelManifest = {};
    }
  }
  return wheelManifest;
}

const installedWheels = new Set();

/** Installs a hosted wheel (and its hosted deps) from our origin, once. */
async function installHostedWheel(pyodide, name) {
  if (installedWheels.has(name) || !HOSTED_WHEELS[name]) return;
  const spec = HOSTED_WHEELS[name];
  for (const dep of spec.wheelDeps) await installHostedWheel(pyodide, dep);
  const loads = [...spec.pyodideDeps, "micropip"];
  await pyodide.loadPackage(loads, QUIET);
  const file = (await getWheelManifest())[name];
  if (!file) return;
  const url = `${self.location.origin}/runtimes/pyodide-wheels/${file}`;
  // deps=False because the dependencies above are already present, so micropip
  // never has to resolve anything against PyPI (which CORS would block anyway).
  await pyodide.runPythonAsync(
    `import micropip\nawait micropip.install(${JSON.stringify(url)}, deps=False)`,
  );
  installedWheels.add(name);
}

/** Which hosted wheels a snippet needs: an explicit import, or read_excel (which
 * pandas services with openpyxl). */
function neededHostedWheels(code) {
  const needed = new Set();
  for (const name of Object.keys(HOSTED_WHEELS)) {
    if (new RegExp(`(^|\\n)\\s*(import\\s+${name}\\b|from\\s+${name}[\\s.])`).test(code)) {
      needed.add(name);
    }
  }
  if (/\bread_excel\b|\.xlsx\b/.test(code)) needed.add("openpyxl");
  return [...needed];
}

/**
 * Reports the user's variables (name, type, a short shape or preview) as JSON,
 * for the Variables pane. Runs in the session namespace but ignores dunder and
 * helper names, so it does not show its own machinery.
 */
const INSPECT_VARS = `
def __chatisa_vars():
    import json as __j
    out = []
    for __k, __v in list(globals().items()):
        if __k.startswith("__"):
            continue
        __t = type(__v).__name__
        if __t in ("module", "function", "builtin_function_or_method", "type"):
            continue
        __cols = None
        try:
            if hasattr(__v, "columns") and hasattr(__v, "dtypes"):
                # A pandas DataFrame: an RStudio-style summary plus column dtypes.
                __r, __c = __v.shape[0], __v.shape[1]
                __info = f"{__r} obs. of {__c} variable" + ("" if __c == 1 else "s")
                __cols = [
                    {"name": str(__c), "type": str(__dt)}
                    for __c, __dt in list(zip(__v.columns, __v.dtypes))[:60]
                ]
            elif hasattr(__v, "shape"):
                __info = " x ".join(str(__d) for __d in __v.shape)
            elif isinstance(__v, (str, bytes)):
                __s = repr(__v)
                __info = __s if len(__s) <= 40 else __s[:37] + "..."
            elif hasattr(__v, "__len__"):
                __info = "len " + str(len(__v))
            else:
                __s = repr(__v)
                __info = __s if len(__s) <= 40 else __s[:37] + "..."
        except Exception:
            __info = ""
        __entry = {"name": __k, "type": __t, "info": __info}
        if __cols is not None:
            __entry["columns"] = __cols
        out.append(__entry)
    return __j.dumps(out)
__chatisa_vars()
`;

/**
 * Renders the current matplotlib figure, if any, to PNG and reads a few facts
 * from it (title, axis labels, mark types, legend entries) for the plot's alt
 * text (#23). Returns JSON {png, info}; the facts are best-effort and never
 * stop the image from coming back.
 */
const CAPTURE_PLOT = `
def __chatisa_plot_info(fig):
    from matplotlib.container import BarContainer
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.patches import Wedge
    axes = [a for a in fig.get_axes() if a.get_visible() and a.get_label() != "<colorbar>"]
    kinds = []
    def add(k):
        if k not in kinds:
            kinds.append(k)
    st = getattr(fig, "_suptitle", None)
    title = st.get_text() if st is not None else ""
    xl = yl = ""
    series = []
    for ax in axes:
        title = title or ax.get_title()
        xl = xl or ax.get_xlabel()
        yl = yl or ax.get_ylabel()
        if any(isinstance(c, BarContainer) for c in ax.containers):
            add("bars")
        if ax.images:
            add("heatmap")
        for c in ax.collections:
            if isinstance(c, PathCollection):
                add("points")
            elif isinstance(c, PolyCollection):
                add("area")
        if any(l.get_visible() and len(l.get_xdata()) > 1 and l.get_linestyle() not in ("None", "", " ") for l in ax.lines):
            add("lines")
        if any(isinstance(p, Wedge) for p in ax.patches):
            add("pie")
        leg = ax.get_legend()
        if leg is not None:
            series += [t.get_text() for t in leg.get_texts()]
    return {"kinds": kinds, "title": title, "xLabel": xl, "yLabel": yl, "series": series[:12], "panels": len(axes)}

def __chatisa_plot_data(fig):
    # The plotted values, for keyboard and sound exploration (#40). Neutral
    # shape, validated and capped again in lib/run/maidr-schema.ts. Anything
    # unrecognised is skipped rather than guessed: the picture and its
    # description stay either way.
    import math
    import numpy as np
    from matplotlib.container import BarContainer
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib import dates as mdates
    axes = [a for a in fig.get_axes() if a.get_visible() and a.get_label() != "<colorbar>"]
    if len(axes) != 1:
        return {"panels": len(axes), "layers": []}
    ax = axes[0]
    counted = [0]
    def fin(v):
        try:
            f = float(v)
        except Exception:
            return None
        return f if math.isfinite(f) else None
    def cap(items, n):
        counted[0] += len(items)
        if len(items) <= n:
            return list(items)
        step = len(items) / n
        return [items[int(i * step)] for i in range(n)]
    def name_of(artist):
        lab = artist.get_label() if hasattr(artist, "get_label") else ""
        return lab if isinstance(lab, str) and lab and not lab.startswith("_") else None
    def tick_text(axis, pos, lo=None, hi=None):
        # The category label drawn at (or within) a position.
        best = None
        for loc, t in zip(axis.get_ticklocs(), axis.get_ticklabels()):
            text = t.get_text()
            if not text:
                continue
            if lo is not None and lo - 1e-9 <= loc <= hi + 1e-9:
                return text
            if abs(loc - pos) < 1e-6:
                best = text
        return best
    def grid_labels(axis, n, offset):
        # Category names when the ticks carry them (seaborn heatmap); plain
        # 1-based positions when they are only numbers.
        labels = [tick_text(axis, i + offset) for i in range(n)]
        def numeric(t):
            try:
                float(t.replace(",", "").replace(chr(8722), "-"))
                return True
            except Exception:
                return False
        if any(t is None or numeric(t) for t in labels):
            return [str(i + 1) for i in range(n)]
        return labels
    def is_date(axis):
        return isinstance(axis.get_major_formatter(), (mdates.DateFormatter, mdates.AutoDateFormatter, mdates.ConciseDateFormatter))
    def categories(axis):
        # Names plotted as strings (ax.scatter(grades, ["Amanda", ...]),
        # plt.plot(["Jan", "Feb"], ...)): matplotlib keeps their positions in
        # the axis units. None for a numeric or date axis.
        try:
            from matplotlib import category as mcat
            u = axis.get_units() if hasattr(axis, "get_units") else getattr(axis, "units", None)
            if isinstance(u, mcat.UnitData) and u._mapping:
                return {int(v): str(k) for k, v in u._mapping.items()}
        except Exception:
            pass
        return None
    x_cats, y_cats = categories(ax.xaxis), categories(ax.yaxis)
    def cat_at(cats, v):
        f = fin(v)
        return cats.get(int(round(f))) if f is not None else None
    def x_value(v):
        if x_cats is not None:
            name = cat_at(x_cats, v)
            return name if name is not None else fin(v)
        if is_date(ax.xaxis):
            try:
                return mdates.num2date(v).date().isoformat()
            except Exception:
                return fin(v)
        return fin(v)

    layers = []
    point_bars = []
    def add_points(name, xy):
        if x_cats is not None and y_cats is not None:
            return  # both axes are categories: nothing to measure
        if x_cats is None and y_cats is None:
            layers.append({"kind": "point", "name": name, "points": cap([{"x": fin(x), "y": fin(y)} for x, y in xy], 4000)})
            return
        # Dots along one category axis (the Coding Studio's dumbbell, a dot
        # plot) are read as bars, one series per dot layer, as the R and SQL
        # extraction do: "Student is Amanda, Grade is 91".
        cats, ci, vi = (y_cats, 1, 0) if y_cats is not None else (x_cats, 0, 1)
        pts = []
        for p in xy:
            c, v = cat_at(cats, p[ci]), fin(p[vi])
            if c is not None and v is not None:
                pts.append({"x": c, "y": v})
        if pts:
            point_bars.append({"name": name, "points": cap(pts, 400)})
    hist_ids = set()
    for rec in ax.__dict__.get("_chatisa_hists", []):
        try:
            n, bins, patches = rec
            n = np.asarray(n, dtype=float)
            if n.ndim != 1:
                continue
            for p in (patches if isinstance(patches, (list, BarContainer)) else []):
                hist_ids.add(id(p))
            layers.append({"kind": "hist", "bins": cap([{"xMin": fin(bins[i]), "xMax": fin(bins[i + 1]), "y": fin(n[i])} for i in range(len(n))], 400)})
        except Exception:
            pass

    bars = []
    for c in ax.containers:
        if not isinstance(c, BarContainer) or any(id(p) in hist_ids for p in c.patches):
            continue
        rects = list(c.patches)
        if not rects:
            continue
        horizontal = getattr(c, "orientation", None) == "horizontal"
        widths = [r.get_height() if horizontal else r.get_width() for r in rects]
        starts = [r.get_y() if horizontal else r.get_x() for r in rects]
        # seaborn histplot draws bins with ax.bar: equal widths that touch.
        touching = len(rects) >= 3 and all(abs(w - widths[0]) <= 1e-9 * max(1.0, abs(widths[0])) for w in widths) and all(abs(starts[i] + widths[i] - starts[i + 1]) <= 1e-6 * max(1.0, abs(widths[0])) for i in range(len(rects) - 1))
        if touching and not horizontal:
            layers.append({"kind": "hist", "bins": cap([{"xMin": fin(r.get_x()), "xMax": fin(r.get_x() + r.get_width()), "y": fin(r.get_height())} for r in rects], 400)})
            continue
        values = getattr(c, "datavalues", None)
        if values is None:
            values = [r.get_width() if horizontal else r.get_height() for r in rects]
        axis = ax.yaxis if horizontal else ax.xaxis
        points = []
        for r, v in zip(rects, values):
            lo = r.get_y() if horizontal else r.get_x()
            hi = lo + (r.get_height() if horizontal else r.get_width())
            label = tick_text(axis, (lo + hi) / 2, lo, hi)
            points.append({"x": label if label is not None else str(round((lo + hi) / 2, 6)), "y": fin(v)})
        stacked_here = any(abs(r.get_x() if horizontal else r.get_y()) > 1e-12 for r in rects)
        bars.append(({"name": name_of(c), "points": cap(points, 400)}, horizontal, stacked_here))
    if bars:
        layers.append({"kind": "bar", "horizontal": bars[0][1], "stacked": len(bars) > 1 and any(b[2] for b in bars), "series": [b[0] for b in bars[:10]]})

    has_bars = bool(bars)
    box_artists = ax.__dict__.get("_chatisa_box_artists", set())
    line_series = []
    for l in ax.lines:
        if not l.get_visible() or id(l) in box_artists:
            continue
        xy = np.asarray(l.get_xydata(), dtype=float)
        if xy.ndim != 2 or len(xy) < 2:
            continue
        no_line = l.get_linestyle() in ("None", "", " ")
        if has_bars and len(xy) == 2 and abs(xy[0][0] - xy[1][0]) < 1e-12:
            continue  # an error bar on a bar chart
        if no_line:
            if l.get_marker() not in (None, "None", "", " "):
                add_points(name_of(l), xy)
            continue
        if y_cats is not None:
            continue  # runs along a category axis (a dumbbell's connector); R and SQL skip these too
        pts = [{"x": x_value(x), "y": fin(y)} for x, y in xy]
        line_series.append({"name": name_of(l), "points": cap(pts, 4000)})
    if line_series:
        layers.append({"kind": "line", "series": line_series[:10]})

    for c in ax.collections:
        if isinstance(c, PathCollection):
            off = np.ma.getdata(c.get_offsets())
            if off.ndim == 2 and len(off) > 0:
                add_points(name_of(c), off)
        elif isinstance(c, QuadMesh):
            z = np.ma.filled(np.ma.asarray(c.get_array(), dtype=float), np.nan)
            coords = c.get_coordinates()
            if z.ndim == 1 and coords.ndim == 3:
                z = z.reshape(coords.shape[0] - 1, coords.shape[1] - 1)
            if z.ndim == 2 and 0 < z.shape[0] <= 60 and 0 < z.shape[1] <= 60:
                rows, cols = z.shape
                xs = grid_labels(ax.xaxis, cols, 0.5)
                ys = grid_labels(ax.yaxis, rows, 0.5)
                grid = [[fin(v) for v in row] for row in z]
                if not ax.yaxis_inverted():
                    ys, grid = ys[::-1], grid[::-1]
                counted[0] += rows * cols
                layers.append({"kind": "heat", "x": xs, "y": ys, "z": grid})
    for im in ax.images:
        z = np.ma.filled(np.ma.asarray(im.get_array(), dtype=float), np.nan)
        if z.ndim == 2 and 0 < z.shape[0] <= 60 and 0 < z.shape[1] <= 60:
            rows, cols = z.shape
            xs = grid_labels(ax.xaxis, cols, 0)
            ys = grid_labels(ax.yaxis, rows, 0)
            grid = [[fin(v) for v in row] for row in z]
            if im.origin == "lower":
                ys, grid = ys[::-1], grid[::-1]
            counted[0] += rows * cols
            layers.append({"kind": "heat", "x": xs, "y": ys, "z": grid})
    if point_bars:
        layers.append({"kind": "bar", "horizontal": y_cats is not None, "stacked": False, "series": point_bars[:10]})

    for stats, horizontal, positions in ax.__dict__.get("_chatisa_boxes", []):
        boxes = []
        axis = ax.yaxis if horizontal else ax.xaxis
        for i, s in enumerate(stats[:100]):
            pos = positions[i] if positions is not None and i < len(positions) else i + 1
            label = s.get("label") or tick_text(axis, pos) or "Group %d" % (i + 1)
            lo, hi = fin(s.get("whislo")), fin(s.get("whishi"))
            fliers = [fin(v) for v in np.ravel(s.get("fliers", []))]
            fliers = [v for v in fliers if v is not None]
            boxes.append({"label": str(label), "min": lo, "q1": fin(s.get("q1")), "q2": fin(s.get("med")), "q3": fin(s.get("q3")), "max": hi, "lowerOutliers": [v for v in fliers if lo is not None and v < lo][:200], "upperOutliers": [v for v in fliers if hi is not None and v > hi][:200]})
        counted[0] += len(boxes)
        if boxes:
            layers.append({"kind": "box", "horizontal": bool(horizontal), "boxes": boxes})

    # What the colours mean, so MAIDR says "Course is ISA 401" rather than its
    # default "Level is": the legend's title, else a colour bar's label.
    z_label = ""
    try:
        lg = ax.get_legend()
        if lg is not None:
            z_label = lg.get_title().get_text()
        if not z_label:
            for a in fig.get_axes():
                if a.get_label() == "<colorbar>":
                    z_label = a.get_ylabel() or a.get_xlabel()
                    break
    except Exception:
        z_label = ""
    st = getattr(fig, "_suptitle", None)
    return {"title": (st.get_text() if st is not None else "") or ax.get_title(), "xLabel": ax.get_xlabel(), "yLabel": ax.get_ylabel(), "zLabel": z_label, "panels": 1, "layers": layers[:6], "total": counted[0]}

def __chatisa_capture_plot():
    import sys
    if "matplotlib.pyplot" not in sys.modules:
        return None
    import matplotlib.pyplot as plt
    if not plt.get_fignums():
        return None
    import io, base64, json
    fig = plt.gcf()
    buf = io.BytesIO()
    # Saving draws the figure, which fills in the tick labels that seaborn and
    # pandas set lazily, so the facts are read after it (#40).
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    try:
        info = __chatisa_plot_info(fig)
    except Exception:
        info = None
    try:
        data = __chatisa_plot_data(fig)
    except Exception:
        data = None
    plt.close("all")
    return __chatisa_dump(buf, info, data)

def __chatisa_dump(buf, info, data):
    import base64, json
    try:
        return json.dumps({"png": base64.b64encode(buf.getvalue()).decode("ascii"), "info": info, "data": data}, allow_nan=False)
    except (TypeError, ValueError):
        # Data that cannot be sent as plain JSON never costs the plot.
        return json.dumps({"png": base64.b64encode(buf.getvalue()).decode("ascii"), "info": info, "data": None})
__chatisa_capture_plot()
`;

/**
 * Records what matplotlib's own hist() and bxp() computed (bin edges and
 * counts; box statistics), so a histogram or box plot can be explored with its
 * exact values (#40). Reading them back from the drawn rectangles and lines
 * would mean guessing. bxp() is what Axes.boxplot, pandas' box plots and
 * seaborn 0.13's boxplot all call. Wrapped with functools.wraps so help()
 * and the HELP pane still show matplotlib's own docstrings. Idempotent, and a
 * no-op when matplotlib is not installed.
 */
const MPL_PATCH = `
def __chatisa_patch_mpl():
    try:
        import functools
        import matplotlib.axes as _ma
    except Exception:
        return
    if getattr(_ma.Axes, "_chatisa_patched", False):
        return
    _bxp = _ma.Axes.bxp
    @functools.wraps(_bxp)
    def bxp(self, bxpstats, *args, **kwargs):
        out = _bxp(self, bxpstats, *args, **kwargs)
        try:
            horizontal = kwargs.get("orientation") == "horizontal" or kwargs.get("vert") is False
            self.__dict__.setdefault("_chatisa_boxes", []).append((list(bxpstats), horizontal, kwargs.get("positions")))
            # The box outlines, whiskers, caps, medians and fliers are Line2D
            # objects; they belong to the box layer, not to a line chart.
            drawn = self.__dict__.setdefault("_chatisa_box_artists", set())
            for group in (out or {}).values():
                for a in group:
                    drawn.add(id(a))
        except Exception:
            pass
        return out
    _hist = _ma.Axes.hist
    @functools.wraps(_hist)
    def hist(self, *args, **kwargs):
        out = _hist(self, *args, **kwargs)
        try:
            self.__dict__.setdefault("_chatisa_hists", []).append(out)
        except Exception:
            pass
        return out
    _ma.Axes.bxp = bxp
    _ma.Axes.hist = hist
    _ma.Axes._chatisa_patched = True
__chatisa_patch_mpl()
`;

/**
 * Turns a Pyodide/Python failure into a message a student can act on. A missing
 * package is the common case in the browser, so it gets a plain explanation
 * rather than a traceback; every other error keeps its traceback, because the
 * traceback is the lesson.
 */
function friendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);
  const bundled =
    "numpy, pandas, matplotlib, scikit-learn, statsmodels, statsforecast, pyarrow, polars, seaborn and openpyxl";
  const missing = /ModuleNotFoundError: No module named '([^']+)'/.exec(message);
  if (missing) {
    const pkg = missing[1].split(".")[0];
    return `The Python package "${pkg}" is not available in the browser. Bundled here are ${bundled} (and what they depend on). To add a pure-Python package, try micropip.`;
  }
  if (/Can't fetch|Failed to fetch|lockfile/i.test(message)) {
    return `A package this code needs is not available in the browser. Bundled here are ${bundled} (and what they depend on).`;
  }
  return message;
}

/** Returns one page of a DataFrame as JSON: {columns, rows, total} or {error}. */
function fetchFramePageCode(name, offset, limit) {
  return `
def __chatisa_page(__name, __off, __lim):
    import json as __j
    __v = globals().get(__name)
    if __v is None or not hasattr(__v, "iloc") or not hasattr(__v, "columns"):
        return __j.dumps({"error": "That variable is not a data frame."})
    __page = __v.iloc[__off:__off + __lim]
    return __j.dumps({
        "columns": [str(__c) for __c in __v.columns],
        "rows": __j.loads(__page.to_json(orient="records", date_format="iso")),
        "total": int(len(__v)),
    })
__chatisa_page(${JSON.stringify(name)}, ${Number(offset)}, ${Number(limit)})
`;
}

/** Serializes a named DataFrame/Series to CSV/TSV. Returns {text} or {error}. */
function exportFrameCode(name, sep) {
  return `
def __chatisa_export(__name, __sep):
    import json as __j
    __v = globals().get(__name)
    if __v is None or not hasattr(__v, "to_csv"):
        return __j.dumps({"error": "That variable is not a data frame or table."})
    return __j.dumps({"text": __v.to_csv(sep=__sep, index=False)})
__chatisa_export(${JSON.stringify(name)}, ${JSON.stringify(sep)})
`;
}

/** Python that pickles every serializable data global into one file at /tmp/__ws.pkl using
 * pickle protocol 4 (a dict of {name: value}), and reports the objects it could not pickle.
 * Returns JSON {included, skipped, empty}. Read-only: it reads existing globals and never
 * re-runs the student's code. Modules, functions, and types are skipped up front; anything
 * else that will not pickle (open files, generators, some custom objects) is reported, never
 * silently dropped. A pickle preserves structure and dtypes, unlike a CSV export, but it runs
 * code when loaded, so the UI adds a trust warning on download. */
function workspacePickleCode(names) {
  const wanted = Array.isArray(names) ? JSON.stringify(names) : "None";
  return `
def __chatisa_workspace():
    import json, pickle
    wanted = ${wanted}
    data, included, skipped = {}, [], []
    for k, v in list(globals().items()):
        if k.startswith("__"):
            continue
        if wanted is not None and k not in wanted:
            continue
        t = type(v).__name__
        if t in ("module", "function", "builtin_function_or_method", "type"):
            continue
        try:
            pickle.dumps(v, protocol=4)  # test picklability without keeping the bytes
            data[k] = v
            included.append(k)
        except Exception:
            skipped.append(k + " (" + t + ")")
    if not included:
        return json.dumps({"included": [], "skipped": skipped, "empty": True})
    with open("/tmp/__ws.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)
    return json.dumps({"included": included, "skipped": skipped, "empty": False})
__chatisa_workspace()
`;
}

/** Python that restores an uploaded .pkl workspace (5d): pickle.load the dict from
 * /tmp/__upload, then either report its members and collisions (preview) or assign each
 * into globals under the conflict rule (import). Import runs ONLY when trusted, because a
 * pickle can run code as it loads. Returns a JSON string. */
function restorePickleCode(mode, trusted, rule) {
  const isImport = mode === "import";
  return `
def __chatisa_restore():
    import json, pickle
    if ${isImport ? "True" : "False"} and not ${trusted ? "True" : "False"}:
        return json.dumps({"error": "Confirm you trust this file before restoring it."})
    try:
        with open("/tmp/__upload", "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return json.dumps({"error": "That file is not a readable pickle (" + type(e).__name__ + ")."})
    if not isinstance(data, dict):
        return json.dumps({"error": "That file is not a ChatISA workspace pickle."})
    g = globals()
    members = [k for k in data.keys() if not str(k).startswith("__")]
    if not ${isImport ? "True" : "False"}:
        return json.dumps({"restore": True, "members": [
            {"name": k, "collides": (k in g)} for k in members]})
    rule = ${JSON.stringify(rule || "rename")}
    restored, skipped, renamed = [], [], []
    for k in members:
        v = data[k]
        if k in g:
            if rule == "skip":
                skipped.append(k); continue
            if rule == "overwrite":
                g[k] = v; restored.append(k); continue
            t = k; n = 2
            while t in g:
                t = k + "_" + str(n); n += 1
            g[t] = v; renamed.append(k + " to " + t); continue
        g[k] = v; restored.append(k)
    return json.dumps({"ok": True, "restored": restored, "skipped": skipped, "renamed": renamed})
__chatisa_restore()
`;
}

/** A plain, em-dash-free sentence describing what a restore did. */
function pyRestoreNote(parsed) {
  const arr = (x) => (Array.isArray(x) ? x : []);
  const restored = arr(parsed.restored), skipped = arr(parsed.skipped), renamed = arr(parsed.renamed);
  const parts = [];
  if (restored.length) parts.push(`Restored ${restored.join(", ")}.`);
  if (renamed.length) parts.push(`Renamed to avoid a clash: ${renamed.join("; ")}.`);
  if (skipped.length) parts.push(`Skipped (already in your session): ${skipped.join(", ")}.`);
  return parts.join(" ") || "Nothing to restore.";
}

/**
 * Resolves a clicked symbol to a live object and returns its docstring as JSON:
 * {found, text?, signature?}. Tries the live receiver (df.groupby), then the bare
 * name (len), then a curated module fallback from the source hint (pandas/numpy), so
 * a pandas method resolves to pandas.DataFrame.<name> even before any df is defined.
 * Runs only introspection (the same technique as the autocomplete op); never the
 * student's code. `__doc_req` (name, qualifier, source) is set from JS.
 */
const DOC_CODE = `
def __chatisa_doc(req):
    import json, inspect, importlib
    name = req.get("name") or ""
    qualifier = req.get("qualifier")
    source = (req.get("source") or "").lower()
    obj = None
    # 1) Live receiver: df.groupby reads the actual bound method when df exists.
    if qualifier:
        try:
            base = eval(qualifier, globals())
            obj = getattr(base, name, None)
        except Exception:
            obj = None
    # 2) Bare name in the session globals or builtins: len, print, a user function.
    if obj is None:
        try:
            obj = eval(name, globals())
        except Exception:
            obj = None
    # 3) Cold fallback: no live object yet. Map a common import alias (pd, np,
    #    sns, plt, sm, sklearn) or a literal module name to its module and read the
    #    attribute, so a click resolves before the student has run their imports.
    #    Covers any bundled library, not just pandas/numpy.
    if obj is None:
        aliases = {
            "pd": "pandas", "np": "numpy", "plt": "matplotlib.pyplot",
            "sns": "seaborn", "sm": "statsmodels.api",
            "smf": "statsmodels.formula.api", "sk": "sklearn", "pl": "polars",
        }
        candidates = []
        if qualifier:
            candidates.append(aliases.get(qualifier, qualifier))
        mod_hint = {"pandas": "pandas", "numpy": "numpy"}.get(source)
        if mod_hint:
            candidates.append(mod_hint)
        for mod_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            # A top-level function (read_csv, array) lives on the module; a
            # DataFrame method (groupby) lives on the pandas DataFrame class.
            cand = getattr(mod, name, None)
            if cand is None and mod_name == "pandas":
                cand = getattr(getattr(mod, "DataFrame", object), name, None)
            if cand is not None:
                obj = cand
                break
    if obj is None:
        return json.dumps({"found": False})
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return json.dumps({"found": False})
    sig = ""
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = ""
    # Hard cap so the message stays small; the pane also caps and notes truncation.
    if len(doc) > 12000:
        doc = doc[:12000]
    return json.dumps({"found": True, "text": doc, "signature": sig})
__chatisa_doc(__doc_req)
`;

/**
 * Runtime-aware autocomplete: inspects the live session for the members of the
 * object before a dot, or the names in scope for a bare word. Returns JSON
 * {partial, options:[{label,type,detail}]}. `__cc_prefix` is set from JS.
 *
 * Safety: this runs entirely inside the student's own Pyodide sandbox (their
 * browser tab, their own session), and the eval only ever receives a dotted
 * identifier path (the regex excludes parentheses and calls), so it resolves an
 * object like `df` or `df.columns` to introspect it. This is the same technique
 * IPython/Jupyter use for tab completion; there is no server or cross-user
 * surface.
 */
const COMPLETE_CODE = `
def __chatisa_complete(__p):
    import json, keyword, builtins, re, inspect
    __out = {"partial": "", "options": []}
    __m = re.search(r"([A-Za-z_][A-Za-z0-9_.]*)\\.([A-Za-z0-9_]*)$", __p)
    if __m:
        __base, __partial = __m.group(1), __m.group(2)
        __out["partial"] = __partial
        try:
            __obj = eval(__base, globals())
        except Exception:
            __obj = None
        if __obj is not None:
            __names = sorted(x for x in dir(__obj)
                             if not x.startswith("_") and x.startswith(__partial))
            for __x in __names[:200]:
                try:
                    __a = getattr(__obj, __x)
                    __call = callable(__a)
                    __doc = ""
                    if __call:
                        __doc = (inspect.getdoc(__a) or "").strip().split("\\n")[0][:120]
                    __out["options"].append({
                        "label": __x,
                        "type": "method" if __call else "property",
                        "detail": __doc,
                    })
                except Exception:
                    __out["options"].append({"label": __x, "type": "property", "detail": ""})
        return json.dumps(__out)
    __m2 = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", __p)
    __partial = __m2.group(1) if __m2 else ""
    __out["partial"] = __partial
    __names = set(k for k in globals().keys() if not k.startswith("_"))
    __names |= set(x for x in dir(builtins) if not x.startswith("_"))
    __names |= set(keyword.kwlist)
    for __x in sorted(x for x in __names if x.startswith(__partial))[:200]:
        __out["options"].append({"label": __x, "type": "variable", "detail": ""})
    return json.dumps(__out)
__chatisa_complete(__cc_prefix)
`;

/**
 * Reads an uploaded file into a pandas DataFrame with the student's options and
 * either previews a sample or imports it under `name`. `__fileop` (mode, name,
 * fmt, opts) is set from JS; the bytes are at /tmp/__upload.
 */
const FILE_CODE = `
def __chatisa_load(op):
    import pandas as pd, json
    path = "/tmp/__upload"
    fmt = op["format"]
    opts = op.get("options", {}) or {}
    skip = int(opts.get("skipRows") or 0)
    header = 0 if opts.get("header", True) else None
    sheets = None
    if fmt == "csv":
        sep = opts.get("delimiter") or ","
        df = pd.read_csv(path, sep=sep, skiprows=skip, header=header)
    elif fmt == "json":
        df = pd.read_json(path)
    elif fmt == "xlsx":
        xl = pd.ExcelFile(path)
        sheets = list(xl.sheet_names)
        sheet = opts.get("sheet") or sheets[0]
        df = pd.read_excel(path, sheet_name=sheet, skiprows=skip, header=header)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        return json.dumps({"error": "That format is not supported in Python."})
    df.columns = [str(c) for c in df.columns]
    if op["mode"] == "import":
        globals()[op["name"]] = df
        return json.dumps({"ok": True, "nrows": int(len(df)), "ncols": int(df.shape[1])})
    __head = df.head(20)
    return json.dumps({
        "columns": list(df.columns),
        "rows": json.loads(__head.to_json(orient="records", date_format="iso")),
        "totalRows": int(len(df)),
        "sheets": sheets,
    })
__chatisa_load(__fileop)
`;

/** Loads the packages a given upload format needs, from our own origin. */
async function loadFormatPackages(pyodide, format) {
  await pyodide.loadPackage(["pandas"], QUIET);
  if (format === "parquet") await pyodide.loadPackage(["pyarrow"], QUIET);
  if (format === "xlsx") await installHostedWheel(pyodide, "openpyxl");
}

self.onmessage = async (event) => {
  const { id, code, keepState, withVariables, dataRequest, completeAt, prewarm, prepareCode, exportRequest, exportWorkspace, names, docRequest, fileOp } =
    event.data ?? {};

  // Documentation request: return the docstring for one symbol. Read-only
  // introspection (same surface as autocomplete); never runs the student's code.
  if (docRequest) {
    try {
      const pyodide = await getPyodide();
      // If we will need a module fallback (no live object) make sure it is present.
      const src = (docRequest.source || "").toLowerCase();
      if (src === "pandas") await pyodide.loadPackage(["pandas"], QUIET);
      if (src === "numpy") await pyodide.loadPackage(["numpy"], QUIET);
      pyodide.globals.set(
        "__doc_req",
        pyodide.toPy({
          name: docRequest.name || "",
          qualifier: docRequest.qualifier ?? null,
          source: docRequest.source ?? null,
        }),
      );
      const parsed = JSON.parse(pyodide.runPython(DOC_CODE));
      self.postMessage({ id, ok: true, doc: parsed });
    } catch {
      // Never surface an error for a doc lookup: the pane falls back to the blurb.
      self.postMessage({ id, ok: true, doc: { found: false } });
    }
    return;
  }

  // Export one DataFrame to CSV/TSV rather than running code.
  if (exportRequest) {
    try {
      const pyodide = await getPyodide();
      const sep = exportRequest.format === "tsv" ? "\t" : ",";
      const parsed = JSON.parse(
        pyodide.runPython(exportFrameCode(exportRequest.name, sep)),
      );
      if (parsed.error) {
        self.postMessage({ id, ok: false, error: parsed.error });
      } else {
        self.postMessage({ id, ok: true, exported: { text: parsed.text } });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Export the whole environment as a pickle (protocol 4) of every serializable value
  // (read-only). Objects that cannot be pickled are reported back, never silently dropped.
  if (exportWorkspace) {
    try {
      const pyodide = await getPyodide();
      const parsed = JSON.parse(pyodide.runPython(workspacePickleCode(names)));
      if (parsed.empty) {
        self.postMessage({ id, ok: true, exported: { empty: true } });
      } else {
        const bytes = pyodide.FS.readFile("/tmp/__ws.pkl");
        try {
          pyodide.FS.unlink("/tmp/__ws.pkl");
        } catch {
          // best-effort cleanup
        }
        self.postMessage({
          id,
          ok: true,
          exported: { bytes, skipped: parsed.skipped, empty: false },
        });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Restore an uploaded .pkl workspace: pickle.load its dict of values into the
  // namespace, handling name clashes per rule, gated on the trust confirmation.
  if (fileOp && fileOp.format === "pkl") {
    let pyodide;
    try {
      pyodide = await getPyodide();
      pyodide.FS.writeFile("/tmp/__upload", fileOp.bytes);
    } catch (error) {
      self.postMessage({ id, ok: false, error: friendlyError(error) });
      return;
    }
    try {
      const rule = fileOp.options?.conflict ?? "rename";
      const trusted = !!fileOp.options?.trusted;
      const parsed = JSON.parse(
        pyodide.runPython(restorePickleCode(fileOp.mode, trusted, rule)),
      );
      if (parsed.error) {
        self.postMessage({ id, ok: false, error: parsed.error });
      } else if (fileOp.mode === "import") {
        let variables = [];
        try {
          variables = JSON.parse(pyodide.runPython(INSPECT_VARS));
        } catch {
          variables = [];
        }
        self.postMessage({
          id,
          ok: true,
          result: { text: pyRestoreNote(parsed), variables },
        });
      } else {
        self.postMessage({
          id,
          ok: true,
          preview: { restore: true, columns: [], rows: [], members: parsed.members ?? [] },
        });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Upload: preview a sample of a file, or import it as a named DataFrame.
  if (fileOp) {
    // Text formats show their literal contents, so a preview can still show the
    // file even when parsing with the current options fails.
    const rawText =
      fileOp.format === "csv" || fileOp.format === "json"
        ? new TextDecoder().decode(fileOp.bytes.slice(0, 50000))
        : undefined;
    let pyodide;
    try {
      pyodide = await getPyodide();
      await loadFormatPackages(pyodide, fileOp.format);
      pyodide.FS.writeFile("/tmp/__upload", fileOp.bytes);
      pyodide.globals.set(
        "__fileop",
        pyodide.toPy({ mode: fileOp.mode, name: fileOp.name, format: fileOp.format, options: fileOp.options ?? {} }),
      );
    } catch (error) {
      self.postMessage({ id, ok: false, error: friendlyError(error) });
      return;
    }
    try {
      const parsed = JSON.parse(pyodide.runPython(FILE_CODE));
      if (parsed.error) {
        self.postMessage({ id, ok: false, error: parsed.error });
      } else if (fileOp.mode === "import") {
        let variables = [];
        try {
          variables = JSON.parse(pyodide.runPython(INSPECT_VARS));
        } catch {
          variables = [];
        }
        self.postMessage({
          id,
          ok: true,
          result: {
            text: `Loaded ${fileOp.name} (${parsed.nrows} rows x ${parsed.ncols} columns).`,
            variables,
          },
        });
      } else {
        self.postMessage({
          id,
          ok: true,
          preview: {
            rawText,
            columns: parsed.columns,
            rows: parsed.rows,
            totalRows: parsed.totalRows,
            sheets: parsed.sheets ?? undefined,
          },
        });
      }
    } catch (error) {
      // A parse failure during preview still returns the raw text, so the
      // student can see the file and adjust the options.
      if (fileOp.mode === "preview") {
        self.postMessage({
          id,
          ok: true,
          preview: {
            rawText,
            columns: [],
            rows: [],
            parseError: friendlyError(error),
          },
        });
      } else {
        self.postMessage({ id, ok: false, error: friendlyError(error) });
      }
    }
    return;
  }

  // Prewarm request (sent when the Python tab opens): load the runtime and the
  // common analytical stack in the background, so the first Run does not stall
  // on the interpreter and the default example's packages downloading.
  if (prewarm) {
    try {
      const pyodide = await getPyodide();
      await pyodide.loadPackage(["numpy", "pandas", "matplotlib", "micropip"], QUIET);
      // `prepareCode` (2026-07-26) asks for the packages ONE SPECIFIC snippet
      // needs, ahead of the student pressing Run. Same calls the run path makes,
      // so this only moves the wait earlier; it never changes what a run does.
      // Best-effort throughout: a prepare that fails must leave the run to try
      // again and report properly, not surface an error nobody asked for.
      if (typeof prepareCode === "string" && prepareCode.trim()) {
        await pyodide.loadPackagesFromImports(prepareCode, QUIET);
        for (const wheel of neededHostedWheels(prepareCode)) {
          await installHostedWheel(pyodide, wheel);
        }
        if (/\blxml\b/.test(prepareCode)) await pyodide.loadPackage(["lxml"], QUIET);
      }
      self.postMessage({ id, ok: true });
    } catch {
      self.postMessage({ id, ok: true }); // best-effort; Python still works
    }
    return;
  }

  // Autocomplete request: inspect the runtime for candidate members/names.
  if (completeAt) {
    try {
      const pyodide = await getPyodide();
      pyodide.globals.set("__cc_prefix", completeAt.prefix ?? "");
      const raw = pyodide.runPython(COMPLETE_CODE);
      self.postMessage({ id, ok: true, completions: JSON.parse(raw) });
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // Data-viewer request: return a page of a DataFrame rather than running code.
  if (dataRequest) {
    try {
      const pyodide = await getPyodide();
      const raw = pyodide.runPython(
        fetchFramePageCode(
          dataRequest.name,
          dataRequest.offset,
          dataRequest.limit,
        ),
      );
      const parsed = JSON.parse(raw);
      if (parsed.error) {
        self.postMessage({ id, ok: false, error: parsed.error });
      } else {
        self.postMessage({
          id,
          ok: true,
          data: {
            columns: parsed.columns,
            rows: parsed.rows,
            totalRows: parsed.total,
          },
        });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  const chunks = [];
  let pyodide;
  try {
    pyodide = await getPyodide();
  } catch {
    self.postMessage({
      id,
      ok: false,
      error:
        "The Python runtime could not be loaded. Check your connection and try again.",
    });
    return;
  }

  try {
    // Load packages first and quietly, so Pyodide's "Loading numpy..." progress
    // lines never reach the student's console. Only packages a snippet imports
    // are downloaded, and only from our origin.
    await pyodide.loadPackagesFromImports(code, {
      messageCallback: () => {},
      errorCallback: () => {},
    });
    // Packages Pyodide does not build (seaborn, openpyxl) are installed from our
    // own hosted wheels when the snippet needs them.
    for (const wheel of neededHostedWheels(code)) {
      await installHostedWheel(pyodide, wheel);
    }
    // BeautifulSoup(html, 'lxml') names the parser as a STRING, which
    // loadPackagesFromImports cannot see, so bs4 would report the parser
    // missing even though we mirror lxml. Any mention of lxml loads it.
    if (/\blxml\b/.test(code)) {
      await pyodide.loadPackage(["lxml"], QUIET);
    }
    // Route requests' cross-origin traffic through our guarded proxy (see the
    // bootstrap). The patch can only apply once `requests` is IMPORTED, and
    // loadPackagesFromImports merely installs it, so a snippet that imports
    // and fetches in the same run would race past an import-guarded patch:
    // when the code mentions requests, import it here first.
    if (/\brequests\b/.test(code)) {
      await pyodide.runPythonAsync(
        "import importlib\nimportlib.import_module('requests')\n__import__('_chatisa_net').patch()",
      );
    } else {
      pyodide.runPython("__import__('_chatisa_net').patch()");
    }
    // Record histogram bins and box statistics as matplotlib computes them
    // (#40), only when the code can plot: importing matplotlib costs time.
    if (/\b(matplotlib|seaborn|pyplot|plt|sns)\b|\.(plot|hist|boxplot)\s*\(/.test(code)) {
      pyodide.runPython(MPL_PATCH);
    }

    // Capture stdout and stderr only around the user's own code, in order.
    pyodide.setStdout({ batched: (s) => chunks.push(s) });
    pyodide.setStderr({ batched: (s) => chunks.push(s) });

    let tail = "";
    const value = await pyodide.runPythonAsync(code);
    if (value !== undefined && value !== null) {
      tail = typeof value?.toString === "function" ? value.toString() : String(value);
      // PyProxies hold WASM memory until released.
      if (typeof value?.destroy === "function") value.destroy();
    }

    let imageDataUrl;
    let plotInfo;
    let plotData;
    const captured = pyodide.runPython(CAPTURE_PLOT);
    if (captured) {
      const { png, info, data } = JSON.parse(captured);
      if (png) imageDataUrl = `data:image/png;base64,${png}`;
      if (info) plotInfo = info;
      if (data) plotData = data;
    }

    let variables;
    if (withVariables) {
      try {
        variables = JSON.parse(pyodide.runPython(INSPECT_VARS));
      } catch {
        variables = [];
      }
    }

    const printed = chunks.join("");
    const text =
      printed && tail
        ? `${printed}${printed.endsWith("\n") ? "" : "\n"}${tail}`
        : printed || tail;

    self.postMessage({
      id,
      ok: true,
      result: { text: text || undefined, imageDataUrl, plotInfo, plotData, variables },
    });
  } catch (error) {
    // Show whatever was printed before the failure, then the error itself.
    const printed = chunks.join("");
    const detail = friendlyError(error);
    self.postMessage({
      id,
      ok: false,
      error: printed ? `${printed}${printed.endsWith("\n") ? "" : "\n"}${detail}` : detail,
    });
  } finally {
    // Clear user-defined globals so the next run starts clean, unless this is a
    // Sandbox session, where variables are meant to carry over.
    if (!keepState) {
      try {
        pyodide.runPython(
          "for __k in [k for k in list(globals()) if not k.startswith('__')]:\n    del globals()[__k]",
        );
      } catch {
        // A failed cleanup must not mask the real result.
      }
    }
  }
};
