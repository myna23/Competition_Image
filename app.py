"""
IMDB Auto-Fill Tool — AI-Driven Image to Item Master Database
=============================================================
Upload product images → Claude Vision extracts 10 IMDB attributes → Export CSV / Excel

Run:  streamlit run app.py
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import base64, json, io, os, re
from PIL import Image, ImageEnhance
import pandas as pd
from datetime import datetime

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # handled at runtime with a friendly message

# ─── IMDB Schema ──────────────────────────────────────────────────────────────

IMDB_COLS = [
    "barcode",
    "category_type",
    "segment_type",
    "manufacturer",
    "brand",
    "product_name",
    "weight_and_unit",
    "packaging_type",
    "country_of_origin",
    "promotional_messages",
]

LABELS = {
    "barcode":               "Barcode",
    "category_type":         "Category Type",
    "segment_type":          "Segment Type",
    "manufacturer":          "Manufacturer",
    "brand":                 "Brand",
    "product_name":          "Product Name",
    "weight_and_unit":       "Weight & Unit",
    "packaging_type":        "Packaging Type",
    "country_of_origin":     "Country of Origin",
    "promotional_messages":  "Promotional Messages",
}

# ─── Normalisation helpers ─────────────────────────────────────────────────────

_COUNTRY_MAP: dict[str, str] = {
    "rsa": "South Africa", "s.a.": "South Africa", "s.a": "South Africa",
    "south african": "South Africa",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
    "england": "United Kingdom", "gb": "United Kingdom",
    "usa": "United States", "u.s.a.": "United States", "u.s.": "United States",
    "america": "United States",
    "prc": "China", "p.r.c.": "China", "pr china": "China",
    "uae": "United Arab Emirates", "u.a.e.": "United Arab Emirates",
    "eu": "European Union",
    "dr congo": "Democratic Republic of Congo", "drc": "Democratic Republic of Congo",
}

_BARCODE_RE = re.compile(r"^\d{8}$|^\d{12,14}$")

def _norm_country(val: str) -> str:
    key = val.strip().lower().rstrip(".")
    return _COUNTRY_MAP.get(key, val.strip().title())

def _norm_weight(val: str) -> str:
    """Convert imperial units to metric where possible."""
    s = val.strip()
    m = re.match(r"^([\d.]+)\s*oz$", s, re.I)
    if m:
        return f"{float(m.group(1)) * 28.3495:.0f}g"
    m = re.match(r"^([\d.]+)\s*fl\.?\s*oz$", s, re.I)
    if m:
        return f"{float(m.group(1)) * 29.5735:.0f}ml"
    m = re.match(r"^([\d.]+)\s*lbs?$", s, re.I)
    if m:
        return f"{float(m.group(1)) * 0.453592:.3f}kg"
    return s

def _validate_barcode(val: str) -> tuple[bool, str]:
    digits = re.sub(r"[\s\-]", "", val)
    return _BARCODE_RE.match(digits) is not None, digits

# ─── Claude Vision ────────────────────────────────────────────────────────────

MODELS = {
    "AI Vision Model — High Accuracy": "claude-opus-4-8",
    "AI Vision Model — Fast":          "claude-sonnet-4-6",
}

# ─── Demo data (used when Demo Mode is on — no API key needed) ────────────────

_DEMO_ROWS = [
    {
        "image": "S221234199_550719011.jpg",
        "barcode": "6030057221077", "category_type": "Personal Care",
        "segment_type": "Bar Soap", "manufacturer": "Meiji Ghana Limited",
        "brand": "MOK", "product_name": "MOK Fine Soap Rose",
        "weight_and_unit": "100g", "packaging_type": "Cardboard Box",
        "country_of_origin": "Ghana",
        "promotional_messages": "Natural and fresh douceur | More natural and fresh",
        **{f"_conf_{c}": 96 for c in ["barcode","category_type","segment_type","manufacturer","brand","product_name","weight_and_unit","packaging_type"]},
        "_conf_country_of_origin": 95, "_conf_promotional_messages": 92,
        "_avg_confidence": 95, "_low_conf_fields": "", "_needs_review": False,
        "_raw_bytes": None,
    },
    {
        "image": "S221712802_552034736.jpg",
        "barcode": "N/A", "category_type": "Food & Beverage",
        "segment_type": "Seasonings & Spices", "manufacturer": "N/A",
        "brand": "Mummy's Kitchen", "product_name": "Mummy's Kitchen Stew Seasoning Powder",
        "weight_and_unit": "10g", "packaging_type": "Sachet",
        "country_of_origin": "Ghana",
        "promotional_messages": "Stew Ragoût",
        **{f"_conf_{c}": 93 for c in ["category_type","segment_type","brand","product_name","weight_and_unit","packaging_type","country_of_origin"]},
        "_conf_barcode": 0, "_conf_manufacturer": 40, "_conf_promotional_messages": 91,
        "_avg_confidence": 78, "_low_conf_fields": "Barcode, Manufacturer",
        "_needs_review": True, "_raw_bytes": None,
    },
    {
        "image": "S222711495_554639782.jpg",
        "barcode": "N/A", "category_type": "Food & Beverage",
        "segment_type": "Juice Drinks", "manufacturer": "U-Fresh Company Limited",
        "brand": "U-Fresh", "product_name": "U-Fresh Orange Juice Drink",
        "weight_and_unit": "350ml", "packaging_type": "Plastic Bottle",
        "country_of_origin": "Ghana",
        "promotional_messages": "N/A",
        **{f"_conf_{c}": 94 for c in ["category_type","segment_type","manufacturer","brand","product_name","weight_and_unit","packaging_type","country_of_origin"]},
        "_conf_barcode": 0, "_conf_promotional_messages": 50,
        "_avg_confidence": 82, "_low_conf_fields": "Barcode",
        "_needs_review": True, "_raw_bytes": None,
    },
    {
        "image": "S222985766_556022646.jpg",
        "barcode": "N/A", "category_type": "Food & Beverage",
        "segment_type": "Chocolate Drinks", "manufacturer": "Atona Foods Investments",
        "brand": "Atona Food", "product_name": "This Way Chocolate Drink",
        "weight_and_unit": "40g", "packaging_type": "Sachet",
        "country_of_origin": "Ghana",
        "promotional_messages": "Good! Why Not Another?!",
        **{f"_conf_{c}": 92 for c in ["category_type","segment_type","manufacturer","brand","product_name","weight_and_unit","packaging_type","country_of_origin"]},
        "_conf_barcode": 0, "_conf_promotional_messages": 95,
        "_avg_confidence": 85, "_low_conf_fields": "Barcode",
        "_needs_review": True, "_raw_bytes": None,
    },
]

_SYSTEM = (
    "You are an expert retail product analyst specialising in extracting structured data "
    "from product images for Item Master Database (IMDB) cataloging. "
    "Always respond with valid JSON only — never add markdown fences or explanation."
)

_PROMPT = """Analyse this product image and return a JSON object with exactly these 10 IMDB fields.

Return ONLY the JSON — no markdown, no prose, just the object.

{
  "barcode":              {"value": "...", "confidence": 0},
  "category_type":        {"value": "...", "confidence": 0},
  "segment_type":         {"value": "...", "confidence": 0},
  "manufacturer":         {"value": "...", "confidence": 0},
  "brand":                {"value": "...", "confidence": 0},
  "product_name":         {"value": "...", "confidence": 0},
  "weight_and_unit":      {"value": "...", "confidence": 0},
  "packaging_type":       {"value": "...", "confidence": 0},
  "country_of_origin":    {"value": "...", "confidence": 0},
  "promotional_messages": {"value": "...", "confidence": 0}
}

Field definitions:
• barcode — Numeric barcode (EAN-8, EAN-13, UPC-A, UPC-E). Read every digit carefully. Use "N/A" if not visible.
• category_type — Broad category: "Food & Beverage", "Personal Care", "Household Cleaning", "Health & Wellness", "Pet Care", "Stationery", "Baby Care", "Electronics Accessories", etc.
• segment_type — Sub-category within category: "Dairy Products", "Carbonated Beverages", "Snacks & Confectionery", "Hair Care", "Surface Cleaners", "Vitamins & Supplements", "Infant Formula", etc.
• manufacturer — Legal company that manufactures the product (may differ from brand owner). Look for small print like "Manufactured by..." or "Distributed by...".
• brand — Primary brand name as displayed on the pack (use exact capitalisation).
• product_name — Full descriptive product name including variant/flavour (e.g. "Coca-Cola Zero Sugar Cherry 500ml Can").
• weight_and_unit — Net weight or volume with unit. Prefer metric: "500g", "1.5L", "250ml", "6×50g". Use "N/A" if not visible.
• packaging_type — Primary format: "Bottle", "Can", "Cardboard Box", "Foil Pouch", "Plastic Bag", "Glass Jar", "Tube", "Blister Pack", "Carton", "Sachet", "Tray", "Aerosol Can", "Tetra Pak".
• country_of_origin — Full country name (e.g. "South Africa", NOT "RSA"). Look for "Made in", "Product of", or flag graphics. Use "N/A" if not visible.
• promotional_messages — Any slogans, claims, certifications, or limited-time offers on pack (e.g. "New Improved Formula", "Buy 2 Get 1 Free", "Certified Organic", "Halal Certified"). Use "N/A" if none.

Confidence scoring:
• 90–100  Clearly legible and unambiguous
• 70–89   Visible but small or slightly obscured
• 40–69   Partially visible or inferred from context
• 1–39    Guessed from brand / product type knowledge
• 0       Cannot determine — set value to "N/A"

Normalisation rules:
• Spell out country names in full
• Use metric units when multiple unit systems are shown
• Barcode: digits only, no spaces or hyphens"""

_CONF_LOW  = 60   # below this → flag for review
_CONF_HIGH = 85   # above this → green


def _preprocess(img: Image.Image, enhance: bool = True) -> Image.Image:
    if max(img.size) > 1920:
        img.thumbnail((1920, 1920), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if enhance:
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call_claude(client, img_b64: str, model: str) -> tuple[dict | None, str | None]:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1400,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }],
        )
        raw = resp.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, str(e)


def _parse_result(result: dict, filename: str) -> dict:
    """Flatten Claude's extraction into a flat row dict."""
    row: dict = {"image": filename}
    flags: list[str] = []

    for col in IMDB_COLS:
        entry = result.get(col, {})
        if isinstance(entry, dict):
            val  = str(entry.get("value", "N/A")).strip() or "N/A"
            conf = max(0, min(100, int(entry.get("confidence", 0))))
        else:
            val, conf = "N/A", 0

        # Normalise
        if col == "country_of_origin" and val != "N/A":
            val = _norm_country(val)
        if col == "barcode" and val not in ("N/A", ""):
            ok, cleaned = _validate_barcode(val)
            val = cleaned
            if not ok:
                conf = min(conf, 40)
                flags.append(f"{LABELS[col]} (invalid format)")
        if col == "weight_and_unit" and val not in ("N/A", ""):
            val = _norm_weight(val)

        row[col] = val
        row[f"_conf_{col}"] = conf
        if conf < _CONF_LOW and val != "N/A":
            flags.append(LABELS[col])

    row["_avg_confidence"] = round(
        sum(row[f"_conf_{c}"] for c in IMDB_COLS) / len(IMDB_COLS)
    )
    row["_low_conf_fields"] = ", ".join(dict.fromkeys(flags))  # dedupe, preserve order
    row["_needs_review"] = bool(flags)
    return row


# ─── Duplicate detection ──────────────────────────────────────────────────────

def _find_dupes(new_rows: list[dict], existing: pd.DataFrame) -> list[dict]:
    # Normalise existing column names
    col_map = {c.lower().replace(" ", "_").replace("&", "and").replace("-", "_"): c
               for c in existing.columns}

    def _ex(row: pd.Series, field: str) -> str:
        mapped = col_map.get(field, field)
        return str(row.get(mapped, "")).strip().lower()

    results = []
    for row in new_rows:
        matches = []
        for _, ex in existing.iterrows():
            score, reasons = 0, []
            bc_new = str(row.get("barcode", "N/A")).strip()
            bc_ex  = _ex(ex, "barcode")
            if bc_new not in ("N/A", "") and bc_new.lower() == bc_ex:
                score += 60; reasons.append("barcode match")
            br_new = str(row.get("brand", "N/A")).lower().strip()
            br_ex  = _ex(ex, "brand")
            if br_new not in ("n/a", "") and br_new == br_ex:
                score += 20; reasons.append("brand match")
            pn_new = str(row.get("product_name", "N/A")).lower().strip()
            pn_ex  = _ex(ex, "product_name")
            if pn_new not in ("n/a", "") and pn_new == pn_ex:
                score += 15; reasons.append("product name match")
            w_new = str(row.get("weight_and_unit", "N/A")).lower().strip()
            w_ex  = _ex(ex, "weight_and_unit")
            if w_new not in ("n/a", "") and w_new == w_ex:
                score += 10; reasons.append("weight match")
            if score >= 50:
                matches.append({"score": score, "reasons": reasons, "record": ex.to_dict()})
        if matches:
            matches.sort(key=lambda x: x["score"], reverse=True)
            results.append({"image": row["image"], "matches": matches})
    return results


# ─── Excel export ─────────────────────────────────────────────────────────────

def _export_excel(df: pd.DataFrame) -> bytes:
    rename_main = {
        "image": "Source Image",
        "_needs_review": "Needs Review",
        "_low_conf_fields": "Low Confidence Fields",
        "_avg_confidence": "Avg Confidence (%)",
        **{c: LABELS[c] for c in IMDB_COLS},
    }
    rename_conf = {
        "image": "Source Image",
        "_avg_confidence": "Average (%)",
        **{f"_conf_{c}": LABELS[c] for c in IMDB_COLS},
    }

    main_cols = ["image"] + IMDB_COLS + ["_needs_review", "_low_conf_fields", "_avg_confidence"]
    conf_cols = ["image"] + [f"_conf_{c}" for c in IMDB_COLS] + ["_avg_confidence"]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df[[c for c in main_cols if c in df.columns]].rename(columns=rename_main).to_excel(
            writer, sheet_name="IMDB", index=False
        )
        df[[c for c in conf_cols if c in df.columns]].rename(columns=rename_conf).to_excel(
            writer, sheet_name="Confidence Scores", index=False
        )
        review = df[df["_needs_review"].astype(bool)] if "_needs_review" in df.columns else pd.DataFrame()
        if not review.empty:
            review[[c for c in main_cols if c in df.columns]].rename(columns=rename_main).to_excel(
                writer, sheet_name="Needs Review", index=False
            )
    return buf.getvalue()


# ─── Streamlit App ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IMDB Auto-Fill",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main .block-container { padding-top: 1.25rem; }
  div[data-testid="metric-container"] > div { background: #1e293b; border-radius: 8px; padding: .75rem 1rem; }
  .status-ok   { color: #22c55e; font-weight: 600; }
  .status-warn { color: #f59e0b; font-weight: 600; }
  .status-err  { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/barcode-scanner.png", width=64)
    st.title("IMDB Auto-Fill")
    st.caption("AI-powered product cataloging")
    st.divider()

    demo_mode = st.toggle(
        "Demo Mode (no API key needed)",
        value=False,
        help="Load sample product data to preview the full workflow without an API key",
    )

    if not demo_mode:
        # Resolve key: .env → Streamlit secrets → sidebar input
        try:
            _default_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
        except Exception:
            _default_key = os.getenv("ANTHROPIC_API_KEY", "")

        api_key = st.text_input(
            "API Key",
            value=_default_key,
            type="password",
            placeholder="Enter your API key…",
            help="Set via Streamlit secrets, .env file, or paste here",
        )
    else:
        api_key = ""

    model_label = st.selectbox("Model", list(MODELS), disabled=demo_mode)
    selected_model = MODELS[model_label]

    enhance = st.checkbox(
        "Auto-enhance images",
        value=True,
        help="Boost contrast & sharpness before analysis — improves OCR accuracy",
        disabled=demo_mode,
    )

    st.divider()
    st.markdown("**10 IMDB columns extracted:**")
    for col in IMDB_COLS:
        st.markdown(f"• {LABELS[col]}")

# ── Session state ─────────────────────────────────────────────────────────────

if "rows" not in st.session_state:
    st.session_state["rows"] = []

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏷️ AI-Driven Image-to-IMDB Auto-Fill")
st.markdown(
    "Upload product images to automatically extract all 10 Item Master Database attributes "
    "using AI Vision, then review, edit, and export to CSV or Excel."
)

tab_upload, tab_table, tab_dupes = st.tabs(
    ["📤  Upload & Extract", "📊  IMDB Table", "🔍  Duplicate Check"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Extract
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    uploaded_files = st.file_uploader(
        "Drop product images here",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Upload one or more product images. JPG, PNG, WebP, TIFF supported.",
        label_visibility="collapsed",
    )

    # ── Demo Mode banner ──────────────────────────────────────────────────────
    if demo_mode:
        st.info(
            "**Demo Mode is ON** — showing 4 sample products. "
            "Toggle off in the sidebar to process real images with your API key."
        )
        if st.button("▶ Load Demo Data", type="primary"):
            existing_names = {r["image"] for r in st.session_state["rows"]}
            for row in _DEMO_ROWS:
                if row["image"] not in existing_names:
                    st.session_state["rows"].append(dict(row))
            st.success("✅ Demo data loaded. Switch to the **IMDB Table** tab to review and export.")
        # Skip the rest of the upload logic in demo mode

    if not demo_mode and not uploaded_files:
        st.info("👆 Upload one or more product images to get started.")

    if not demo_mode and uploaded_files:
        existing_names = {r["image"] for r in st.session_state["rows"]}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        hdr_col, btn_col = st.columns([3, 1])
        hdr_col.markdown(
            f"**{len(uploaded_files)}** image(s) uploaded — "
            f"**{len(new_files)}** new, **{len(uploaded_files) - len(new_files)}** already processed"
        )

        run = btn_col.button(
            f"▶ Extract {len(new_files)} image(s)",
            disabled=(not new_files) or (not api_key) or (_anthropic is None),
            type="primary",
            use_container_width=True,
        )

        if _anthropic is None:
            st.error("The `anthropic` package is not installed. Run: `pip install anthropic`")
        elif not api_key:
            st.warning("Enter your API key in the sidebar to proceed.")

        if run and new_files and api_key and _anthropic:
            client = _anthropic.Anthropic(api_key=api_key)
            progress_bar = st.progress(0.0, text="Starting…")
            status_msg   = st.empty()

            for idx, file in enumerate(new_files):
                frac = idx / len(new_files)
                progress_bar.progress(frac, text=f"Processing {file.name} ({idx + 1}/{len(new_files)})…")
                status_msg.info(f"Analysing **{file.name}** ({idx + 1}/{len(new_files)})…")

                try:
                    raw_bytes = file.read()
                    img = Image.open(io.BytesIO(raw_bytes))
                    img = _preprocess(img, enhance=enhance)
                    img_b64 = _to_b64(img)

                    result, err = _call_claude(client, img_b64, selected_model)

                    if err:
                        st.error(f"**{file.name}**: {err}")
                        st.session_state["rows"].append({
                            "image": file.name,
                            **{c: "ERROR" for c in IMDB_COLS},
                            **{f"_conf_{c}": 0 for c in IMDB_COLS},
                            "_avg_confidence": 0,
                            "_low_conf_fields": err,
                            "_needs_review": True,
                            "_raw_bytes": raw_bytes,
                        })
                    else:
                        row = _parse_result(result, file.name)
                        row["_raw_bytes"] = raw_bytes
                        st.session_state["rows"].append(row)

                except Exception as exc:
                    st.error(f"**{file.name}**: Failed — {exc}")

            progress_bar.progress(1.0, text="Done!")
            status_msg.success(
                f"✅ Processed {len(new_files)} image(s). "
                "Switch to the **IMDB Table** tab to review and export."
            )

    # ── Per-image result cards (real + demo) ──────────────────────────────────
    if st.session_state["rows"]:
        st.divider()
        st.subheader("Extraction Results")

        uploaded_names = {f.name: f for f in (uploaded_files or [])}

        for row in st.session_state["rows"]:
            icon  = "⚠️" if row.get("_needs_review") else "✅"
            avg   = row.get("_avg_confidence", 0)
            _display_name = row.get("product_name") or row["image"]
            label = f"{icon}  {_display_name}  —  Avg confidence: {avg}%"

            with st.expander(label, expanded=row.get("_needs_review", False)):
                img_col, data_col = st.columns([1, 2])

                with img_col:
                    raw = row.get("_raw_bytes")
                    _local_img = os.path.join(
                        os.path.dirname(__file__), "product_images", row["image"]
                    )
                    if raw:
                        st.image(raw, use_container_width=True)
                    elif row["image"] in uploaded_names:
                        try:
                            f = uploaded_names[row["image"]]
                            f.seek(0)
                            st.image(f.read(), use_container_width=True)
                        except Exception:
                            st.write("*(preview unavailable)*")
                    elif os.path.exists(_local_img):
                        st.image(_local_img, use_container_width=True)
                    else:
                        brand = row.get("brand", "")
                        product = row.get("product_name", "Product")
                        st.markdown(
                            f"""<div style="border:2px dashed #555;border-radius:10px;
                            padding:30px 10px;text-align:center;color:#aaa;font-size:13px;">
                            🏷️<br><br><b style="color:#ddd;font-size:15px;">{brand}</b>
                            <br><span style="font-size:12px;">{product}</span>
                            <br><br><span style="font-size:11px;">Image pending upload</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                with data_col:
                    for col in IMDB_COLS:
                        val  = row.get(col, "N/A")
                        conf = row.get(f"_conf_{col}", 0)
                        dot  = "🟢" if conf >= _CONF_HIGH else ("🟡" if conf >= _CONF_LOW else "🔴")
                        st.markdown(f"**{LABELS[col]}**: {val} {dot} *{conf}%*")

                    if row.get("_low_conf_fields"):
                        st.warning(f"Low-confidence fields: {row['_low_conf_fields']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMDB Table
# ══════════════════════════════════════════════════════════════════════════════
with tab_table:
    rows = st.session_state["rows"]

    if not rows:
        st.info("No data yet. Upload and process images in the **Upload & Extract** tab.")
    else:
        # Summary metrics
        total       = len(rows)
        n_review    = sum(1 for r in rows if r.get("_needs_review"))
        avg_conf    = sum(r.get("_avg_confidence", 0) for r in rows) / total
        n_ready     = total - n_review

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Products Processed", total)
        c2.metric("Avg Confidence", f"{avg_conf:.0f}%")
        c3.metric("Needs Human Review", n_review)
        c4.metric("Ready to Export", n_ready)
        st.divider()

        # Build editable DataFrame
        display_rows = []
        for r in rows:
            display_rows.append({
                "Source Image":         r["image"],
                **{LABELS[c]: r.get(c, "N/A") for c in IMDB_COLS},
                "Avg Conf %":           r.get("_avg_confidence", 0),
                "⚠️ Needs Review":      r.get("_needs_review", False),
            })
        display_df = pd.DataFrame(display_rows)

        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Source Image":     st.column_config.TextColumn(width="medium"),
                "Avg Conf %":       st.column_config.ProgressColumn(
                                        "Avg Conf %", min_value=0, max_value=100, width="small"
                                    ),
                "⚠️ Needs Review":  st.column_config.CheckboxColumn(width="small"),
                **{LABELS[c]: st.column_config.TextColumn(LABELS[c], width="medium")
                   for c in IMDB_COLS},
            },
        )

        st.divider()

        # Sync edits back to internal rows
        def _sync(orig: list[dict], edited: pd.DataFrame) -> list[dict]:
            synced = []
            for i, r in enumerate(orig):
                updated = dict(r)
                if i < len(edited):
                    er = edited.iloc[i]
                    for col in IMDB_COLS:
                        updated[col] = str(er.get(LABELS[col], r.get(col, "N/A")))
                    updated["_needs_review"] = bool(er.get("⚠️ Needs Review", r.get("_needs_review", False)))
                synced.append(updated)
            return synced

        synced = _sync(rows, edited_df)
        full_df = pd.DataFrame(synced)

        csv_df = pd.DataFrame([
            {"Source Image": r["image"], **{LABELS[c]: r.get(c, "N/A") for c in IMDB_COLS}}
            for r in synced
        ])
        csv_bytes  = csv_df.to_csv(index=False).encode("utf-8-sig")
        xlsx_bytes = _export_excel(full_df)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        dl1, dl2, clr = st.columns([1, 1, 1])
        dl1.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name=f"IMDB_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl2.download_button(
            "⬇ Download Excel",
            data=xlsx_bytes,
            file_name=f"IMDB_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        if clr.button("🗑 Clear all results", use_container_width=True):
            st.session_state["rows"] = []
            st.rerun()

        # Field-level confidence heatmap
        with st.expander("📈 Field confidence breakdown"):
            hm_data = {
                LABELS[c]: [r.get(f"_conf_{c}", 0) for r in rows]
                for c in IMDB_COLS
            }
            hm_df = pd.DataFrame(hm_data, index=[r.get("product_name") or r["image"] for r in rows])
            try:
                st.dataframe(
                    hm_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(hm_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Duplicate Check
# ══════════════════════════════════════════════════════════════════════════════
with tab_dupes:
    st.subheader("🔍 Duplicate Detection")
    st.markdown(
        "Upload your existing IMDB file (CSV or Excel) to identify whether any newly "
        "extracted products are already in your database."
    )

    existing_file = st.file_uploader(
        "Existing IMDB (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="existing_imdb",
        label_visibility="visible",
    )

    if existing_file:
        try:
            if existing_file.name.lower().endswith(".csv"):
                existing_df = pd.read_csv(existing_file)
            else:
                existing_df = pd.read_excel(existing_file)
            st.success(f"Loaded **{len(existing_df)}** existing records from *{existing_file.name}*")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            existing_df = None

        if existing_df is not None:
            if not st.session_state["rows"]:
                st.info("Process images first (Upload & Extract tab), then run the duplicate check.")
            else:
                if st.button("🔍 Check for Duplicates", type="primary"):
                    dupes = _find_dupes(st.session_state["rows"], existing_df)
                    if not dupes:
                        st.success("✅ No duplicates found — all extracted products appear to be new records.")
                    else:
                        st.warning(f"⚠️ Found **{len(dupes)}** potential duplicate(s):")
                        for d in dupes:
                            with st.expander(f"🔁 {d['image']} — {len(d['matches'])} match(es)"):
                                for m in d["matches"]:
                                    badge = f"Match score: {m['score']} | " + ", ".join(m["reasons"])
                                    st.markdown(f"**{badge}**")
                                    st.json(m["record"])

    else:
        st.info(
            "No existing IMDB uploaded yet. "
            "Upload a CSV or Excel file with existing product records to compare against."
        )
        with st.expander("Expected column names"):
            st.markdown("The existing IMDB should have columns matching these names (case-insensitive):")
            for col in IMDB_COLS:
                st.code(col)
