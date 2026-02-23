"""
TTB Alcohol Label Verification App - Streamlit entry.
Modes: Single label | Batch review.
"""
from collections import defaultdict
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.ui_utils import draw_bbox_on_image

st.set_page_config(
    page_title="TTB Label Verification",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { font-size: 1.05rem; }
    [data-testid="stSidebar"] { font-size: 1.05rem; }
    .status-banner {
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-pass { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .status-review { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .status-fail { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Demo / sample data
# ---------------------------------------------------------------------------

_DEMO_DATA = {
    "brand_name": "ABC",
    "class_type": "Straight Rye Whisky",
    "alcohol_pct": "45",
    "proof": "",
    "net_contents_ml": "750",
    "bottler_name": "ABC Distillery",
    "bottler_city": "Frederick",
    "bottler_state": "MD",
    "imported": False,
    "country_of_origin": "",
}


def main():
    st.title("TTB Label Verification")
    st.caption("Upload a label image, enter application data, and verify compliance.")

    mode = st.radio(
        "Mode",
        ["Single label", "Batch review"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    if mode == "Single label":
        _single_label_screen()
    else:
        _batch_screen()


def _single_label_screen():
    upload = st.file_uploader(
        "Upload label image",
        type=["png", "jpg", "jpeg"],
        key="single_upload",
        help="Supported formats: PNG, JPG. Photos of labels, scans, or digital artwork.",
    )

    # Image preview before submission
    if upload is not None:
        st.image(upload, width=320, caption="Uploaded image preview")

    # Demo fill button
    if st.button("Fill with sample data (ABC Distillery)", type="secondary"):
        st.session_state["demo_fill"] = True
        st.rerun()
    demo = st.session_state.pop("demo_fill", False)

    def _dv(key: str, default: str = "") -> str:
        return _DEMO_DATA.get(key, default) if demo else default

    with st.form("single_form"):
        st.markdown("#### Application data")
        brand = st.text_input("Brand name", value=_dv("brand_name"), placeholder="e.g. ABC")
        class_type = st.text_input("Class / type", value=_dv("class_type"), placeholder="e.g. Straight Rye Whisky")

        col1, col2, col3 = st.columns(3)
        with col1:
            alcohol_pct = st.text_input("Alcohol %", value=_dv("alcohol_pct"), placeholder="e.g. 45")
        with col2:
            proof = st.text_input("Proof", value=_dv("proof"), placeholder="e.g. 90")
        with col3:
            net_contents_ml = st.text_input("Net contents (mL)", value=_dv("net_contents_ml"), placeholder="e.g. 750")

        bottler_name = st.text_input("Bottler / Producer", value=_dv("bottler_name"), placeholder="e.g. ABC Distillery")
        c1, c2 = st.columns(2)
        with c1:
            bottler_city = st.text_input("City", value=_dv("bottler_city"), placeholder="e.g. Frederick")
        with c2:
            bottler_state = st.text_input("State", value=_dv("bottler_state"), placeholder="e.g. MD")

        imported = st.checkbox("Imported product", value=_DEMO_DATA.get("imported", False) if demo else False)
        country_of_origin = st.text_input("Country of origin (if imported)", value=_dv("country_of_origin"))

        with st.expander("Special statements (optional)"):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                sulfites = st.checkbox("Sulfites")
                fd_c_yellow_5 = st.checkbox("FD&C Yellow No. 5")
            with sc2:
                carmine = st.checkbox("Carmine")
                wood_treatment = st.checkbox("Wood treatment")
            with sc3:
                age_statement = st.checkbox("Age statement")
                neutral_spirits = st.checkbox("Neutral spirits %")

        submitted = st.form_submit_button("Check label", type="primary")

    if submitted and upload is not None:
        app_data = {
            "brand_name": brand or "",
            "class_type": class_type or "",
            "alcohol_pct": alcohol_pct or "",
            "proof": proof or "",
            "net_contents_ml": net_contents_ml or "",
            "bottler_name": bottler_name or "",
            "bottler_city": bottler_city or "",
            "bottler_state": bottler_state or "",
            "imported": imported,
            "country_of_origin": country_of_origin or "",
            "sulfites_required": sulfites,
            "fd_c_yellow_5_required": fd_c_yellow_5,
            "carmine_required": carmine,
            "wood_treatment_required": wood_treatment,
            "age_statement_required": age_statement,
            "neutral_spirits_required": neutral_spirits,
        }
        if "single_highlight_bbox" in st.session_state:
            del st.session_state["single_highlight_bbox"]
        with st.spinner("Analyzing label..."):
            try:
                from src.pipeline import run_pipeline
                result = run_pipeline(upload.getvalue(), app_data)
                st.session_state["last_single_result"] = result
                st.session_state["last_single_image_bytes"] = upload.getvalue()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
        result = st.session_state["last_single_result"]
        if result.get("error"):
            st.error("**OCR unavailable**")
            st.markdown(result["error"])
            st.markdown(
                "**To analyze real labels:** Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) "
                "(Windows) or `brew install tesseract` (Mac) / `apt install tesseract-ocr` (Linux), then add it to your PATH."
            )
            st.image(result.get("image") or upload.getvalue(), use_container_width=True, caption="Your label image")
            return
        _render_single_result(result, st.session_state["last_single_image_bytes"])
    elif submitted and upload is None:
        st.warning("Please upload a label image.")
    elif "last_single_result" in st.session_state:
        _render_single_result(st.session_state["last_single_result"], st.session_state.get("last_single_image_bytes"))


def _render_single_result(result: dict, image_bytes: bytes | None):
    overall = result.get("overall_status", "—")
    counts = result.get("counts", {})

    # Prominent status banner
    css_class = {
        "Ready to approve": "status-pass",
        "Needs review": "status-review",
        "Critical issues": "status-fail",
    }.get(overall, "status-review")
    st.markdown(f'<div class="status-banner {css_class}">{overall}</div>', unsafe_allow_html=True)
    st.caption(
        f"Pass: {counts.get('pass', 0)}  |  "
        f"Needs review: {counts.get('needs_review', 0)}  |  "
        f"Fail: {counts.get('fail', 0)}"
    )

    img = result.get("image")
    if img is None and image_bytes:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    highlight_bbox = st.session_state.get("single_highlight_bbox")

    col_img, col_list = st.columns([1, 1])
    with col_img:
        if img is not None:
            display_img = draw_bbox_on_image(img, highlight_bbox) if highlight_bbox else img
            st.image(display_img, use_container_width=True, caption="Label image")
        elif image_bytes:
            st.image(image_bytes, use_container_width=True, caption="Label image")

    with col_list:
        # Extracted vs expected comparison table
        extracted = result.get("extracted", {})
        _render_comparison_table(extracted, result)

        st.markdown("---")

        # Rule checklist by category
        by_category: dict[str, list] = defaultdict(list)
        for r in result.get("rule_results", []):
            by_category[r.get("category", "Other")].append(r)

        for cat in ("Identity", "Alcohol & contents", "Warning", "Origin", "Other"):
            rules = by_category.get(cat, [])
            if not rules:
                continue
            st.markdown(f"**{cat}**")
            for i, r in enumerate(rules):
                status = r.get("status", "pass")
                icon = "✅" if status == "pass" else "⚠️" if status == "needs_review" else "❌"
                rule_id = r.get("rule_id", "?")
                msg = r.get("message", "")
                bbox_ref = r.get("bbox_ref")
                st.markdown(f"{icon} **{rule_id}**: {msg}")
                if bbox_ref and img is not None:
                    if st.button("Show on label", key=f"show_{cat.replace(' ', '_')}_{i}"):
                        st.session_state["single_highlight_bbox"] = bbox_ref
                        st.rerun()
            st.markdown("")

        if highlight_bbox and st.button("Clear highlight"):
            if "single_highlight_bbox" in st.session_state:
                del st.session_state["single_highlight_bbox"]
            st.rerun()

    with st.expander("Raw OCR output"):
        ocr_blocks = result.get("ocr_blocks", [])
        if ocr_blocks:
            st.caption(f"{len(ocr_blocks)} text blocks detected.")
            texts = [b.get("text", "") for b in ocr_blocks]
            st.code(" | ".join(texts), language=None)
        else:
            st.info("No OCR blocks detected.")
    st.divider()


def _render_comparison_table(extracted: dict, result: dict):
    """Show extracted vs expected fields side-by-side."""
    fields = [
        ("Brand name", "brand_name"),
        ("Class / type", "class_type"),
        ("ABV %", "alcohol_pct"),
        ("Proof", "proof"),
        ("Net contents", "net_contents"),
        ("Bottler", "bottler"),
        ("Gov. warning", "government_warning"),
        ("Country of origin", "country_of_origin"),
    ]
    rows = []
    for label, key in fields:
        val = extracted.get(key, {})
        if isinstance(val, dict):
            val = val.get("value", "")
        rows.append({"Field": label, "Extracted from label": val or "(not found)"})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Batch screen
# ---------------------------------------------------------------------------

def _batch_screen():
    import zipfile
    from src.pipeline import run_pipeline

    st.subheader("Batch review")
    zip_upload = st.file_uploader("Upload ZIP of label images", type=["zip"], key="batch_zip")
    csv_upload = st.file_uploader("Upload CSV (label_id and application data)", type=["csv"], key="batch_csv")

    batch_results = st.session_state.get("batch_results", [])
    selected_id = st.session_state.get("batch_selected_id")

    if st.button("Run batch checks", key="batch_run"):
        if not zip_upload or not csv_upload:
            st.warning("Please upload both a ZIP of images and a CSV.")
        else:
            with st.spinner("Processing batch..."):
                try:
                    df = pd.read_csv(csv_upload)
                    df = _normalize_csv_columns(df)
                    z = zipfile.ZipFile(zip_upload, "r")
                    name_to_bytes = {info.filename: z.read(info.filename) for info in z.infolist() if not info.is_dir()}
                    z.close()
                    results = []
                    for _, row in df.iterrows():
                        label_id = str(row.get("label_id", row.iloc[0])).strip()
                        app_data = _row_to_app_data(row)
                        img_bytes = _find_image_for_label(name_to_bytes, label_id)
                        if img_bytes is None:
                            results.append({
                                "label_id": label_id,
                                "overall_status": "Critical issues",
                                "fail_count": 1,
                                "brand_name": app_data.get("brand_name", ""),
                                "class_type": app_data.get("class_type", ""),
                                "result": None,
                                "error": f"No image found for label_id '{label_id}'.",
                            })
                            continue
                        try:
                            r = run_pipeline(img_bytes, app_data)
                            fail_count = r.get("counts", {}).get("fail", 0)
                            results.append({
                                "label_id": label_id,
                                "overall_status": r.get("overall_status", "—"),
                                "fail_count": fail_count,
                                "brand_name": app_data.get("brand_name", ""),
                                "class_type": app_data.get("class_type", ""),
                                "result": r,
                                "error": None,
                            })
                        except Exception as e:
                            results.append({
                                "label_id": label_id,
                                "overall_status": "Critical issues",
                                "fail_count": 1,
                                "brand_name": app_data.get("brand_name", ""),
                                "class_type": app_data.get("class_type", ""),
                                "result": None,
                                "error": str(e),
                            })
                    st.session_state["batch_results"] = results
                    if "batch_selected_id" in st.session_state:
                        del st.session_state["batch_selected_id"]
                    st.success(f"Processed {len(results)} labels.")
                except Exception as e:
                    st.error(f"Batch failed: {e}")

    if batch_results:
        df_display = pd.DataFrame([
            {
                "label_id": r["label_id"],
                "brand_name": r["brand_name"],
                "class_type": r["class_type"],
                "overall_status": r["overall_status"],
                "failed_rules": r["fail_count"],
            }
            for r in batch_results
        ])
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "Critical issues", "Needs review", "Ready to approve"],
            key="batch_filter",
        )
        if status_filter != "All":
            df_display = df_display[df_display["overall_status"] == status_filter]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("**View detail**")
        label_ids = [r["label_id"] for r in batch_results]
        chosen = st.selectbox("Select a label to view full checklist", label_ids, key="batch_select")
        if st.button("Show detail", key="batch_show_detail"):
            st.session_state["batch_selected_id"] = chosen
            st.rerun()

    if selected_id and batch_results:
        st.divider()
        st.subheader(f"Detail: {selected_id}")
        match = next((r for r in batch_results if r["label_id"] == selected_id), None)
        if match and match.get("result"):
            _render_single_result(match["result"], None)
        elif match and match.get("error"):
            st.error(match["error"])
        if st.button("Back to batch table", key="batch_back"):
            if "batch_selected_id" in st.session_state:
                del st.session_state["batch_selected_id"]
            st.rerun()

    if not batch_results:
        st.caption(
            "Upload a ZIP of label images and a CSV with columns: label_id, brand_name, "
            "class_type, alcohol_pct, proof, net_contents_ml, bottler_name, bottler_city, "
            "bottler_state, imported, country_of_origin, and optional flags."
        )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _normalize_csv_columns(df):
    col_map = {
        "label id": "label_id", "label_id": "label_id",
        "brand name": "brand_name", "brand_name": "brand_name",
        "class type": "class_type", "class_type": "class_type",
        "alcohol pct": "alcohol_pct", "alcohol_pct": "alcohol_pct",
        "proof": "proof",
        "net contents": "net_contents_ml", "net_contents_ml": "net_contents_ml", "net contents ml": "net_contents_ml",
        "bottler name": "bottler_name", "bottler_name": "bottler_name",
        "bottler city": "bottler_city", "bottler_city": "bottler_city",
        "bottler state": "bottler_state", "bottler_state": "bottler_state",
        "imported": "imported",
        "country of origin": "country_of_origin", "country_of_origin": "country_of_origin",
        "sulfites required": "sulfites_required", "sulfites_required": "sulfites_required",
        "fd_c_yellow_5_required": "fd_c_yellow_5_required", "fd&c yellow 5": "fd_c_yellow_5_required",
        "carmine_required": "carmine_required", "carmine required": "carmine_required",
        "wood_treatment_required": "wood_treatment_required", "wood treatment": "wood_treatment_required",
        "age_statement_required": "age_statement_required", "age statement": "age_statement_required",
        "neutral_spirits_required": "neutral_spirits_required", "neutral spirits": "neutral_spirits_required",
    }
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    new_cols = {c: col_map[c] for c in df.columns if c in col_map}
    new_cols.update({c: col_map[c.replace(" ", "_")] for c in df.columns if c not in new_cols and c.replace(" ", "_") in col_map})
    if new_cols:
        df = df.rename(columns=new_cols)
    return df


def _row_to_app_data(row):
    def v(key, default=""):
        if key not in row:
            return default
        x = row[key]
        if pd.isna(x):
            return default
        if key == "imported":
            return str(x).strip().lower() in ("1", "true", "yes", "y")
        return str(x).strip()

    def b(key):
        return str(row.get(key, "")).strip().lower() in ("1", "true", "yes", "y") if key in row else False

    return {
        "brand_name": v("brand_name"),
        "class_type": v("class_type"),
        "alcohol_pct": v("alcohol_pct"),
        "proof": v("proof"),
        "net_contents_ml": v("net_contents_ml"),
        "bottler_name": v("bottler_name"),
        "bottler_city": v("bottler_city"),
        "bottler_state": v("bottler_state"),
        "imported": b("imported"),
        "country_of_origin": v("country_of_origin"),
        "sulfites_required": b("sulfites_required"),
        "fd_c_yellow_5_required": b("fd_c_yellow_5_required"),
        "carmine_required": b("carmine_required"),
        "wood_treatment_required": b("wood_treatment_required"),
        "age_statement_required": b("age_statement_required"),
        "neutral_spirits_required": b("neutral_spirits_required"),
    }


def _find_image_for_label(name_to_bytes: dict, label_id: str):
    import os
    label_id = label_id.strip()
    for fname, data in name_to_bytes.items():
        base = os.path.splitext(os.path.basename(fname))[0].strip()
        if base == label_id:
            return data
    lid_lower = label_id.lower()
    for fname, data in name_to_bytes.items():
        base = os.path.splitext(os.path.basename(fname))[0].strip().lower()
        if base == lid_lower:
            return data
    return None


if __name__ == "__main__":
    main()
