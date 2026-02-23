"""
TTB Alcohol Label Verification App - Streamlit entry.
Modes: Single label | Batch review.
"""
from collections import defaultdict
import io
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


st.set_page_config(
    page_title="TTB Label Verification",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { font-size: 1.05rem; }
    [data-testid="stSidebar"] { font-size: 0.95rem; }
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


_SAMPLE_PRESETS = {
    "test_1 — ABC Distillery (Spirits)": {
        "brand_name": "ABC Distillery",
        "class_type": "Single Barrel Straight Rye Whisky",
        "alcohol_pct": "45",
        "proof": "90",
        "net_contents_ml": "750 mL",
        "bottler_name": "ABC Distillery",
        "bottler_city": "Frederick",
        "bottler_state": "MD",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "Distilled Spirits",
    },
    "test_2 — Malt & Hop Brewery (Beer)": {
        "brand_name": "Malt & Hop Brewery",
        "class_type": "Pale Ale",
        "alcohol_pct": "5",
        "proof": "",
        "net_contents_ml": "24 fl oz",
        "bottler_name": "Malt & Hop Brewery",
        "bottler_city": "Hyattsville",
        "bottler_state": "MD",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "Beer / Malt Beverage",
    },
    "test_3 — Milo's Ale / Example Brewing (Beer)": {
        "brand_name": "Milo's Ale",
        "class_type": "Ale",
        "alcohol_pct": "5",
        "proof": "",
        "net_contents_ml": "1 qt",
        "bottler_name": "Example Brewing Company",
        "bottler_city": "",
        "bottler_state": "",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "Beer / Malt Beverage",
    },
    "test_4 — Tiger's Special Barleywine (Beer)": {
        "brand_name": "Tiger's Special",
        "class_type": "Barleywine Ale",
        "alcohol_pct": "9",
        "proof": "",
        "net_contents_ml": "12 fl oz",
        "bottler_name": "",
        "bottler_city": "",
        "bottler_state": "",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "Beer / Malt Beverage",
    },
    "test_5 — Downunder Winery (Wine, imported)": {
        "brand_name": "Downunder Winery",
        "class_type": "Red Wine",
        "alcohol_pct": "12",
        "proof": "",
        "net_contents_ml": "750 mL",
        "bottler_name": "OZ Imports",
        "bottler_city": "",
        "bottler_state": "",
        "imported": True,
        "country_of_origin": "Australia",
        "beverage_type": "Wine",
    },
    "test_6 — ABC Winery (Wine)": {
        "brand_name": "ABC Winery",
        "class_type": "Merlot",
        "alcohol_pct": "13",
        "proof": "",
        "net_contents_ml": "750 mL",
        "bottler_name": "ABC Winery",
        "bottler_city": "",
        "bottler_state": "",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "Wine",
    },
}

_BEVERAGE_TYPES = ["Distilled Spirits", "Wine", "Beer / Malt Beverage"]
_BEV_TYPE_KEY_MAP = {
    "Distilled Spirits": "spirits",
    "Wine": "wine",
    "Beer / Malt Beverage": "beer",
}


def _init_app_lists():
    if "applications_under_review" not in st.session_state:
        st.session_state["applications_under_review"] = []
    if "applications_approved" not in st.session_state:
        st.session_state["applications_approved"] = []
    if "applications_rejected" not in st.session_state:
        st.session_state["applications_rejected"] = []
    if "app_list_view" not in st.session_state:
        st.session_state["app_list_view"] = "under_review"
    if "selected_app_id" not in st.session_state:
        st.session_state["selected_app_id"] = None
    if "selected_app_bucket" not in st.session_state:
        st.session_state["selected_app_bucket"] = None
    if "adding_new_application" not in st.session_state:
        st.session_state["adding_new_application"] = False


def main():
    mode = st.sidebar.radio(
        "Mode",
        ["Single label", "Batch review"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Single label":
        _init_app_lists()
        _single_label_screen()
    else:
        _batch_screen()


def _single_label_screen():
    with st.sidebar:
        st.header("TTB Label Verification")

        _options = ["Applications Under Review", "Approved", "Rejected"]
        _view_to_option = {"under_review": _options[0], "approved": _options[1], "rejected": _options[2]}
        _current_view = st.session_state.get("app_list_view", "under_review")
        if st.session_state.get("app_list_radio") != _view_to_option.get(_current_view, _options[0]):
            st.session_state["app_list_radio"] = _view_to_option.get(_current_view, _options[0])

        view = st.radio(
            "Applications",
            _options,
            key="app_list_radio",
            label_visibility="collapsed",
        )
        view_key = {"Applications Under Review": "under_review", "Approved": "approved", "Rejected": "rejected"}[view]

        if view_key != st.session_state.get("app_list_view"):
            st.session_state["app_list_view"] = view_key
            st.session_state["selected_app_id"] = None
            st.session_state["selected_app_bucket"] = None
            st.session_state["adding_new_application"] = False

        st.divider()

        # Form in sidebar when adding new OR when an application is selected (auto-load)
        _show_form = st.session_state.get("adding_new_application") or st.session_state.get("selected_app_id")
        if _show_form:
            if st.session_state.get("adding_new_application"):
                upload = st.file_uploader(
                    "Upload label image",
                    type=["png", "jpg", "jpeg"],
                    key="single_upload",
                    help="PNG, JPG. Photos of labels, scans, or digital artwork.",
                )
                if upload is not None:
                    st.image(upload, width=220, caption="Preview")
            else:
                upload = None

            preset_names = ["(none)"] + list(_SAMPLE_PRESETS.keys())
            def _on_preset_change():
                v = st.session_state.get("preset_select")
                st.session_state["demo_fill"] = v if v and v != "(none)" else None
                st.session_state["selected_app_id"] = None
                st.session_state["selected_app_bucket"] = None

            chosen_preset = st.selectbox(
                "Application presets",
                preset_names,
                key="preset_select",
                on_change=_on_preset_change,
            )
            if st.button("Create new application", type="primary", key="btn_create_new"):
                st.session_state["adding_new_application"] = True
                st.session_state["selected_app_id"] = None
                st.session_state["selected_app_bucket"] = None
                st.session_state["demo_fill"] = None
                st.rerun()

        # Form fill: selected app (auto-load) > preset > empty
        def _get_form_fill():
            sel_id = st.session_state.get("selected_app_id")
            sel_bucket = st.session_state.get("selected_app_bucket")
            if sel_id and sel_bucket:
                apps = st.session_state.get(sel_bucket, [])
                entry = next((a for a in apps if a.get("id") == sel_id), None)
                if entry and entry.get("app_data"):
                    ad = entry["app_data"]
                    bev = ad.get("beverage_type", "spirits")
                    bev_display = {"spirits": "Distilled Spirits", "wine": "Wine", "beer": "Beer / Malt Beverage"}.get(bev, "Distilled Spirits")
                    return {
                        "brand_name": ad.get("brand_name", ""),
                        "class_type": ad.get("class_type", ""),
                        "alcohol_pct": ad.get("alcohol_pct", ""),
                        "proof": ad.get("proof", ""),
                        "net_contents_ml": ad.get("net_contents_ml", ""),
                        "bottler_name": ad.get("bottler_name", ""),
                        "bottler_city": ad.get("bottler_city", ""),
                        "bottler_state": ad.get("bottler_state", ""),
                        "imported": ad.get("imported", False),
                        "country_of_origin": ad.get("country_of_origin", ""),
                        "beverage_type": bev_display,
                    }
            demo_key = st.session_state.get("demo_fill")
            if demo_key and demo_key in _SAMPLE_PRESETS:
                return _SAMPLE_PRESETS[demo_key]
            return {}

        _form_fill = _get_form_fill()

        def _dv(key: str, default: str = "") -> str:
            if _form_fill:
                v = _form_fill.get(key, default)
                return str(v) if v is not None else default
            return default

        def _bev_idx() -> int:
            if _form_fill and _form_fill.get("beverage_type") in _BEVERAGE_TYPES:
                return _BEVERAGE_TYPES.index(_form_fill["beverage_type"])
            return 0

        if not _show_form:
            upload = None
            submitted = False
        else:
            with st.form("single_form"):
                beverage_type = st.selectbox("Beverage type", _BEVERAGE_TYPES, index=_bev_idx())
                brand = st.text_input("Brand name", value=_dv("brand_name"), placeholder="e.g. ABC Distillery")
                class_type = st.text_input("Class / type", value=_dv("class_type"), placeholder="e.g. Straight Rye Whisky")

                c1, c2 = st.columns(2)
                with c1:
                    alcohol_pct = st.text_input("Alcohol %", value=_dv("alcohol_pct"), placeholder="45")
                with c2:
                    proof = st.text_input("Proof", value=_dv("proof"), placeholder="90")

                net_contents_ml = st.text_input(
                    "Net contents",
                    value=_dv("net_contents_ml"),
                    placeholder="e.g. 750 mL, 1 QT, 12 FL OZ",
                )
                bottler_name = st.text_input("Bottler / Producer", value=_dv("bottler_name"), placeholder="ABC Distillery")

                c3, c4 = st.columns(2)
                with c3:
                    bottler_city = st.text_input("City", value=_dv("bottler_city"), placeholder="Frederick")
                with c4:
                    bottler_state = st.text_input("State", value=_dv("bottler_state"), placeholder="MD")

                imported = st.checkbox("Imported product", value=_form_fill.get("imported", False))
                country_of_origin = st.text_input("Country of origin", value=_dv("country_of_origin"))

                with st.expander("Conditional statements"):
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        sulfites = st.checkbox("Sulfites")
                        fd_c_yellow_5 = st.checkbox("FD&C Yellow No. 5")
                        carmine = st.checkbox("Cochineal / Carmine")
                    with sc2:
                        wood_treatment = st.checkbox("Wood treatment")
                        age_statement = st.checkbox("Age statement")
                        neutral_spirits = st.checkbox("Neutral spirits %")
                        aspartame = st.checkbox("Aspartame")

                    if beverage_type == "Wine":
                        appellation_required = st.checkbox("Appellation of origin")
                        varietal_required = st.checkbox("Varietal designation")
                    else:
                        appellation_required = False
                        varietal_required = False

                submitted = st.form_submit_button("Check label", type="primary", width="stretch")

    view = st.session_state.get("app_list_view", "under_review")
    adding_new = st.session_state.get("adding_new_application", False)
    if not adding_new:
        upload = None
        submitted = False

    if submitted and upload is not None and adding_new:
        app_data = {
            "beverage_type": _BEV_TYPE_KEY_MAP.get(beverage_type, "spirits"),
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
            "aspartame_required": aspartame,
            "appellation_required": appellation_required,
            "varietal_required": varietal_required,
        }
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
                "(Windows) or `brew install tesseract` (Mac) / `apt install tesseract-ocr` (Linux), then add to PATH."
            )
            st.image(result.get("image") or upload.getvalue(), width="stretch", caption="Your label image")
            return
        st.session_state["last_single_result"] = result
        st.session_state["last_single_image_bytes"] = upload.getvalue()
        st.session_state["last_single_app_data"] = app_data
        _render_single_result(result, st.session_state["last_single_image_bytes"])
        if st.button("Add to Under Review", type="primary", key="add_to_under_review"):
            entry = {
                "id": str(uuid.uuid4()),
                "brand_name": app_data.get("brand_name", ""),
                "class_type": app_data.get("class_type", ""),
                "overall_status": result.get("overall_status", "—"),
                "app_data": app_data,
                "image_bytes": upload.getvalue(),
                "result": {k: v for k, v in result.items() if k != "image"},
            }
            st.session_state["applications_under_review"].append(entry)
            st.session_state["adding_new_application"] = False
            st.session_state["selected_app_id"] = None
            st.rerun()
        return
    if submitted and upload is None and adding_new:
        st.warning("Please upload a label image.")
        return

    # Adding new: show placeholder when form in sidebar, no result yet
    if adding_new and "last_single_result" not in st.session_state:
        st.info("Fill the form in the sidebar and click **Check label**.")
        if st.button("Cancel", key="add_cancel_early"):
            st.session_state["adding_new_application"] = False
            st.rerun()
        return

    # List or detail view for Under Review / Approved / Rejected
    if view in ("under_review", "approved", "rejected"):
        bucket = {"under_review": "applications_under_review", "approved": "applications_approved", "rejected": "applications_rejected"}[view]
        apps = st.session_state.get(bucket, [])
        selected_id = st.session_state.get("selected_app_id")
        selected_bucket = st.session_state.get("selected_app_bucket")

        if selected_id and selected_bucket == bucket:
            entry = next((a for a in apps if a["id"] == selected_id), None)
            if entry:
                st.subheader(f"{entry.get('brand_name', '—')} — {entry.get('class_type', '')}")
                result_for_display = dict(entry["result"])
                result_for_display["image"] = None
                _approve_reject = {"entry": entry, "selected_id": selected_id} if view == "under_review" else None
                _render_single_result(result_for_display, entry.get("image_bytes"), approve_reject=_approve_reject)
                if st.button("Back to list", key="back_to_list"):
                    st.session_state["selected_app_id"] = None
                    st.session_state["selected_app_bucket"] = None
                    st.rerun()
            else:
                st.session_state["selected_app_id"] = None
                st.session_state["selected_app_bucket"] = None
                st.rerun()
        else:
            if view == "under_review":
                if st.button("Create new application", type="primary", key="btn_add_new"):
                    st.session_state["adding_new_application"] = True
                    st.session_state["selected_app_id"] = None
                    st.session_state["selected_app_bucket"] = None
                    st.rerun()
                st.divider()
            if not apps:
                st.info("No applications here yet." if view == "under_review" else ("No approved applications." if view == "approved" else "No rejected applications."))
            else:
                for a in apps:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{a.get('brand_name', '—')}** — {a.get('class_type', '')}  \n_{a.get('overall_status', '—')}_")
                        with col2:
                            if st.button("View", key=f"view_{a['id']}"):
                                st.session_state["selected_app_id"] = a["id"]
                                st.session_state["selected_app_bucket"] = bucket
                                st.rerun()
                        st.divider()
        return

    if adding_new and "last_single_result" in st.session_state:
        _render_single_result(st.session_state["last_single_result"], st.session_state.get("last_single_image_bytes"))
        if st.button("Add to Under Review", type="primary", key="add_to_under_review_2"):
            result = st.session_state["last_single_result"]
            app_data = st.session_state.get("last_single_app_data") or {}
            entry = {
                "id": str(uuid.uuid4()),
                "brand_name": app_data.get("brand_name", ""),
                "class_type": app_data.get("class_type", ""),
                "overall_status": result.get("overall_status", "—"),
                "app_data": app_data,
                "image_bytes": st.session_state.get("last_single_image_bytes") or b"",
                "result": {k: v for k, v in result.items() if k != "image"},
            }
            st.session_state["applications_under_review"].append(entry)
            st.session_state["adding_new_application"] = False
            st.session_state["selected_app_id"] = None
            st.rerun()
        if st.button("Cancel", key="add_cancel"):
            st.session_state["adding_new_application"] = False
            st.rerun()
        return

    st.title("TTB Label Verification")
    st.caption("Select **Applications Under Review** to add or review pending applications, **Approved** or **Rejected** to view resolved applications.")


def _render_single_result(result: dict, image_bytes: bytes | None, approve_reject: dict | None = None):
    """
    approve_reject: when set, show Approve/Reject in upper right. Expects {"entry", "selected_id"}.
    """
    overall = result.get("overall_status", "—")
    counts = result.get("counts", {})

    css_class = {
        "Ready to approve": "status-pass",
        "Needs review": "status-review",
        "Critical issues": "status-fail",
    }.get(overall, "status-review")

    # Top row: status on left, Approve/Reject on right when applicable
    if approve_reject:
        top_left, top_right = st.columns([3, 1])
    else:
        top_left = st.container()
    with top_left:
        st.markdown(f'<div class="status-banner {css_class}">{overall}</div>', unsafe_allow_html=True)
        st.caption(
            f"Pass: {counts.get('pass', 0)}  |  "
            f"Needs review: {counts.get('needs_review', 0)}  |  "
            f"Fail: {counts.get('fail', 0)}"
        )
    if approve_reject:
        with top_right:
            entry = approve_reject["entry"]
            selected_id = approve_reject["selected_id"]
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Approve", type="primary", key="btn_approve"):
                    st.session_state["applications_under_review"] = [a for a in st.session_state["applications_under_review"] if a["id"] != selected_id]
                    st.session_state["applications_approved"].append(entry)
                    st.session_state["selected_app_id"] = None
                    st.session_state["selected_app_bucket"] = None
                    st.rerun()
            with btn_col2:
                if st.button("Reject", key="btn_reject"):
                    st.session_state["applications_under_review"] = [a for a in st.session_state["applications_under_review"] if a["id"] != selected_id]
                    st.session_state["applications_rejected"].append(entry)
                    st.session_state["selected_app_id"] = None
                    st.session_state["selected_app_bucket"] = None
                    st.rerun()

    img = result.get("image")
    if img is None and image_bytes:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col_img, col_tabs = st.columns([1, 1])

    with col_img:
        if img is not None:
            st.image(img, width="stretch", caption="Label image")
        elif image_bytes:
            st.image(image_bytes, width="stretch", caption="Label image")

    with col_tabs:
        tab_check, tab_fields, tab_raw = st.tabs(["Checklist", "Extracted Fields", "Raw OCR"])

        with tab_check:
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
                    icon = {"pass": "✅", "needs_review": "⚠️", "fail": "❌"}.get(status, "⚠️")

                    rule_id = r.get("rule_id", "?")
                    ext_val = r.get("extracted_value", "")
                    app_val = r.get("app_value", "")
                    msg = r.get("message", "")

                    if ext_val or app_val:
                        ext_display = f'"{ext_val}"' if ext_val else "(not found)"
                        app_display = f'"{app_val}"' if app_val else "(not provided)"
                        comparison = f'label {ext_display} vs application {app_display}'
                        st.markdown(f'{icon} **{rule_id}**: {comparison}')
                        if msg and status != "pass":
                            st.caption(f"    _{msg}_")
                    else:
                        st.markdown(f"{icon} **{rule_id}**: {msg}")

        with tab_fields:
            extracted = result.get("extracted", {})
            _render_comparison_table(extracted, result)

        with tab_raw:
            ocr_blocks = result.get("ocr_blocks", [])
            if img is not None:
                with st.expander("Preprocessing (images fed to Tesseract)"):
                    from src.ocr import get_preprocessing_preview
                    orig, sharpened, binary = get_preprocessing_preview(img)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(orig, width="stretch", caption="1. Resized original (psm 3)")
                    with c2:
                        st.image(sharpened, width="stretch", caption="2. CLAHE + sharpen (psm 6)")
                    with c3:
                        st.image(binary, width="stretch", caption="3. Binarized (psm 6)")
            if ocr_blocks:
                st.caption(f"{len(ocr_blocks)} text blocks detected.")
                for b in ocr_blocks:
                    st.text(b.get("text", ""))
            else:
                st.info("No OCR blocks detected.")


def _render_comparison_table(extracted: dict, result: dict):
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
        display_val = val if val else "(not found)"
        if key == "government_warning" and len(display_val) > 80:
            display_val = display_val[:80] + "..."
        rows.append({"Field": label, "Extracted from label": display_val})
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Batch screen
# ---------------------------------------------------------------------------

def _batch_screen():
    import zipfile
    from src.pipeline import run_pipeline

    st.title("TTB Label Verification — Batch")

    with st.sidebar:
        st.header("Batch Upload")
        zip_upload = st.file_uploader("Upload ZIP of label images", type=["zip"], key="batch_zip")
        csv_upload = st.file_uploader("Upload CSV (application data)", type=["csv"], key="batch_csv")
        run_batch = st.button("Run batch checks", key="batch_run", type="primary", width="stretch")

    batch_results = st.session_state.get("batch_results", [])
    selected_id = st.session_state.get("batch_selected_id")

    if run_batch:
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
        st.dataframe(df_display, width="stretch", hide_index=True)

        st.markdown("**View detail**")
        label_ids = [r["label_id"] for r in batch_results]
        chosen = st.selectbox("Select a label", label_ids, key="batch_select")
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
            "bottler_state, imported, country_of_origin, beverage_type, and optional flags."
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
        "beverage type": "beverage_type", "beverage_type": "beverage_type",
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

    bev_type_raw = v("beverage_type", "spirits").lower()
    bev_type = "spirits"
    if "wine" in bev_type_raw:
        bev_type = "wine"
    elif "beer" in bev_type_raw or "malt" in bev_type_raw:
        bev_type = "beer"

    return {
        "beverage_type": bev_type,
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
