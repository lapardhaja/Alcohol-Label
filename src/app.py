"""
BottleProof — Computer Based Alcohol Label Validation.
Modes: Single Labeling | Batch Labeling.
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

_LOGO_PATH = _root / "assets" / "logo.png"

st.set_page_config(
    page_title="BottleProof — Computer Based Alcohol Label Validation",
    page_icon=str(_LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { font-size: 1.05rem; }
    [data-testid="stSidebar"] { font-size: 0.95rem; }
    .brand-bottleproof { color: #28a745; font-weight: 700; }
    .brand-proof { color: #fd7e14; font-weight: 700; }
    .brand-subtitle { font-size: 1.45rem; font-weight: 500; }
    .st-key-batch_run_btn button { background-color: #28a745 !important; border-color: #28a745 !important; color: white !important; }
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
    [class*="approve_btn"] button { background-color: #28a745 !important; border-color: #28a745 !important; color: white !important; }
    [class*="decline_btn"] button { background-color: #dc3545 !important; border-color: #dc3545 !important; color: white !important; }
    html, .stApp { scrollbar-width: 14px; }
    ::-webkit-scrollbar { width: 14px; height: 14px; }
    ::-webkit-scrollbar-thumb { background: #ccc; border-radius: 7px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
</style>
""", unsafe_allow_html=True)


def _render_brand_title(mode: str):
    """Render BottleProof title (Bottle=green, Proof=orange) and larger subtitle. mode: 'single' | 'batch'."""
    suffix = "Single Labeling" if mode == "single" else "Batch Labeling"
    st.markdown(
        f'<h1 style="margin-bottom: 0.25rem;">'
        f'<span class="brand-bottleproof">Bottle</span><span class="brand-proof">Proof</span> — {suffix}'
        f'</h1>'
        f'<p class="brand-subtitle" style="margin-top: 0.25rem;">Computer Based Alcohol Label Validation</p>',
        unsafe_allow_html=True,
    )


_SAMPLE_PRESETS = {
    "test_1 — ABC Distillery (Spirits)": {
        "brand_name": "ABC Distillery",
        "class_type": "Single Barrel Straight Rye Whisky",
        "alcohol_pct": "45",
        "proof": "",
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


def _get_form_fill_from_session():
    """Return preset or selected-app form data for form prefill."""
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


def _init_app_lists():
    from src.storage import load_applications
    if "applications_under_review" not in st.session_state:
        data = load_applications()
        st.session_state["applications_under_review"] = data["applications_under_review"]
        st.session_state["applications_approved"] = data["applications_approved"]
        st.session_state["applications_rejected"] = data["applications_rejected"]
    if "app_list_view" not in st.session_state:
        st.session_state["app_list_view"] = "create_new"  # enter create new immediately
    if "selected_app_id" not in st.session_state:
        st.session_state["selected_app_id"] = None
    if "selected_app_bucket" not in st.session_state:
        st.session_state["selected_app_bucket"] = None
    if "adding_new_application" not in st.session_state:
        st.session_state["adding_new_application"] = True  # default to create new


def main():
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=180)
        mode = st.radio(
            "Mode",
            ["Single Labeling", "Batch Labeling"],
            horizontal=True,
            label_visibility="collapsed",
        )
    if mode == "Single Labeling":
        _init_app_lists()
        _single_label_screen()
    else:
        _batch_screen()


def _single_label_screen():
    with st.sidebar:
        view_key = st.session_state.get("app_list_view", "create_new")
        if view_key != "create_new":
            if st.button("New label", key="sidebar_new_label"):
                st.session_state["app_list_view"] = "create_new"
                st.session_state["selected_app_id"] = None
                st.session_state["selected_app_bucket"] = None
                st.rerun()
        if view_key == "create_new":
            if st.button("Reset", key="single_reset", width="stretch"):
                for key in ("last_single_result", "last_single_image_bytes", "last_single_app_data", "last_single_entry_id"):
                    st.session_state.pop(key, None)
                st.rerun()
        st.divider()

        _show_form = (view_key == "create_new") or st.session_state.get("selected_app_id")
        if _show_form:
            preset_names = ["New Application"] + list(_SAMPLE_PRESETS.keys())
            _create_keys = (
                "create_beverage_type", "create_brand_name", "create_class_type", "create_alcohol_pct", "create_proof",
                "create_net_contents_ml", "create_bottler_name", "create_bottler_city", "create_bottler_state",
                "create_imported", "create_country_of_origin", "create_sulfites", "create_fd_c_yellow_5", "create_carmine",
                "create_wood_treatment", "create_age_statement", "create_neutral_spirits", "create_aspartame",
                "create_appellation_required", "create_varietal_required",
            )

            def _on_preset_change():
                v = st.session_state.get("preset_select")
                st.session_state["demo_fill"] = v if v and v != "New Application" else None
                st.session_state["selected_app_id"] = None
                st.session_state["selected_app_bucket"] = None
                st.session_state["preset_just_changed"] = True
                for k in _create_keys:
                    st.session_state.pop(k, None)

            st.selectbox(
                "Application presets",
                preset_names,
                key="preset_select",
                on_change=_on_preset_change,
            )
            if view_key == "create_new":
                _form_fill = _get_form_fill_from_session()
                ss = st.session_state
                _bool_keys = ("create_imported", "create_sulfites", "create_fd_c_yellow_5", "create_carmine", "create_wood_treatment", "create_age_statement", "create_neutral_spirits", "create_aspartame", "create_appellation_required", "create_varietal_required")
                if _form_fill:
                    ss.setdefault("create_beverage_type", _form_fill.get("beverage_type") or _BEVERAGE_TYPES[0])
                    ss.setdefault("create_brand_name", _form_fill.get("brand_name") or "")
                    ss.setdefault("create_class_type", _form_fill.get("class_type") or "")
                    ss.setdefault("create_alcohol_pct", _form_fill.get("alcohol_pct") or "")
                    ss.setdefault("create_proof", _form_fill.get("proof") or "")
                    ss.setdefault("create_net_contents_ml", _form_fill.get("net_contents_ml") or "")
                    ss.setdefault("create_bottler_name", _form_fill.get("bottler_name") or "")
                    ss.setdefault("create_bottler_city", _form_fill.get("bottler_city") or "")
                    ss.setdefault("create_bottler_state", _form_fill.get("bottler_state") or "")
                    ss.setdefault("create_imported", _form_fill.get("imported", False))
                    ss.setdefault("create_country_of_origin", _form_fill.get("country_of_origin") or "")
                    for _k in _bool_keys:
                        ss.setdefault(_k, False)
                    if ss.get("preset_just_changed"):
                        ss["create_details_last_saved"] = {k: ss.get(k) for k in _create_keys}
                        ss["preset_just_changed"] = False
                else:
                    ss.setdefault("create_beverage_type", _BEVERAGE_TYPES[0])
                    for _k in _create_keys:
                        if _k == "create_beverage_type":
                            continue
                        ss.setdefault(_k, False if _k in _bool_keys else "")
                    ss["preset_just_changed"] = False

                def _bev_idx() -> int:
                    _bev = ss.get("create_beverage_type", _BEVERAGE_TYPES[0])
                    return _BEVERAGE_TYPES.index(_bev) if _bev in _BEVERAGE_TYPES else 0

                with st.expander("Application details", expanded=True):
                    _bev = ss.get("create_beverage_type", _BEVERAGE_TYPES[0])
                    _bev_idx_val = _BEVERAGE_TYPES.index(_bev) if _bev in _BEVERAGE_TYPES else 0
                    st.selectbox("Beverage type", _BEVERAGE_TYPES, index=_bev_idx_val, key="create_beverage_type")
                    st.text_input("Brand name", placeholder="e.g. ABC Distillery", key="create_brand_name")
                    st.text_input("Class / type", placeholder="e.g. Straight Rye Whisky", key="create_class_type")
                    _cur_bev = ss.get("create_beverage_type", _BEVERAGE_TYPES[0])
                    _abv_label = "Alcohol % (optional)" if _cur_bev == "Beer / Malt Beverage" else "Alcohol %"
                    if _cur_bev == "Distilled Spirits":
                        c1, c2 = st.columns(2)
                        with c1:
                            st.text_input(_abv_label, placeholder="45", key="create_alcohol_pct")
                        with c2:
                            st.text_input("Proof", placeholder="90", key="create_proof")
                    else:
                        st.text_input(_abv_label, placeholder="45", key="create_alcohol_pct")
                    st.text_input("Net contents", placeholder="e.g. 750 mL, 1 QT, 12 FL OZ", key="create_net_contents_ml")
                    st.text_input("Bottler / Producer", placeholder="ABC Distillery", key="create_bottler_name")
                    c3, c4 = st.columns(2)
                    with c3:
                        st.text_input("City", placeholder="Frederick", key="create_bottler_city")
                    with c4:
                        st.text_input("State", placeholder="MD", key="create_bottler_state")
                    st.checkbox("Imported product", key="create_imported")
                    st.text_input("Country of origin", key="create_country_of_origin")
                    with st.expander("Conditional statements"):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.checkbox("Sulfites", key="create_sulfites")
                            st.checkbox("FD&C Yellow No. 5", key="create_fd_c_yellow_5")
                            st.checkbox("Cochineal / Carmine", key="create_carmine")
                        with sc2:
                            if _cur_bev == "Distilled Spirits":
                                st.checkbox("Wood treatment", key="create_wood_treatment")
                                st.checkbox("Age statement", key="create_age_statement")
                                st.checkbox("Neutral spirits %", key="create_neutral_spirits")
                            if _cur_bev == "Beer / Malt Beverage":
                                st.checkbox("Aspartame", key="create_aspartame")
                        if _cur_bev == "Wine":
                            st.checkbox("Appellation of origin", key="create_appellation_required")
                            st.checkbox("Varietal designation", key="create_varietal_required")

                # Save changes: show when Application details differ from last saved
                if view_key == "create_new":
                    _current_snapshot = {k: ss.get(k) for k in _create_keys}
                    if "create_details_last_saved" not in ss:
                        ss["create_details_last_saved"] = _current_snapshot
                    _last_saved = ss.get("create_details_last_saved") or {}
                    _dirty = _current_snapshot != _last_saved
                    if _dirty:
                        if st.button("Save changes", key="sidebar_save_changes", type="primary"):
                            ss["create_details_last_saved"] = {k: ss.get(k) for k in _create_keys}
                            st.success("Changes saved.")
                            st.rerun()
        if view_key != "create_new" and st.session_state.get("selected_app_id"):
            _form_fill = _get_form_fill_from_session()
            def _dv(key: str, default: str = "") -> str:
                v = _form_fill.get(key, default) if _form_fill else default
                return str(v) if v is not None else default
            def _bev_idx() -> int:
                if _form_fill and _form_fill.get("beverage_type") in _BEVERAGE_TYPES:
                    return _BEVERAGE_TYPES.index(_form_fill["beverage_type"])
                return 0
            with st.form("sidebar_form"):
                beverage_type = st.selectbox("Beverage type", _BEVERAGE_TYPES, index=_bev_idx())
                brand = st.text_input("Brand name", value=_dv("brand_name"), placeholder="e.g. ABC Distillery")
                class_type = st.text_input("Class / type", value=_dv("class_type"), placeholder="e.g. Straight Rye Whisky")
                _prev_bev = _form_fill.get("beverage_type", "Distilled Spirits") if _form_fill else "Distilled Spirits"
                _abv_lbl = "Alcohol % (optional)" if _prev_bev == "Beer / Malt Beverage" else "Alcohol %"
                if _prev_bev == "Distilled Spirits":
                    c1, c2 = st.columns(2)
                    with c1:
                        alcohol_pct = st.text_input(_abv_lbl, value=_dv("alcohol_pct"), placeholder="45")
                    with c2:
                        proof = st.text_input("Proof", value=_dv("proof"), placeholder="90")
                else:
                    alcohol_pct = st.text_input(_abv_lbl, value=_dv("alcohol_pct"), placeholder="45")
                    proof = ""
                net_contents_ml = st.text_input("Net contents", value=_dv("net_contents_ml"), placeholder="e.g. 750 mL")
                bottler_name = st.text_input("Bottler / Producer", value=_dv("bottler_name"), placeholder="ABC Distillery")
                c3, c4 = st.columns(2)
                with c3:
                    bottler_city = st.text_input("City", value=_dv("bottler_city"), placeholder="Frederick")
                with c4:
                    bottler_state = st.text_input("State", value=_dv("bottler_state"), placeholder="MD")
                imported = st.checkbox("Imported product", value=_form_fill.get("imported", False))
                country_of_origin = st.text_input("Country of origin", value=_dv("country_of_origin"))
                wood_treatment = age_statement = neutral_spirits = aspartame = False
                appellation_required = varietal_required = False
                with st.expander("Conditional statements"):
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        sulfites = st.checkbox("Sulfites")
                        fd_c_yellow_5 = st.checkbox("FD&C Yellow No. 5")
                        carmine = st.checkbox("Cochineal / Carmine")
                    with sc2:
                        if _prev_bev == "Distilled Spirits":
                            wood_treatment = st.checkbox("Wood treatment")
                            age_statement = st.checkbox("Age statement")
                            neutral_spirits = st.checkbox("Neutral spirits %")
                        if _prev_bev == "Beer / Malt Beverage":
                            aspartame = st.checkbox("Aspartame")
                    if _prev_bev == "Wine":
                        appellation_required = st.checkbox("Appellation of origin")
                        varietal_required = st.checkbox("Varietal designation")
                submitted = st.form_submit_button("Check label", type="primary", width="stretch")
            upload = None  # sidebar form has no upload when viewing selected item
        else:
            upload = None
            submitted = False

    view = st.session_state.get("app_list_view", "create_new")
    adding_new = view == "create_new"
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
        entry_id = st.session_state.get("last_single_entry_id") or str(uuid.uuid4())
        st.session_state["last_single_entry_id"] = entry_id
        entry = {
            "id": entry_id,
            "brand_name": app_data.get("brand_name", ""),
            "class_type": app_data.get("class_type", ""),
            "overall_status": result.get("overall_status", "—"),
            "app_data": app_data,
            "image_bytes": upload.getvalue(),
            "result": {k: v for k, v in result.items() if k != "image"},
        }
        _render_single_result(result, upload.getvalue(), approve_reject={"entry": entry, "selected_id": None}, app_data=app_data)
        return
    if submitted and upload is None and adding_new:
        st.warning("Please upload a label image.")
        return

    # Create new: main area — title, large upload, preview, replace option, Check label
    if adding_new and "last_single_result" not in st.session_state:
        _render_brand_title("single")

        with st.form("main_form", clear_on_submit=False):
            # Large upload area (wider center column)
            _, center_col, _ = st.columns([0.5, 3, 0.5])
            with center_col:
                upload = st.file_uploader(
                    "Upload label image",
                    type=["png", "jpg", "jpeg"],
                    key="single_upload",
                    help="PNG, JPG. Photos of labels, scans, or digital artwork.",
                )
                if upload is not None:
                    st.image(upload, width=500, caption="Preview")

            st.markdown(
                """
                <style>
                div[data-testid="stFormSubmitButton"] button {
                    background-color: #28a745 !important;
                    border: 1px solid #28a745 !important;
                    outline: none !important;
                    box-shadow: none !important;
                    font-size: 1.15rem !important;
                    padding: 0.6rem 1.8rem !important;
                }
                div[data-testid="stFormSubmitButton"] button:hover,
                div[data-testid="stFormSubmitButton"] button:focus {
                    background-color: #218838 !important;
                    border-color: #218838 !important;
                    outline: none !important;
                    box-shadow: none !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            _, btn_col, _ = st.columns([0.5, 3, 0.5])
            with btn_col:
                submitted = st.form_submit_button("Check label", type="primary", width="stretch")

        if submitted and upload is not None:
            # Read application details from sidebar (session state)
            ss = st.session_state
            beverage_type = ss.get("create_beverage_type", _BEVERAGE_TYPES[0])
            app_data = {
                "beverage_type": _BEV_TYPE_KEY_MAP.get(beverage_type, "spirits"),
                "brand_name": ss.get("create_brand_name", "") or "",
                "class_type": ss.get("create_class_type", "") or "",
                "alcohol_pct": ss.get("create_alcohol_pct", "") or "",
                "proof": ss.get("create_proof", "") or "",
                "net_contents_ml": ss.get("create_net_contents_ml", "") or "",
                "bottler_name": ss.get("create_bottler_name", "") or "",
                "bottler_city": ss.get("create_bottler_city", "") or "",
                "bottler_state": ss.get("create_bottler_state", "") or "",
                "imported": ss.get("create_imported", False),
                "country_of_origin": ss.get("create_country_of_origin", "") or "",
                "sulfites_required": ss.get("create_sulfites", False),
                "fd_c_yellow_5_required": ss.get("create_fd_c_yellow_5", False),
                "carmine_required": ss.get("create_carmine", False),
                "wood_treatment_required": ss.get("create_wood_treatment", False),
                "age_statement_required": ss.get("create_age_statement", False),
                "neutral_spirits_required": ss.get("create_neutral_spirits", False),
                "aspartame_required": ss.get("create_aspartame", False),
                "appellation_required": ss.get("create_appellation_required", False),
                "varietal_required": ss.get("create_varietal_required", False),
            }
            with st.spinner("Analyzing label..."):
                try:
                    from src.pipeline import run_pipeline
                    result = run_pipeline(upload.getvalue(), app_data)
                    if result.get("error"):
                        st.error("**OCR unavailable**")
                        st.markdown(result["error"])
                    else:
                        st.session_state["last_single_result"] = result
                        st.session_state["last_single_image_bytes"] = upload.getvalue()
                        st.session_state["last_single_app_data"] = app_data
                        st.session_state["last_single_entry_id"] = st.session_state.get("last_single_entry_id") or str(uuid.uuid4())
                        st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        return

    # Under Review list or detail view
    if view == "under_review":
        from src.storage import load_applications
        data = load_applications()
        apps = data["applications_under_review"]
        selected_id = st.session_state.get("selected_app_id")
        selected_bucket = st.session_state.get("selected_app_bucket")
        bucket = "applications_under_review"

        if selected_id and selected_bucket == bucket:
            entry = next((a for a in apps if a["id"] == selected_id), None)
            if entry:
                st.subheader(f"{entry.get('brand_name', '—')} — {entry.get('class_type', '')}")
                result_for_display = dict(entry["result"])
                result_for_display["image"] = None
                _render_single_result(result_for_display, entry.get("image_bytes"), approve_reject={"entry": entry, "selected_id": selected_id}, app_data=entry.get("app_data", {}))
                if st.button("Back to list", key="back_to_list"):
                    st.session_state["selected_app_id"] = None
                    st.session_state["selected_app_bucket"] = None
                    st.rerun()
            else:
                st.session_state["selected_app_id"] = None
                st.session_state["selected_app_bucket"] = None
                st.rerun()
        else:
            if st.button("New label", type="primary", key="btn_add_new"):
                st.session_state["app_list_view"] = "create_new"
                st.session_state["selected_app_id"] = None
                st.session_state["selected_app_bucket"] = None
                st.rerun()
            st.divider()
            if not apps:
                st.info("No applications under review yet.")
            else:
                for a in apps:
                    with st.container():
                        img_col, text_col, btn_col = st.columns([1, 3, 1])
                        with img_col:
                            img_bytes = a.get("image_bytes")
                            if img_bytes:
                                st.image(img_bytes, width=120, caption="")
                            else:
                                st.caption("(no image)")
                        with text_col:
                            st.markdown(f"**{a.get('brand_name', '—')}** — {a.get('class_type', '')}  \n_{a.get('overall_status', '—')}_")
                        with btn_col:
                            if st.button("View", key=f"view_{a['id']}"):
                                st.session_state["selected_app_id"] = a["id"]
                                st.session_state["selected_app_bucket"] = bucket
                                st.rerun()
                        st.divider()
        return

    # Create new: show result after Check label
    if adding_new and "last_single_result" in st.session_state:
        result = st.session_state["last_single_result"]
        app_data = st.session_state.get("last_single_app_data") or {}
        image_bytes = st.session_state.get("last_single_image_bytes") or b""
        entry_id = st.session_state.get("last_single_entry_id") or str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "brand_name": app_data.get("brand_name", ""),
            "class_type": app_data.get("class_type", ""),
            "overall_status": result.get("overall_status", "—"),
            "app_data": app_data,
            "image_bytes": image_bytes,
            "result": {k: v for k, v in result.items() if k != "image"},
        }
        _render_single_result(result, image_bytes, approve_reject={"entry": entry, "selected_id": None}, app_data=app_data)
        return

    _render_brand_title("single")
    st.caption("Upload a label and click **Check label**.")


def _build_validation_matrix(rule_results: list, app_data: dict) -> list[dict]:
    """Build ordered matrix rows: Criteria, Application, Label, Status. Uses app_data for imported/beverage_type."""
    by_rule: dict[str, dict] = {r.get("rule_id", ""): r for r in rule_results if r.get("rule_id")}
    bev = (app_data.get("beverage_type") or "spirits").lower().replace("/", "_").replace(" ", "_")
    is_spirits = bev in ("spirits", "distilled_spirits")
    imported = bool(app_data.get("imported"))

    def pick(*rule_ids: str) -> dict | None:
        for rid in rule_ids:
            if rid in by_rule:
                return by_rule[rid]
        return None

    def row(criteria: str, *rule_ids: str) -> dict | None:
        r = pick(*rule_ids)
        if r is None:
            return None
        app_val = str(r.get("app_value") or "")
        ext_val = str(r.get("extracted_value") or "")
        if len(app_val) > 80:
            app_val = app_val[:77] + "..."
        if len(ext_val) > 80:
            ext_val = ext_val[:77] + "..."
        status = r.get("status", "pass")
        status_display = {"pass": "Pass", "needs_review": "Needs review", "fail": "Fail"}.get(status, "Needs review")
        return {"Criteria": criteria, "Application": app_val or "—", "Label": ext_val or "—", "Status": status_display}

    rows: list[dict] = []
    for criteria, rids in [
        ("Brand Name", ["Brand name matches", "Brand name present"]),
        ("Type", ["Class/type matches", "Class/type present"]),
        ("Government Warning (All Caps)", ["GOVERNMENT WARNING in caps"]),
        ("Government Warning Wording", ["Exact warning wording", "Government warning present"]),
        ("Bottler/Producer", ["Bottler matches", "Bottler/producer statement"]),
        ("Bottler/Address", ["Bottler address"]),
    ]:
        r = row(criteria, *rids)
        if r:
            rows.append(r)

    if imported:
        r = row("Country of Origin", "Country of origin matches", "Country of origin")
        if r:
            rows.append(r)

    for criteria, rids in [
        ("Alcohol content (ABV)", ["Alcohol content matches", "Alcohol content", "Alcohol content present"]),
        ("Proof", ["Proof/ABV consistency", "Proof matches", "Proof present", "Proof"]),
        ("Net contents", ["Net contents matches", "Net contents standard of fill", "Net contents", "Net contents present"]),
    ]:
        if criteria == "Proof" and not is_spirits:
            continue
        r = row(criteria, *rids)
        if r:
            rows.append(r)

    for rule_id, criteria in [
        ("Sulfites statement", "Sulfites"),
        ("FD&C Yellow No. 5", "FD&C Yellow No. 5"),
        ("Cochineal/Carmine statement", "Cochineal/Carmine"),
        ("Wood treatment", "Wood treatment"),
        ("Age statement", "Age statement"),
        ("Neutral spirits / commodity", "Neutral spirits"),
        ("Aspartame statement", "Aspartame"),
        ("Appellation of origin", "Appellation of origin"),
        ("Varietal designation", "Varietal designation"),
    ]:
        r = row(criteria, rule_id)
        if r:
            rows.append(r)

    return rows


def _render_validation_matrix(rows: list[dict]) -> None:
    """Render Criteria × Application × Label × Status table."""
    if not rows:
        st.info("No validation results to display.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, column_order=["Criteria", "Application", "Label", "Status"], hide_index=True, use_container_width=True)


def _render_single_result(result: dict, image_bytes: bytes | None, approve_reject: dict | None = None, app_data: dict | None = None):
    """Render label check result: status banner, caption, image, validation matrix, checklist. approve_reject: {"entry", "selected_id"} to show Approve/Decline."""
    overall = result.get("overall_status", "—")
    counts = result.get("counts", {})

    css_class = {
        "Ready to approve": "status-pass",
        "Needs review": "status-review",
        "Critical issues": "status-fail",
    }.get(overall, "status-review")

    st.markdown(f'<div class="status-banner {css_class}">{overall}</div>', unsafe_allow_html=True)
    if approve_reject:
        cap_col, btn1_col, btn2_col = st.columns([2, 1, 1])
        with cap_col:
            st.caption(
                f"Pass: {counts.get('pass', 0)}  |  "
                f"Needs review: {counts.get('needs_review', 0)}  |  "
                f"Fail: {counts.get('fail', 0)}"
            )
        entry = approve_reject["entry"]
        selected_id = approve_reject.get("selected_id")
        with btn1_col:
            with st.container(key="approve_btn"):
                if st.button("Approve", type="primary", key="btn_approve", width="stretch"):
                    from src.storage import load_applications, save_applications
                    data = load_applications()
                    if selected_id:
                        data["applications_under_review"] = [a for a in data["applications_under_review"] if a.get("id") != selected_id]
                    data["applications_approved"] = data["applications_approved"] + [entry]
                    save_applications(data["applications_under_review"], data["applications_approved"], data["applications_rejected"])
                    for k in ("last_single_result", "last_single_image_bytes", "last_single_app_data", "last_single_entry_id"):
                        st.session_state.pop(k, None)
                    d = load_applications()
                    st.session_state["applications_under_review"] = d["applications_under_review"]
                    st.session_state["applications_approved"] = d["applications_approved"]
                    st.session_state["applications_rejected"] = d["applications_rejected"]
                    st.session_state["app_list_view"] = "create_new"
                    st.rerun()
        with btn2_col:
            with st.container(key="decline_btn"):
                if st.button("Decline", key="btn_decline", width="stretch"):
                    from src.storage import load_applications, save_applications
                    data = load_applications()
                    if selected_id:
                        data["applications_under_review"] = [a for a in data["applications_under_review"] if a.get("id") != selected_id]
                    data["applications_rejected"] = data["applications_rejected"] + [entry]
                    save_applications(data["applications_under_review"], data["applications_approved"], data["applications_rejected"])
                    for k in ("last_single_result", "last_single_image_bytes", "last_single_app_data", "last_single_entry_id"):
                        st.session_state.pop(k, None)
                    d = load_applications()
                    st.session_state["applications_under_review"] = d["applications_under_review"]
                    st.session_state["applications_approved"] = d["applications_approved"]
                    st.session_state["applications_rejected"] = d["applications_rejected"]
                    st.session_state["app_list_view"] = "create_new"
                    st.rerun()
    else:
        st.caption(
            f"Pass: {counts.get('pass', 0)}  |  "
            f"Needs review: {counts.get('needs_review', 0)}  |  "
            f"Fail: {counts.get('fail', 0)}"
        )

    # Validation matrix: Criteria, Application, Label, Status
    resolved_app_data = app_data if app_data is not None else (approve_reject["entry"]["app_data"] if approve_reject else {})
    matrix_rows = _build_validation_matrix(result.get("rule_results", []), resolved_app_data)
    st.subheader("Validation")
    _render_validation_matrix(matrix_rows)
    st.divider()

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

    _render_brand_title("batch")

    with st.sidebar:
        if st.button("Reset", key="batch_reset", width="stretch"):
            for key in ("batch_results", "batch_selected_id", "batch_decisions"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        zip_upload = st.file_uploader("Upload ZIP of label images", type=["zip"], key="batch_zip")
        csv_upload = st.file_uploader("Upload CSV (application data)", type=["csv"], key="batch_csv")
        with st.container(key="batch_run_btn"):
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
                                "app_data": app_data,
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
                                "app_data": app_data,
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
                                "app_data": app_data,
                            })
                    st.session_state["batch_results"] = results
                    if "batch_selected_id" in st.session_state:
                        del st.session_state["batch_selected_id"]
                    st.success(f"Processed {len(results)} labels.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Batch failed: {e}")

    if batch_results:
        batch_decisions = st.session_state.get("batch_decisions", {})

        df_display = pd.DataFrame([
            {
                "label_id": r["label_id"],
                "brand_name": r["brand_name"],
                "class_type": r["class_type"],
                "overall_status": r["overall_status"],
                "failed_rules": r["fail_count"],
                "decision": batch_decisions.get(r["label_id"], "—"),
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

        st.markdown("**Decisions**")
        for r in batch_results:
            lid = r["label_id"]
            dec = batch_decisions.get(lid)
            c1, c2, c3, c4, c5, c6 = st.columns([1.2, 2, 2, 1.2, 0.8, 2])
            with c1:
                st.text(lid)
            with c2:
                st.text(r["brand_name"])
            with c3:
                st.text(r["class_type"])
            with c4:
                st.text(r["overall_status"])
            with c5:
                st.text(str(r["fail_count"]))
            with c6:
                if dec == "approved":
                    st.markdown("**Approved**")
                    if st.button("Undo", key=f"batch_undo_{lid}"):
                        d = dict(st.session_state.get("batch_decisions", {}))
                        d.pop(lid, None)
                        st.session_state["batch_decisions"] = d
                        st.rerun()
                elif dec == "declined":
                    st.markdown("**Declined**")
                    if st.button("Undo", key=f"batch_undo_{lid}"):
                        d = dict(st.session_state.get("batch_decisions", {}))
                        d.pop(lid, None)
                        st.session_state["batch_decisions"] = d
                        st.rerun()
                else:
                    if st.button("Approve", key=f"batch_approve_{lid}", type="primary"):
                        d = dict(st.session_state.get("batch_decisions", {}))
                        d[lid] = "approved"
                        st.session_state["batch_decisions"] = d
                        st.rerun()
                    if st.button("Decline", key=f"batch_decline_{lid}"):
                        d = dict(st.session_state.get("batch_decisions", {}))
                        d[lid] = "declined"
                        st.session_state["batch_decisions"] = d
                        st.rerun()
            st.divider()

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
            _render_single_result(match["result"], None, app_data=match.get("app_data") or {})
        elif match and match.get("error"):
            st.error(match["error"])
        if st.button("Back to batch table", key="batch_back"):
            if "batch_selected_id" in st.session_state:
                del st.session_state["batch_selected_id"]
            st.rerun()

    if not batch_results:
        st.markdown("Upload a **ZIP** of label images and a **CSV** file with one row per label.")
        st.markdown("**Required CSV columns:**")
        st.markdown(
            "`label_id`, `brand_name`, `class_type`, `alcohol_pct`, `proof`, `net_contents_ml`, "
            "`bottler_name`, `bottler_city`, `bottler_state`, `imported`, `country_of_origin`, `beverage_type`"
        )
        st.markdown("You can add optional columns: `sulfites_required`, `age_statement_required`, `wood_treatment_required`, `neutral_spirits_required`, etc.")
        st.markdown("**Example (first 2 rows):**")
        st.code(
            "label_id,brand_name,class_type,alcohol_pct,proof,net_contents_ml,bottler_name,bottler_city,bottler_state,imported,country_of_origin,beverage_type\n"
            "test_1,ABC Distillery,Single Barrel Straight Rye Whisky,45,90,750 mL,ABC Distillery,Frederick,MD,false,,Distilled Spirits\n"
            "test_2,Malt & Hop Brewery,Pale Ale,5,,24 fl oz,Malt & Hop Brewery,Hyattsville,MD,false,,Beer / Malt Beverage",
            language=None,
        )
        st.caption("See sample_data/batch_example.csv for a full example.")


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
