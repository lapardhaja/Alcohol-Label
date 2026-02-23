"""
TTB Alcohol Label Verification App - Streamlit entry.
Modes: Single label | Batch review.
"""
import streamlit as st

st.set_page_config(
    page_title="TTB Label Verification",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Accessibility: larger base font and contrast (custom CSS)
st.markdown("""
<style>
    .stApp { font-size: 1.05rem; }
    .big-font { font-size: 1.2rem !important; }
    [data-testid="stSidebar"] { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("TTB Label Verification")
    st.caption("Verify distilled spirits labels against application data.")

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
    st.subheader("Single label")
    upload = st.file_uploader("Upload label image", type=["png", "jpg", "jpeg"], key="single_upload")
    with st.form("single_form"):
        brand = st.text_input("Brand name")
        class_type = st.text_input("Class/type")
        col1, col2 = st.columns(2)
        with col1:
            alcohol_pct = st.text_input("Alcohol % (e.g. 45)")
            proof = st.text_input("Proof (e.g. 90)")
        with col2:
            net_contents_ml = st.text_input("Net contents (mL)")
        bottler_name = st.text_input("Bottler/Producer name")
        col3, col4 = st.columns(2)
        with col3:
            bottler_city = st.text_input("Bottler city")
        with col4:
            bottler_state = st.text_input("Bottler state")
        imported = st.checkbox("Imported")
        country_of_origin = st.text_input("Country of origin (if imported)")
        st.markdown("**Special statements required**")
        c1, c2, c3 = st.columns(3)
        with c1:
            sulfites = st.checkbox("Sulfites")
            fd_c_yellow_5 = st.checkbox("FD&C Yellow No. 5")
        with c2:
            carmine = st.checkbox("Carmine")
            wood_treatment = st.checkbox("Wood treatment")
        with c3:
            age_statement = st.checkbox("Age statement")
            neutral_spirits = st.checkbox("Neutral spirits %")
        submitted = st.form_submit_button("Check label")

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
        with st.spinner("Analyzing label…"):
            try:
                from .pipeline import run_pipeline
                result = run_pipeline(upload.getvalue(), app_data)
                st.session_state["last_single_result"] = result
                st.session_state["last_single_image_bytes"] = upload.getvalue()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return
        _render_single_result(st.session_state["last_single_result"], st.session_state["last_single_image_bytes"])
    elif submitted and upload is None:
        st.warning("Please upload a label image.")
    elif "last_single_result" in st.session_state:
        _render_single_result(st.session_state["last_single_result"], st.session_state.get("last_single_image_bytes"))


def _render_single_result(result: dict, image_bytes: bytes | None):
    st.subheader("Results")
    overall = result.get("overall_status", "—")
    counts = result.get("counts", {})
    status_color = {"Ready to approve": "green", "Needs review": "orange", "Critical issues": "red"}
    st.markdown(f"**Overall: <span style='color:{status_color.get(overall, \"gray\")}'>{overall}</span>**", unsafe_allow_html=True)
    st.caption(f"Pass: {counts.get('pass', 0)}  |  Needs review: {counts.get('needs_review', 0)}  |  Fail: {counts.get('fail', 0)}")

    col_img, col_list = st.columns([1, 1])
    with col_img:
        if result.get("image") is not None:
            st.image(result["image"], use_container_width=True, caption="Label image")
        elif image_bytes:
            st.image(image_bytes, use_container_width=True, caption="Label image")
    with col_list:
        for r in result.get("rule_results", []):
            status = r.get("status", "pass")
            icon = "✅" if status == "pass" else "⚠️" if status == "needs_review" else "❌"
            st.markdown(f"{icon} **{r.get('rule_id', r.get('category', '?'))}**: {r.get('message', '')}")
    st.divider()
    st.caption("Show on label: highlight feature can be wired to bbox_ref in each rule result.")


def _batch_screen():
    st.subheader("Batch review")
    zip_upload = st.file_uploader("Upload ZIP of label images", type=["zip"], key="batch_zip")
    csv_upload = st.file_uploader("Upload CSV (label_id and application data)", type=["csv"], key="batch_csv")
    if st.button("Run batch checks"):
        if not zip_upload or not csv_upload:
            st.warning("Please upload both a ZIP of images and a CSV.")
        else:
            st.info("Batch processing: parse ZIP and CSV, then run pipeline per label. Results in table with link to detail.")
    st.caption("After running batch: table with label_id, brand, class/type, overall status, failed count; click row for single-label view.")


if __name__ == "__main__":
    main()
