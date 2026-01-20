import requests
import streamlit as st

API_BASE = "https://customer-lifetime-value-tjoy.onrender.com"

st.set_page_config(page_title="Churn + CLV", page_icon="💸", layout="centered")

# ---- light CSS polish ----
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 900px;}
  div[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,.2); border-radius: 14px; padding: 12px;}
  .badge {display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.85rem;}
  .badge-low {background:#e7f7ee; color:#1b7f3a; border:1px solid #b7ebc8;}
  .badge-mid {background:#fff7e6; color:#8a5a00; border:1px solid #ffe1a6;}
  .badge-high{background:#ffe9e9; color:#a61b1b; border:1px solid #ffc1c1;}
</style>
""", unsafe_allow_html=True)


def clv_segment(final_clv: float) -> str:
    # same bins as your pipeline
    if final_clv <= 100:
        return "Low Value"
    if final_clv <= 1000:
        return "Mid Value"
    return "High Value"


def badge_html(segment: str) -> str:
    if segment == "Low Value":
        return '<span class="badge badge-low">Low Value</span>'
    if segment == "Mid Value":
        return '<span class="badge badge-mid">Mid Value</span>'
    return '<span class="badge badge-high">High Value</span>'


st.title("💸 Churn + CLV Calculator")
st.caption("Fill the customer inputs and get churn probability + risk-adjusted CLV from your deployed model.")

# ---- sidebar presets ----
with st.sidebar:
    st.header("⚡ Presets")
    preset = st.radio(
        "Pick a sample customer",
        ["Custom", "High activity", "Low activity", "New customer"],
        index=0
    )

    if preset == "High activity":
        default = dict(avg_order_value=600.0, invoice_count=12, total_quantity=140, tenure_days=365, raw_clv=5000.0)
    elif preset == "Low activity":
        default = dict(avg_order_value=120.0, invoice_count=2, total_quantity=8, tenure_days=60, raw_clv=200.0)
    elif preset == "New customer":
        default = dict(avg_order_value=250.0, invoice_count=1, total_quantity=3, tenure_days=10, raw_clv=250.0)
    else:
        default = dict(avg_order_value=350.0, invoice_count=6, total_quantity=40, tenure_days=180, raw_clv=500.0)

    st.divider()
    st.caption("API")
    st.code(API_BASE, language="text")


with st.form("clv_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        avg_order_value = st.number_input("Avg Order Value", min_value=0.0, value=float(default["avg_order_value"]), step=1.0)
        raw_clv = st.number_input("Raw CLV (₹ / $)", min_value=0.0, value=float(default["raw_clv"]), step=10.0)

    with c2:
        invoice_count = st.number_input("Invoice Count", min_value=0, value=int(default["invoice_count"]), step=1)
        tenure_days = st.number_input("Tenure Days", min_value=0, value=int(default["tenure_days"]), step=1)

    with c3:
        total_quantity = st.number_input("Total Quantity", min_value=0, value=int(default["total_quantity"]), step=1)
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Predict")

if submitted:
    payload = {
        "avg_order_value": float(avg_order_value),
        "invoice_count": int(invoice_count),
        "total_quantity": int(total_quantity),
        "tenure_days": int(tenure_days),
    }

    try:
        with st.spinner("Calling the model..."):
            r1 = requests.post(f"{API_BASE}/predict_churn", json=payload, timeout=30)
            r1.raise_for_status()
            churn = r1.json()

            r2 = requests.post(
                f"{API_BASE}/predict_clv",
                params={"raw_clv": float(raw_clv)},
                json=payload,
                timeout=30
            )
            r2.raise_for_status()
            clv = r2.json()

        prob = float(churn["churn_probability"])
        pred = int(churn["churn_prediction"])
        threshold = float(churn["threshold"])
        final_clv = float(clv["final_clv"])

        st.success("Prediction ready ✅")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn Probability", f"{prob:.3f}")
        m2.metric("Threshold", f"{threshold:.2f}")
        m3.metric("Prediction", "Churn" if pred == 1 else "Not Churn")
        m4.metric("Final CLV", f"{final_clv:,.2f}")

        st.write("### 📊 Probability Bar")
        st.progress(min(max(prob, 0.0), 1.0))

        seg = clv_segment(final_clv)
        st.write("### 🧩 CLV Segment")
        st.markdown(badge_html(seg), unsafe_allow_html=True)

        with st.expander("View raw response"):
            st.json({"churn": churn, "clv": clv})

    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
        st.info("If you are on Render free tier, the first request after idle may take longer (cold start).")
