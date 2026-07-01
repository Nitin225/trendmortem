import streamlit as st
import requests

st.set_page_config(page_title="TrendMortem", page_icon="🔥")
st.title("TrendMortem")
st.write("Analyze Reddit post virality using AI")

API_URL = "https://trendmortem-production.up.railway.app/analyze"

url = st.text_input("Enter Reddit Post URL")

if st.button("Analyze"):
    if url:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json={"url": url}, timeout=30)
                result = response.json()

                if result.get("success"):
                    data = result["data"]
                    st.success("Analysis Complete!")
                    st.write("### Prediction")
                    st.write(f"Viral: {'Yes 🔥' if data['viral'] == 1 else 'No'}")
                    st.metric("Viral Probability", f"{data['probability']*100:.1f}%")
                    st.write("### Explanation")
                    st.write(data["explanation"])
                else:
                    st.error(result.get("error", "Something went wrong"))
            except Exception as e:
                st.error(f"Could not reach the backend: {e}")
    else:
        st.warning("Please enter a URL")