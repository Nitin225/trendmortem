import streamlit as st
import requests

st.set_page_config(page_title="TrendMortem", page_icon="")

st.title("TrendMortem")
st.write("Analyze Reddit post virality using AI")

url = st.text_input("Enter Reddit Post URL")

if st.button("Analyze"):
    if url:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    json={"url": url}
                )

                result = response.json()

                if result.get("success"):
                    data = result["data"]

                    st.success("Analysis Complete!")

                    st.write("### Prediction")
                    st.write(f"Viral: {'Yes ' if data['viral'] == 1 else 'No '}")
                    st.metric("Viral Probability", f"{data['probability']*100:.1f}%")

                    st.write("### Explanation")
                    st.write(data["explanation"])

                else:
                    st.error(result.get("error"))

            except Exception as e:
                st.error(str(e))
    else:
        st.warning("Please enter a URL")