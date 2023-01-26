import streamlit as st

st.set_page_config(page_title="Model Analysis", layout='wide', page_icon='./webApp/images/camera.png')
st.title('Result & Analysis')
st.markdown("""---""")

st.subheader("Confusion Matrix")
st.image("./webApp/images/model_results/confusion_matrix.png")

st.markdown("""---""")
st.markdown("""---""")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
with col1:
    st.subheader("F1_curve")
    st.image("./webApp/images/model_results/F1_curve.png")
with col2:
    st.subheader("P_curve")
    st.image("./webApp/images/model_results/P_curve.png")
with col3:
    st.subheader("PR_curve")
    st.image("./webApp/images/model_results/PR_curve.png")
with col4:
    st.subheader("R_curve")
    st.image("./webApp/images/model_results/R_curve.png")

st.markdown("""---""")
st.markdown("""---""")

st.subheader("Overall Result")
st.image("./webApp/images/model_results/results.png")