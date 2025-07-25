import streamlit as st

st.set_page_config(
    page_title="Contrast Test",
    page_icon="🔍",
    layout="wide"
)

# Load CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("MAXIMUM CONTRAST TEST")
st.markdown("### This text should be WHITE on a DARK background")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("TEST METRIC 1", "$100,000", "+15%")
    
with col2:
    st.metric("TEST METRIC 2", "75.5%", "-2.3%")
    
with col3:
    st.metric("TEST METRIC 3", "42", "0")
    
with col4:
    st.metric("TEST METRIC 4", "ACTIVE", "")

st.markdown("""
<div style="
    background: #1A1A1A;
    border: 2px solid #444444;
    padding: 24px;
    border-radius: 16px;
    margin: 16px 0;
">
    <h2 style="color: #FFFFFF; font-size: 24px; margin: 0;">CUSTOM CARD TEST</h2>
    <p style="color: #FFFFFF; font-size: 16px; margin: 8px 0;">This text should be clearly readable</p>
    <p style="color: #D0D0D0; font-size: 14px; margin: 0;">Secondary text in lighter gray</p>
</div>
""", unsafe_allow_html=True)

st.write("Regular streamlit text - should be white")
st.info("Info box - text should be white")
st.success("Success box - text should be white")
st.warning("Warning box - text should be white")
st.error("Error box - text should be white")

# Test input fields
st.text_input("Test Input", placeholder="Type here...")
st.selectbox("Test Select", ["Option 1", "Option 2", "Option 3"])
st.button("TEST BUTTON", type="primary")
