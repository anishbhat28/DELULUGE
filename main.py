import streamlit as st

pages = [
    st.Page("app.py", title="Upload", icon="📤", default=True),
    st.Page("pages/dashboard.py", title="Dashboard", icon="📊"),
]

pg = st.navigation(pages)
pg.run()
