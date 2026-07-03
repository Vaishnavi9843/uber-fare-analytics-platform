import streamlit as st


def load_css():
    """
    Apply global styling to the Streamlit application.
    """

    st.markdown(
        """
        <style>

        /* Main content */
        .main{
            padding-top:1rem;
            padding-bottom:2rem;
        }

        /* Metric cards */
        div[data-testid="metric-container"]{
            padding:15px;
            border-radius:12px;
            box-shadow:0px 2px 6px rgba(0,0,0,0.08);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }

        div[data-testid="metric-container"]:hover{
            transform: translateY(-1px);
            box-shadow:0px 6px 18px rgba(0,0,0,0.12);
        }


        /* Buttons */
        .stButton>button{
            width:100%;
            border-radius:10px;
            height:3rem;
            font-size:16px;
            font-weight:600;
        }

        /* DataFrames */
        .stDataFrame{
            border-radius:12px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"]{
            background-color:#0E1117;
        }

        /* Expander */
        .streamlit-expanderHeader{
            font-weight:600;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )