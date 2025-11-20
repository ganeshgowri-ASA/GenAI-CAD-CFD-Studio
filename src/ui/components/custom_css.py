"""
Custom CSS Styling for GenAI CAD CFD Studio
Professional theme with responsive design and animations
"""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""

    css = """
    <style>
    /* Global Theme Variables */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00cc66;
        --background-color: #f5f7fa;
        --card-bg: #ffffff;
        --text-primary: #1a1a1a;
        --text-secondary: #6c757d;
        --border-radius: 8px;
        --box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        --transition: all 0.3s ease;
    }

    /* Main App Background */
    .stApp {
        background-color: var(--background-color);
    }

    /* Header Styling */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }

    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--secondary-color);
        padding-bottom: 0.5rem;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--card-bg);
        padding: 10px;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 24px;
        background-color: transparent;
        border-radius: var(--border-radius);
        color: var(--text-secondary);
        font-weight: 500;
        transition: var(--transition);
        border: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white !important;
        border: 2px solid var(--primary-color);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 102, 204, 0.1);
        transform: translateY(-1px);
    }

    /* Tab Panel Content */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 24px;
        background-color: var(--card-bg);
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        margin-top: 16px;
        animation: fadeIn 0.4s ease-in;
    }

    /* Card Styling */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 20px;
        box-shadow: var(--box-shadow);
        transition: var(--transition);
    }

    .css-1r6slb0:hover, .css-12oz5g7:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: var(--transition);
        box-shadow: var(--box-shadow);
    }

    .stButton button:hover {
        box-shadow: 0 6px 20px rgba(0, 102, 204, 0.4);
        transform: translateY(-2px);
    }

    .stButton button:active {
        transform: translateY(0);
    }

    /* Info Box Styling */
    .stAlert {
        border-radius: var(--border-radius);
        border-left: 4px solid var(--primary-color);
        box-shadow: var(--box-shadow);
        animation: slideIn 0.4s ease-out;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        padding: 20px 10px;
    }

    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }

    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: var(--border-radius);
        border: 2px solid #e0e0e0;
        transition: var(--transition);
    }

    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: 20px;
        background-color: rgba(0, 102, 204, 0.05);
        transition: var(--transition);
    }

    .stFileUploader:hover {
        background-color: rgba(0, 102, 204, 0.1);
        border-color: var(--secondary-color);
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: var(--primary-color);
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 16px;
            font-size: 14px;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding: 16px;
        }

        .stButton button {
            padding: 10px 24px;
            font-size: 14px;
        }
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
        border-right-color: var(--secondary-color) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--card-bg);
        border-radius: var(--border-radius);
        border: 2px solid #e0e0e0;
        font-weight: 600;
        transition: var(--transition);
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--primary-color);
        background-color: rgba(0, 102, 204, 0.05);
    }

    /* Code Block */
    code {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 2px 6px;
        color: var(--primary-color);
        font-family: 'Courier New', monospace;
    }

    /* Success/Warning/Error Messages */
    .stSuccess {
        background-color: rgba(0, 204, 102, 0.1);
        border-left-color: var(--secondary-color) !important;
    }

    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left-color: #ffc107 !important;
    }

    .stError {
        background-color: rgba(220, 53, 69, 0.1);
        border-left-color: #dc3545 !important;
    }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
