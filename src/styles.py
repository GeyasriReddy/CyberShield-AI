"""
Custom CSS for CyberShield AI Trust and Safety Moderation Dashboard.
Refined clean, modern light SaaS design system inspired by Stripe and GitHub light UI.
"""

APP_CSS = """
<style>
/* Base Theme & Palette Overrides - Light Mode */
:root {
    --bg-primary: #f8fafc;
    --bg-surface: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f5f9;
    --border-subtle: #e2e8f0;
    --border-medium: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --accent-blue: #2563eb;
    --accent-blue-dim: rgba(37, 99, 235, 0.08);
    --status-allow: #059669;
    --status-allow-dim: rgba(5, 150, 105, 0.1);
    --status-review: #d97706;
    --status-review-dim: rgba(217, 119, 6, 0.1);
    --status-flag: #dc2626;
    --status-flag-dim: rgba(220, 38, 38, 0.1);
}

/* Hide Streamlit default chrome (deploy button, footer, hamburger menu) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stToolbar"] {visibility: hidden;}

/* Global Container */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* Sidebar Overrides - Light */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid var(--border-subtle);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Top App Breadcrumb Header */
.top-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.85rem;
    margin-bottom: 1.5rem;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}

.brand-badge .shield-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #2563eb, #0284c7);
    border-radius: 6px;
    color: white;
    font-size: 0.85rem;
    font-weight: 800;
}

.env-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background-color: #f1f5f9;
    border: 1px solid var(--border-subtle);
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.pulse-dot {
    width: 6px;
    height: 6px;
    background-color: #059669;
    border-radius: 50%;
    display: inline-block;
}

.pulse-dot.amber {
    background-color: #d97706;
}

/* Headings */
h1 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.25rem !important;
}

h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary) !important;
}

/* Metric Cards */
.metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 1rem 1.15rem;
    position: relative;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: var(--border-medium);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
    transform: translateY(-1px);
}

.metric-card .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.metric-card .metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    margin-top: 0.35rem;
    margin-bottom: 0.2rem;
}

.metric-card .metric-subtext {
    font-size: 0.74rem;
    color: var(--text-muted);
}

/* System Banner */
.system-notice {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #2563eb;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    margin-bottom: 1.25rem;
    font-size: 0.82rem;
    color: #1e3a8a;
    line-height: 1.5;
}

.system-notice strong {
    color: #1e293b;
}

/* Badges with Indicator Dots */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.6rem;
    font-size: 0.74rem;
    font-weight: 600;
    border-radius: 9999px;
    letter-spacing: 0.03em;
    font-family: ui-monospace, SFMono-Regular, monospace;
}

.badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}

.badge-allow {
    background-color: var(--status-allow-dim);
    color: #047857;
    border: 1px solid rgba(5, 150, 105, 0.3);
}
.badge-allow .badge-dot { background-color: #059669; }

.badge-review {
    background-color: var(--status-review-dim);
    color: #b45309;
    border: 1px solid rgba(217, 119, 6, 0.3);
}
.badge-review .badge-dot { background-color: #d97706; }

.badge-flag {
    background-color: var(--status-flag-dim);
    color: #b91c1c;
    border: 1px solid rgba(220, 38, 38, 0.3);
}
.badge-flag .badge-dot { background-color: #dc2626; }

.badge-neutral {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
}
.badge-neutral .badge-dot { background-color: #64748b; }

/* User Chips */
.user-chip {
    display: inline-block;
    padding: 0.15rem 0.45rem;
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.75rem;
    color: #1d4ed8;
    font-weight: 600;
}

/* Card Surface / Panels */
.panel-box {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

/* Streamlit Tabs Customization */
div[data-testid="stTabs"] button[role="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-secondary) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid #2563eb !important;
    font-weight: 600 !important;
}

/* Inputs & Forms */
div[data-testid="stTextInput"] input, 
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] > div {
    background-color: #ffffff !important;
    border: 1px solid var(--border-medium) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
}

div[data-testid="stTextInput"] input:focus, 
div[data-testid="stTextArea"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}

/* Button overrides */
.stButton > button {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid var(--border-medium);
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.84rem;
    padding: 0.45rem 1rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: all 0.15s ease-in-out;
}

.stButton > button:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
    color: #0f172a;
}

.stButton > button[kind="primary"] {
    background-color: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1d4ed8;
    border-color: #1e40af;
    color: #ffffff;
}

/* Modern Review Item Card */
.review-item {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.review-item:hover {
    border-color: var(--border-medium);
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.06);
}

.review-item.state-pending {
    border-left: 3px solid #d97706;
}

.review-item.state-reviewed {
    border-left: 3px solid #059669;
}

.review-item.state-flagged {
    border-left: 3px solid #dc2626;
}

.comment-quote-box {
    background-color: #f8fafc;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--text-primary);
    margin: 0.65rem 0;
}

/* Monospace & Sub-labels */
.meta-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.meta-value {
    font-size: 0.92rem;
    color: var(--text-primary);
    font-weight: 600;
    margin-top: 0.15rem;
}

/* DataFrame styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
</style>
"""
