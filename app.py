"""
CyberShield AI - Human-Reviewed Cyberbullying Detection & Social Network Analysis
Trust & Safety Moderation Console
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

from src.styles import APP_CSS
from src.classifier import predict_comment, ClassifierResult
from src.network import (
    load_dataset_interaction_graph,
    compute_network_metrics,
    get_user_network_context,
    create_network_figure,
    create_attack_network_figure,
    create_ego_tree_figure,
    create_mst_tree_figure,
    create_centrality_ranking_figure,
    create_community_breakdown_figure
)
from src.recommender import generate_recommendation, ModerationRecommendation
from src.data_store import (
    get_initial_sample_comments,
    calculate_overview_metrics,
    validate_comment_input,
    validate_user_id,
    process_batch_dataframe
)

# Page configuration
st.set_page_config(
    page_title="CyberShield AI - Trust & Safety Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom SaaS styling
st.markdown(APP_CSS, unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "comments" not in st.session_state:
        st.session_state.comments = get_initial_sample_comments()
    
    if "graph" not in st.session_state or "metrics_map" not in st.session_state:
        G, df_cb, summary = load_dataset_interaction_graph()
        metrics_map, df_metrics = compute_network_metrics(G)
        st.session_state.graph = G
        st.session_state.df_cb = df_cb
        st.session_state.graph_summary = summary
        st.session_state.metrics_map = metrics_map
        st.session_state.df_metrics = df_metrics


init_session_state()


# Badge Renderer with Indicator Dots
def render_badge(action: str) -> str:
    action_lower = action.lower()
    if "allow" in action_lower or "clean" in action_lower or "approved" in action_lower:
        return f'<span class="badge badge-allow"><span class="badge-dot"></span>{action}</span>'
    elif "review" in action_lower or "pending" in action_lower:
        return f'<span class="badge badge-review"><span class="badge-dot"></span>{action}</span>'
    elif "flag" in action_lower or "toxic" in action_lower or "harassment" in action_lower:
        return f'<span class="badge badge-flag"><span class="badge-dot"></span>{action}</span>'
    return f'<span class="badge badge-neutral"><span class="badge-dot"></span>{action}</span>'


# Top Application Header Bar
st.markdown(
    """
    <div class="top-header-bar">
        <div class="brand-badge">
            <span class="shield-icon">🛡</span>
            <span>CyberShield AI</span>
            <span style="font-size: 0.8rem; color: #64748b; font-weight: 500; margin-left: 0.4rem;">/ Trust & Safety Console</span>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
            <span class="env-pill"><span class="pulse-dot"></span> Pipeline Active</span>
            <span class="env-pill">Workspace: Community Forum A</span>
            <span class="env-pill"><span class="pulse-dot amber"></span> Demo Mode (Mock Engine)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation & Telemetry
pending_count = sum(1 for c in st.session_state.comments if c.get("status") == "Pending Review")

with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; font-weight: 600;">Workspace Context</div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #f1f5f9; margin-top: 0.2rem;">Production Moderation</div>
            <div style="font-size: 0.76rem; color: #94a3b8;">Review Policy: Human-in-the-Loop</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    nav_options = [
        "Overview",
        "Comment Analysis",
        "Network Analysis",
        f"Review Queue ({pending_count})"
    ]

    nav_selection = st.radio(
        "Navigation Menu",
        options=nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Live Health & Telemetry Box
    st.markdown(
        f"""
        <div class="panel-box" style="padding: 0.85rem; margin-bottom: 1rem;">
            <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.4rem;">Platform Status</div>
            <div style="font-size: 0.8rem; color: #475569; line-height: 1.6;">
                <div style="display:flex; justify-content:space-between;">
                    <span>Classifier:</span>
                    <span style="color: #d97706; font-family: monospace; font-weight: 600;">Mock v0.1</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>Interaction Graph:</span>
                    <span style="color: #2563eb; font-family: monospace; font-weight: 600;">{st.session_state.graph_summary.get('total_nodes', 100)} Nodes</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>Queue Backlog:</span>
                    <span style="color: {'#d97706' if pending_count > 0 else '#059669'}; font-weight: 600;">{pending_count} pending</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Reset Sample Data", help="Reload the original sample audit comments"):
        st.session_state.comments = get_initial_sample_comments()
        st.rerun()

    st.caption("CyberShield AI • v0.2.0-preview")


# Clean page name resolution
current_page = "Review Queue" if "Review Queue" in nav_selection else nav_selection


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if current_page == "Overview":
    st.title("Moderation Operations Overview")
    st.markdown(
        """
        <div class="system-notice">
            <strong>Demonstration Notice:</strong> The system is running against a baseline mock classifier and synthetic 20-node interaction network. 
            All activity records and audit counts below represent controlled sample data for interface verification.
        </div>
        """,
        unsafe_allow_html=True
    )

    metrics = calculate_overview_metrics(st.session_state.comments)

    # 4 High-level KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Analyzed</div>
                <div class="metric-value">{metrics['total_analyzed']}</div>
                <div class="metric-subtext">Cumulative audit volume</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Needs Moderator Review</div>
                <div class="metric-value" style="color: #d97706;">{metrics['recommended_review']}</div>
                <div class="metric-subtext">Borderline / ambiguous content</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recommended for Flagging</div>
                <div class="metric-value" style="color: #dc2626;">{metrics['recommended_flag']}</div>
                <div class="metric-subtext">High confidence harassment/toxicity</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Safe & Permitted</div>
                <div class="metric-value" style="color: #059669;">{metrics['allowed_count']}</div>
                <div class="metric-subtext">Conforms to community standards</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Middle Section: Chart and Operational Principles
    c_left, c_right = st.columns([1.1, 1], gap="medium")

    with c_left:
        st.subheader("Classification Distribution")
        label_counts = {}
        for c in st.session_state.comments:
            lbl = c.get("predicted_label", "Clean")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        chart_df = pd.DataFrame({
            "Classification": list(label_counts.keys()),
            "Count": list(label_counts.values())
        })

        color_map = {
            "Clean": "#059669",
            "Toxic": "#d97706",
            "Harassment": "#dc2626"
        }

        fig = px.pie(
            chart_df,
            names="Classification",
            values="Count",
            hole=0.6,
            color="Classification",
            color_discrete_map=color_map
        )
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#475569", family="Inter, sans-serif"),
            margin=dict(t=15, b=15, l=15, r=15),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
            height=270
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("Trust & Safety Framework")
        st.markdown(
            """
            <div class="panel-box" style="font-size: 0.86rem; line-height: 1.6; color: #334155;">
                <p style="margin-top:0;"><strong>CyberShield AI</strong> provides human-supervised moderation for online communities, combining automated NLP scanning with graph context to prevent abuse without opaque automated bans.</p>
                <div style="border-top: 1px solid #e2e8f0; padding-top: 0.6rem; margin-top: 0.6rem;">
                    <div style="margin-bottom: 0.4rem;">
                        <span style="color: #2563eb; font-weight: 600;">1. Human Primacy:</span> 
                        Algorithms suggest priorities; authorized human moderators make all sanction decisions.
                    </div>
                    <div style="margin-bottom: 0.4rem;">
                        <span style="color: #2563eb; font-weight: 600;">2. Structural Context:</span> 
                        Betweenness and degree centrality gauge audience exposure, never guilt.
                    </div>
                    <div>
                        <span style="color: #2563eb; font-weight: 600;">3. Explainable Audit Trail:</span> 
                        Every recommendation produces clear reasons explaining why an item was queued.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Recent Moderation Audit Stream
    st.subheader("Recent Moderation Stream (Live Audit Log)")
    
    table_search = st.text_input("Filter Stream by User or Keyword", placeholder="Search comments...", label_visibility="collapsed")
    
    stream_data = []
    for c in st.session_state.comments:
        if table_search:
            s_low = table_search.lower()
            if s_low not in c["comment_text"].lower() and s_low not in c["user_id"].lower() and s_low not in c["predicted_label"].lower():
                continue

        stream_data.append({
            "Audit ID": c["id"],
            "Timestamp": c["timestamp"],
            "User": c["user_id"],
            "Comment Snippet": (c["comment_text"][:65] + "...") if len(c["comment_text"]) > 65 else c["comment_text"],
            "Label": c["predicted_label"],
            "Confidence": f"{c['confidence'] * 100:.1f}%",
            "Recommendation": c["recommended_action"],
            "Status": c["status"]
        })

    df_stream = pd.DataFrame(stream_data)
    st.dataframe(df_stream, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 2: COMMENT ANALYSIS
# ==========================================
elif current_page == "Comment Analysis":
    st.title("Comment Analysis Workbench")
    st.markdown(
        """
        <div class="system-notice">
            <strong>Active Classifier:</strong> Rule-based baseline mock. Connect your teammate's trained model in <code>src/classifier.py</code> 
            to automatically replace mock outputs without altering the user interface.
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_single, tab_batch = st.tabs(["Single Comment Inspector", "Batch File Upload (CSV / JSON)"])

    # ---------------- TAB 1: SINGLE COMMENT ----------------
    with tab_single:
        col_in1, col_in2 = st.columns([2.4, 1.2], gap="large")

        with col_in1:
            comment_input = st.text_area(
                "Comment Content",
                placeholder="Paste or type a user comment here...",
                height=130,
                key="single_comment_text"
            )

        with col_in2:
            user_input = st.text_input(
                "Author User ID (Optional)",
                placeholder="e.g. usr_107",
                help="If the user ID exists in the social graph, structural network reach metrics will be retrieved.",
                key="single_user_id"
            )

            st.markdown("<div class='meta-label' style='margin-top:0.5rem;'>Quick Test Presets:</div>", unsafe_allow_html=True)
            preset_btn_cols = st.columns(2)
            with preset_btn_cols[0]:
                if st.button("Toxic Sample", use_container_width=True):
                    st.session_state["preset_text"] = "You are an absolute idiot and a complete loser."
                    st.session_state["preset_user"] = "usr_107"
                    st.rerun()
            with preset_btn_cols[1]:
                if st.button("Harassment", use_container_width=True):
                    st.session_state["preset_text"] = "Keep posting and see what happens. I will destroy you."
                    st.session_state["preset_user"] = "usr_110"
                    st.rerun()

            preset_btn_cols2 = st.columns(2)
            with preset_btn_cols2[0]:
                if st.button("Constructive", use_container_width=True):
                    st.session_state["preset_text"] = "Thanks for sharing the documentation link and helping troubleshoot."
                    st.session_state["preset_user"] = "usr_103"
                    st.rerun()
            with preset_btn_cols2[1]:
                if st.button("Ambiguous", use_container_width=True):
                    st.session_state["preset_text"] = "Just stop talking, nobody asked for your perspective here."
                    st.session_state["preset_user"] = "usr_105"
                    st.rerun()

        # Handle preset auto-fill
        if "preset_text" in st.session_state:
            comment_input = st.session_state.pop("preset_text")
            user_input = st.session_state.pop("preset_user")

        analyze_clicked = st.button("Run Moderation Analysis", type="primary")

        if analyze_clicked:
            valid_txt, txt_err = validate_comment_input(comment_input)
            valid_usr, usr_err = validate_user_id(user_input)

            if not valid_txt:
                st.error(txt_err)
            elif not valid_usr:
                st.error(usr_err)
            else:
                clf_result = predict_comment(comment_input)
                clean_usr = user_input.strip() if user_input else ""
                net_context = get_user_network_context(clean_usr, st.session_state.graph, st.session_state.metrics_map)
                recommendation = generate_recommendation(clf_result, net_context)

                st.markdown("<hr style='border-color: #e2e8f0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
                st.subheader("Moderation Findings & Audit Record")

                res_c1, res_c2, res_c3 = st.columns(3)

                with res_c1:
                    badge_html = render_badge(clf_result.predicted_label)
                    st.markdown(
                        f"""
                        <div class="panel-box">
                            <div class="meta-label">NLP Classifier Output</div>
                            <div style="margin: 0.6rem 0;">
                                <span style="font-size: 1.35rem; font-weight: 700; color: #0f172a; margin-right: 0.5rem;">
                                    {clf_result.predicted_label}
                                </span>
                                {badge_html}
                            </div>
                            <div class="meta-label">Confidence Score</div>
                            <div class="meta-value" style="font-size: 1.1rem; color: #2563eb;">{clf_result.confidence * 100:.1f}%</div>
                            <div style="font-size: 0.74rem; color: #64748b; margin-top: 0.5rem;">
                                Model: {clf_result.model_version} (Mock)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with res_c2:
                    action_badge = render_badge(recommendation.action)
                    st.markdown(
                        f"""
                        <div class="panel-box">
                            <div class="meta-label">Action Recommendation</div>
                            <div style="margin: 0.6rem 0;">
                                <span style="font-size: 1.35rem; font-weight: 700; color: #0f172a; margin-right: 0.5rem;">
                                    {recommendation.action}
                                </span>
                                {action_badge}
                            </div>
                            <div class="meta-label">Priority Assessment</div>
                            <div class="meta-value" style="font-size: 1.1rem; color: #d97706;">{recommendation.urgency}</div>
                            <div style="font-size: 0.74rem; color: #64748b; margin-top: 0.5rem;">
                                Human verification required
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with res_c3:
                    if net_context:
                        net_title = f"<span class='user-chip'>{net_context['user_id']}</span> • {net_context['community']}"
                        net_desc = f"Degree: {net_context['degree']} ties &bull; Betweenness: {net_context['betweenness_centrality']:.2f}"
                    else:
                        net_title = "<span style='color: #64748b;'>No Graph Records</span>"
                        net_desc = "User identifier not indexed in current interaction graph"

                    st.markdown(
                        f"""
                        <div class="panel-box">
                            <div class="meta-label">Social Network Reach</div>
                            <div style="margin: 0.6rem 0; font-size: 0.95rem; font-weight: 600;">
                                {net_title}
                            </div>
                            <div class="meta-label">Structural Influence</div>
                            <div class="meta-value" style="font-size: 0.88rem; color: #334155;">{net_desc}</div>
                            <div style="font-size: 0.74rem; color: #64748b; margin-top: 0.5rem;">
                                Informs exposure potential, not culpability
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Explainability Block
                st.markdown(
                    f"""
                    <div class="panel-box" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #2563eb;">
                        <div style="font-weight: 600; color: #0f172a; font-size: 0.95rem; margin-bottom: 0.35rem;">
                            Decision Explanation & Audit Rationale
                        </div>
                        <p style="font-size: 0.88rem; color: #334155; margin-bottom: 0.5rem; line-height: 1.55;">
                            {recommendation.combined_explanation}
                        </p>
                        <div style="font-size: 0.76rem; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 0.5rem;">
                            <strong>Rule Note:</strong> {clf_result.inference_notes}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Send to Queue Button
                if recommendation.action in ["Review", "Flag"]:
                    if st.button("Dispatch to Moderator Review Queue", key="btn_send_queue", type="primary"):
                        new_id = f"CMT-{len(st.session_state.comments) + 1001}"
                        new_record = {
                            "id": new_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "user_id": clean_usr if clean_usr else "anonymous",
                            "comment_text": comment_input.strip(),
                            "predicted_label": clf_result.predicted_label,
                            "confidence": clf_result.confidence,
                            "recommended_action": recommendation.action,
                            "reason": recommendation.primary_reason,
                            "status": "Pending Review",
                            "moderator_notes": ""
                        }
                        st.session_state.comments.insert(0, new_record)
                        st.success(f"Item queued for moderator review with Audit ID {new_id}.")

    # ---------------- TAB 2: BATCH FILE UPLOAD ----------------
    with tab_batch:
        st.subheader("Batch Dataset Ingestion & Moderation")
        st.caption("Upload a CSV or JSON file containing user comments to run batch moderation and export results.")

        # Download sample template helper
        sample_csv_data = "user_id,comment_text\nusr_107,You are a complete idiot\nusr_103,Great collaboration on the project today\nusr_110,I will find you and make you regret posting that\nusr_105,This solution is totally pathetic and broken\nusr_118,Can anyone help explain this algorithm?\n"
        st.download_button(
            label="Download Sample CSV Template",
            data=sample_csv_data,
            file_name="sample_moderation_batch.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload Comment Batch File (CSV or JSON)", type=["csv", "json"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    batch_df = pd.read_json(uploaded_file)

                st.write(f"**Loaded {len(batch_df)} rows from `{uploaded_file.name}`**")
                
                # Column selection
                col_sel1, col_sel2 = st.columns(2)
                cols = list(batch_df.columns)
                
                with col_sel1:
                    text_candidate = next((c for c in cols if "comment" in c.lower() or "text" in c.lower() or "content" in c.lower()), cols[0])
                    selected_text_col = st.selectbox("Comment Text Column", options=cols, index=cols.index(text_candidate))
                
                with col_sel2:
                    user_candidate = next((c for c in cols if "user" in c.lower() or "author" in c.lower()), None)
                    selected_user_col = st.selectbox("Author / User ID Column (Optional)", options=["(None)"] + cols, index=(cols.index(user_candidate) + 1) if user_candidate else 0)

                user_col_arg = None if selected_user_col == "(None)" else selected_user_col

                if st.button("Run Batch Moderation", type="primary"):
                    with st.spinner("Processing batch through classifier & network context engine..."):
                        processed_df, batch_stats = process_batch_dataframe(
                            df=batch_df,
                            text_col=selected_text_col,
                            user_col=user_col_arg,
                            predict_fn=predict_comment,
                            network_context_fn=get_user_network_context,
                            recommend_fn=generate_recommendation,
                            graph=st.session_state.graph,
                            metrics_map=st.session_state.metrics_map
                        )

                    st.markdown("<hr style='border-color: #1e293b; margin: 1rem 0;'>", unsafe_allow_html=True)
                    
                    # Batch summary pills
                    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                    with b_col1:
                        st.metric("Total Analyzed", batch_stats["Total"])
                    with b_col2:
                        st.metric("Allowed (Safe)", batch_stats.get("Allow", 0))
                    with b_col3:
                        st.metric("Review Needed", batch_stats.get("Review", 0))
                    with b_col4:
                        st.metric("Flagged", batch_stats.get("Flag", 0))

                    st.dataframe(processed_df, use_container_width=True, hide_index=True)

                    # Export & Queue buttons
                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        csv_export = processed_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Export Results to CSV",
                            data=csv_export,
                            file_name=f"moderated_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                    with btn_c2:
                        if st.button("Send Flagged & Review Items to Review Queue"):
                            added_count = 0
                            for _, r in processed_df.iterrows():
                                if r["Recommended Action"] in ["Review", "Flag"]:
                                    new_id = f"CMT-{len(st.session_state.comments) + 1001}"
                                    st.session_state.comments.insert(0, {
                                        "id": new_id,
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        "user_id": r["User ID"],
                                        "comment_text": r["Comment"],
                                        "predicted_label": r["Predicted Label"],
                                        "confidence": r["Confidence"],
                                        "recommended_action": r["Recommended Action"],
                                        "reason": r["Reason"],
                                        "status": "Pending Review",
                                        "moderator_notes": ""
                                    })
                                    added_count += 1
                            st.success(f"Successfully enqueued {added_count} comments into the Review Queue!")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")


# ==========================================
# PAGE 3: NETWORK ANALYSIS
# ==========================================
elif current_page == "Network Analysis":
    st.title("Social Network Interaction Topology & Analytics")
    
    summary = st.session_state.graph_summary
    is_real = summary.get("is_real", False)
    
    if is_real:
        banner_text = f"<strong>Connected Real Dataset:</strong> {summary.get('source')} &bull; Analyzing <strong>{summary.get('total_nodes')} users</strong>, <strong>{summary.get('total_edges'):,} communication dyads</strong>, and <strong>{summary.get('total_messages'):,} total messages</strong>."
    else:
        banner_text = f"<strong>Synthetic Benchmark:</strong> {summary.get('source')} &bull; Using controlled 20-node fallback network."

    st.markdown(
        f"""
        <div class="system-notice">
            {banner_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    G = st.session_state.graph
    metrics_map = st.session_state.metrics_map
    df_metrics = st.session_state.df_metrics
    df_cb = st.session_state.df_cb

    # Graph High-level Stats (4 KPIs)
    st_col1, st_col2, st_col3, st_col4 = st.columns(4)
    with st_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Network Nodes</div>
                <div class="metric-value">{summary.get('total_nodes', len(G.nodes))}</div>
                <div class="metric-subtext">Active student accounts</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with st_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Communication Dyads</div>
                <div class="metric-value">{summary.get('total_edges', len(G.edges)):,}</div>
                <div class="metric-subtext">Active directed ties</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with st_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Network Reciprocity</div>
                <div class="metric-value">{summary.get('reciprocity', 0.96) * 100:.1f}%</div>
                <div class="metric-subtext">Two-way message exchanges</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with st_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Cyberbullying Dyads</div>
                <div class="metric-value" style="color: #dc2626;">{summary.get('cb_dyads', 992):,}</div>
                <div class="metric-subtext">Flagged relationship pairs</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

    # User Selection & Interactive Network Canvas (Graph 1)
    st.subheader("1. Interactive Network Topology Map")
    st.caption("Visual layout of user nodes and communications. Select a node below to highlight their ego-network.")

    user_list = ["(None)"] + sorted(list(G.nodes))
    default_user = "usr_015" if "usr_015" in user_list else (user_list[1] if len(user_list) > 1 else "(None)")
    sel_user = st.selectbox(
        "Select User Node to Highlight Neighborhood:",
        options=user_list,
        index=user_list.index(default_user) if default_user in user_list else 0
    )

    highlight_target = None if sel_user == "(None)" else sel_user

    # Render Plotly Graph 1
    fig_topo = create_network_figure(G, metrics_map, highlight_user=highlight_target)
    st.plotly_chart(fig_topo, use_container_width=True)

    # Detailed Node Inspection Drawer
    if highlight_target and highlight_target in metrics_map:
        node_info = metrics_map[highlight_target]
        neighbors = node_info["direct_neighbors"]
        is_bridge = node_info['betweenness_centrality'] > 0.00035

        neighbor_chips = "".join([f"<span class='user-chip' style='margin-right: 0.35rem; margin-bottom: 0.25rem;'>{n}</span>" for n in neighbors[:25]])
        if len(neighbors) > 25:
            neighbor_chips += f"<span style='color: #64748b; font-size: 0.78rem;'> +{len(neighbors) - 25} more</span>"

        st.markdown(
            f"""
            <div class="panel-box" style="border-left: 3px solid #2563eb;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin: 0; color: #0f172a;">User Profile: <span class="user-chip">{highlight_target}</span></h4>
                    <span class="badge {'badge-review' if is_bridge else 'badge-allow'}">
                        <span class="badge-dot"></span>{'Key Network Bridge' if is_bridge else 'Active Participant'}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 0.85rem 0;">
                    <div>
                        <div class="meta-label">Community</div>
                        <div class="meta-value">{node_info['community']}</div>
                    </div>
                    <div>
                        <div class="meta-label">Degree (Direct Ties)</div>
                        <div class="meta-value">{node_info['degree']}</div>
                    </div>
                    <div>
                        <div class="meta-label">Betweenness Centrality</div>
                        <div class="meta-value" style="color: {'#d97706' if is_bridge else '#0f172a'};">{node_info['betweenness_centrality']:.4f}</div>
                    </div>
                    <div>
                        <div class="meta-label">Aggressive Sent</div>
                        <div class="meta-value" style="color: {'#dc2626' if node_info['aggressive_messages_sent'] > 10 else '#0f172a'};">{node_info['aggressive_messages_sent']}</div>
                    </div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div class="meta-label" style="margin-bottom: 0.3rem;">Direct Neighbors:</div>
                    {neighbor_chips if neighbors else 'None'}
                </div>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 0.65rem; border-top: 1px solid #e2e8f0; padding-top: 0.5rem;">
                    <strong>Exposure Interpretation:</strong> {
                        'High betweenness centrality indicates this user acts as a critical bridge between communities. Content originating from or passing through this node has disproportionate reach across the network.'
                        if is_bridge else
                        'Normal connectivity pattern. Communication is distributed across standard intra-cluster ties.'
                    }
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color: #e2e8f0; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Section 2: Linked Network & Tree Visualizations
    st.subheader("2. Linked Network & Tree Visualizations")
    st.caption("Advanced node-link and hierarchical tree structures visualizing relational dynamics and communication backbones.")

    tree_c1, tree_c2 = st.columns(2, gap="medium")
    with tree_c1:
        st.markdown("**Graph 2: Cyberbullying Attack Network (Node-Link)**")
        st.caption("Directed harassment links showing bullies (red) and targeted victims (amber).")
        fig_atk = create_attack_network_figure(df_cb, max_edges=120)
        st.plotly_chart(fig_atk, use_container_width=True)

    with tree_c2:
        st.markdown(f"**Graph 3: Radial Ego-Tree for {highlight_target if highlight_target else 'usr_015'} (Tree Graph)**")
        st.caption("Hierarchical radial star tree centered on root user branching to 1st-degree contacts.")
        target_for_tree = highlight_target if highlight_target else "usr_015"
        fig_ego = create_ego_tree_figure(target_for_tree, G, df_cb)
        st.plotly_chart(fig_ego, use_container_width=True)

    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)

    tree_c3, tree_c4 = st.columns(2, gap="medium")
    with tree_c3:
        st.markdown("**Graph 4: Minimum Spanning Tree (MST Backbone Tree)**")
        st.caption("NetworkX acyclic tree preserving highest-frequency interaction paths without loops.")
        fig_mst = create_mst_tree_figure(df_cb)
        st.plotly_chart(fig_mst, use_container_width=True)

    with tree_c4:
        st.markdown("**Graph 5: Top 10 Betweenness Centrality Bridges (Bar Chart)**")
        st.caption("Intermediary accounts with highest cross-community bridge influence.")
        fig_bet = create_centrality_ranking_figure(df_metrics)
        st.plotly_chart(fig_bet, use_container_width=True)

    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)

    # Section 3: Community & Multi-Agent Architecture
    row3_c1, row3_c2 = st.columns(2, gap="medium")
    with row3_c1:
        st.markdown("**Graph 6: Community Clusters & Aggressive Volume (Bar Chart)**")
        st.caption("Student group sizes and cumulative aggressive communication counts.")
        fig_comm = create_community_breakdown_figure(df_metrics)
        st.plotly_chart(fig_comm, use_container_width=True)

    with row3_c2:
        st.markdown("**Agentic AI Architecture (Multi-Agent System)**")
        st.markdown(
            f"""
            <div class="panel-box" style="font-size: 0.84rem; line-height: 1.55; color: #334155; height: 280px; overflow-y: auto;">
                <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">4-Agent Collaborative Moderation Pipeline:</div>
                <div style="margin-bottom: 0.4rem;">
                    <span style="color: #2563eb; font-weight: 600;">• Agent 1: NLP Content Classifier</span><br>
                    Scans incoming text for toxic, aggressive, and harassment language patterns.
                </div>
                <div style="margin-bottom: 0.4rem;">
                    <span style="color: #059669; font-weight: 600;">• Agent 2: SNA Graph Analyst (NetworkX)</span><br>
                    Evaluates user centrality, betweenness bridges, and audience broadcast reach.
                </div>
                <div style="margin-bottom: 0.4rem;">
                    <span style="color: #d97706; font-weight: 600;">• Agent 3: Decision & Triage Agent</span><br>
                    Synthesizes text + network signals to propose transparent Allow / Review / Flag actions.
                </div>
                <div>
                    <span style="color: #7c3aed; font-weight: 600;">• Agent 4: Human-in-the-Loop Moderator</span><br>
                    Supervisory human agent confirms or dismisses flags in the Review Queue.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color: #e2e8f0; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Centrality Ledger Table
    st.subheader("3. Centrality Metrics Ledger")
    st.caption("Tabular view of all user nodes sorted by betweenness centrality.")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 4: REVIEW QUEUE
# ==========================================
elif current_page == "Review Queue":
    st.title("Moderator Review Queue")
    st.caption("Human-in-the-loop review workbench. Moderators verify context and confirm or dismiss flags without automated account penalties.")

    filter_col1, filter_col2 = st.columns([1.5, 2.5])
    with filter_col1:
        status_filter = st.selectbox(
            "Filter Queue by Status",
            options=["Pending Review", "All Records", "Reviewed - Approved", "Reviewed - Confirmed Flag", "Dismissed"]
        )
    with filter_col2:
        search_kw = st.text_input("Search Queue (Text or User ID)", placeholder="Filter records...", label_visibility="collapsed")

    # Filter comments
    if status_filter == "All Records":
        filtered = st.session_state.comments
    else:
        filtered = [c for c in st.session_state.comments if c.get("status") == status_filter]

    if search_kw:
        sk = search_kw.lower()
        filtered = [c for c in filtered if sk in c["comment_text"].lower() or sk in c["user_id"].lower()]

    st.markdown(f"**Showing {len(filtered)} item(s) in review view**")

    if not filtered:
        st.info("No moderation records match the current filter criteria.")
    else:
        for idx, item in enumerate(filtered):
            item_id = item["id"]
            current_status = item.get("status", "Pending Review")

            state_class = "state-pending"
            if "Approved" in current_status or "Allow" in current_status:
                state_class = "state-reviewed"
            elif "Flag" in current_status:
                state_class = "state-flagged"

            action_badge = render_badge(item.get("recommended_action", "Review"))
            label_badge = render_badge(item.get("predicted_label", "Clean"))
            status_badge = render_badge(current_status)

            with st.container():
                st.markdown(
                    f"""
                    <div class="review-item {state_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <div>
                                <span style="font-weight: 700; color: #0f172a; font-family: monospace;">{item_id}</span>
                                <span style="color: #64748b; font-size: 0.78rem; margin-left: 0.6rem;">{item.get('timestamp')}</span>
                                <span class="user-chip" style="margin-left: 0.6rem;">{item.get('user_id')}</span>
                            </div>
                            <div>
                                Status: {status_badge}
                            </div>
                        </div>
                        <div class="comment-quote-box">
                            "{item.get('comment_text')}"
                        </div>
                        <div style="display: flex; gap: 1.5rem; font-size: 0.8rem; color: #475569; margin-bottom: 0.4rem;">
                            <div>Model Prediction: {label_badge} ({item.get('confidence', 0.0) * 100:.1f}%)</div>
                            <div>Recommendation: {action_badge}</div>
                        </div>
                        <div style="font-size: 0.82rem; color: #334155;">
                            <strong>Reason:</strong> {item.get('reason')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Moderator Decision Actions
                act_c1, act_c2, act_c3, _ = st.columns([1.1, 1.1, 1.1, 2])
                with act_c1:
                    if st.button("Mark Safe (Approve)", key=f"app_{item_id}", use_container_width=True):
                        for c in st.session_state.comments:
                            if c["id"] == item_id:
                                c["status"] = "Reviewed - Approved"
                        st.rerun()
                with act_c2:
                    if st.button("Confirm Flag", key=f"flg_{item_id}", use_container_width=True):
                        for c in st.session_state.comments:
                            if c["id"] == item_id:
                                c["status"] = "Reviewed - Confirmed Flag"
                        st.rerun()
                with act_c3:
                    if st.button("Dismiss", key=f"dsm_{item_id}", use_container_width=True):
                        for c in st.session_state.comments:
                            if c["id"] == item_id:
                                c["status"] = "Dismissed"
                        st.rerun()

                st.markdown("<div style='margin-bottom: 0.4rem;'></div>", unsafe_allow_html=True)
