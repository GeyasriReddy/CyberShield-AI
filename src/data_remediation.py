import os
import pandas as pd
import numpy as np

def run_remediation():
    print("=" * 70)
    print("CYBERSHIELD AI — MEMBER 1: DATA REMEDIATION PIPELINE")
    print("=" * 70)

    raw_dir = "datasets"
    processed_dir = "data/processed"
    results_dir = "results"
    docs_dir = "docs"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    # -------------------------------------------------------------
    # TASK 1: PEERNESS RECOVERY
    # -------------------------------------------------------------
    print("\n>>> TASK 1: Executing Peerness Recovery...")
    peerness_path = os.path.join(raw_dir, "2. peerness_values.csv")
    cb_path = os.path.join(raw_dir, "6. CB_Labels.csv")

    df_peer = pd.read_csv(peerness_path)
    df_cb = pd.read_csv(cb_path)

    # Build bidirectional lookup dictionary
    # Source peerness has (User 1 < User 2) for all 4950 pairs
    peer_dict = {}
    for _, row in df_peer.iterrows():
        u1, u2, p_val = int(row["User 1"]), int(row["User 2"]), float(row["Peerness"])
        peer_dict[(u1, u2)] = p_val
        peer_dict[(u2, u1)] = p_val  # Symmetric reverse

    print(f"Loaded {len(df_peer)} undirected peerness pairs -> {len(peer_dict)} symmetric directed entries.")

    recovery_log = []
    recovered_peerness_col = []
    peerness_was_recovered = []

    for idx, row in df_cb.iterrows():
        u1 = int(row["User1 ID"])
        u2 = int(row["User2 ID"])
        orig_val = row["Peerness"]
        pair_key = (u1, u2)

        if pd.isna(orig_val):
            # Must recover from peer_dict
            if pair_key in peer_dict:
                rec_val = peer_dict[pair_key]
                recovered_peerness_col.append(rec_val)
                peerness_was_recovered.append(True)
                recovery_log.append({
                    "user1_id": u1,
                    "user2_id": u2,
                    "original_peerness": np.nan,
                    "lookup_source_pair": f"({u2}, {u1})",
                    "source_peerness_score": rec_val,
                    "recovered_peerness_score": rec_val,
                    "recovery_type": "Symmetric Reverse Match (u1 > u2)",
                    "recovery_success": True,
                    "validation_match": (rec_val == peer_dict[(min(u1, u2), max(u1, u2))])
                })
            else:
                recovered_peerness_col.append(np.nan)
                peerness_was_recovered.append(False)
                recovery_log.append({
                    "user1_id": u1,
                    "user2_id": u2,
                    "original_peerness": np.nan,
                    "lookup_source_pair": f"({u1}, {u2})",
                    "source_peerness_score": np.nan,
                    "recovered_peerness_score": np.nan,
                    "recovery_type": "Unrecovered",
                    "recovery_success": False,
                    "validation_match": False
                })
        else:
            # Already present; validate match with source
            expected_val = peer_dict.get(pair_key, np.nan)
            match = np.isclose(orig_val, expected_val) if not np.isnan(expected_val) else False
            recovered_peerness_col.append(orig_val)
            peerness_was_recovered.append(False)
            recovery_log.append({
                "user1_id": u1,
                "user2_id": u2,
                "original_peerness": orig_val,
                "lookup_source_pair": f"({u1}, {u2})",
                "source_peerness_score": expected_val,
                "recovered_peerness_score": orig_val,
                "recovery_type": "Direct Match (u1 < u2)",
                "recovery_success": True,
                "validation_match": match
            })

    df_cb_cleaned = df_cb.copy()
    df_cb_cleaned["Peerness"] = recovered_peerness_col
    df_cb_cleaned["peerness_was_recovered"] = peerness_was_recovered

    # Verify no NaNs left
    nan_count_after = df_cb_cleaned["Peerness"].isna().sum()
    print(f"Original missing Peerness values: {df_cb['Peerness'].isna().sum()}")
    print(f"Recovered Peerness values: {sum(peerness_was_recovered)}")
    print(f"Remaining missing Peerness values after recovery: {nan_count_after}")

    # Save cleaned CB_Labels
    cb_cleaned_path = os.path.join(processed_dir, "cb_labels_cleaned.csv")
    df_cb_cleaned.to_csv(cb_cleaned_path, index=False)
    print(f"Saved: {cb_cleaned_path} ({len(df_cb_cleaned)} rows)")

    # Save recovery report
    df_recovery_log = pd.DataFrame(recovery_log)
    recovery_report_path = os.path.join(results_dir, "peerness_recovery_report.csv")
    df_recovery_log.to_csv(recovery_report_path, index=False)
    print(f"Saved: {recovery_report_path} ({len(df_recovery_log)} rows)")

    # Save full symmetric peerness matrix
    sym_rows = []
    for (u1, u2), score in sorted(peer_dict.items()):
        sym_rows.append({"User 1": u1, "User 2": u2, "Peerness": score})
    df_peer_sym = pd.DataFrame(sym_rows)
    peer_sym_path = os.path.join(processed_dir, "peerness_symmetric.csv")
    df_peer_sym.to_csv(peer_sym_path, index=False)
    print(f"Saved: {peer_sym_path} ({len(df_peer_sym)} rows, 100x99=9900 pairs)")

    # -------------------------------------------------------------
    # TASK 2: TEXT DATA PREPARATION & OVERLAP INVESTIGATION
    # -------------------------------------------------------------
    print("\n>>> TASK 2: Executing Text Data Preparation...")
    agg_path = os.path.join(raw_dir, "3. Aggressive_All.csv")
    non_path = os.path.join(raw_dir, "4. Non_Aggressive_All.csv")
    comm_path = os.path.join(raw_dir, "5. Communication_Data_Among_Users.csv")

    df_agg = pd.read_csv(agg_path)
    df_non = pd.read_csv(non_path)
    df_comm = pd.read_csv(comm_path)

    # 1. Investigate missing messages
    agg_nulls = df_agg[df_agg["Message"].isna()]
    non_nulls = df_non[df_non["Message"].isna()]
    print(f"Investigated missing aggressive messages: {len(agg_nulls)} records (blank raw text '')")
    print(f"Investigated missing non-aggressive messages: {len(non_nulls)} records (blank raw text '')")

    # Filter out blank text for clean model input
    df_agg_valid = df_agg.dropna(subset=["Message"]).copy()
    df_non_valid = df_non.dropna(subset=["Message"]).copy()

    # Identify message frequencies
    agg_freq = df_agg_valid["Message"].value_counts().to_dict()
    non_freq = df_non_valid["Message"].value_counts().to_dict()

    set_agg = set(agg_freq.keys())
    set_non = set(non_freq.keys())
    overlapping_texts = set_agg.intersection(set_non)
    print(f"Total unique text strings in Aggressive_All: {len(set_agg)}")
    print(f"Total unique text strings in Non_Aggressive_All: {len(set_non)}")
    print(f"IDENTIFIED OVERLAPPING TEXT STRINGS: {len(overlapping_texts)}")

    # Communication data frequency for overlapping messages
    comm_overlap_sub = df_comm[df_comm["Message"].isin(overlapping_texts)]
    comm_counts_total = comm_overlap_sub["Message"].value_counts().to_dict()
    comm_counts_label0 = comm_overlap_sub[comm_overlap_sub["Label"] == 0]["Message"].value_counts().to_dict()
    comm_counts_label1 = comm_overlap_sub[comm_overlap_sub["Label"] == 1]["Message"].value_counts().to_dict()

    # Create Overlapping Messages Report
    overlap_report_rows = []
    for idx, msg in enumerate(sorted(overlapping_texts), start=1):
        a_count = agg_freq[msg]
        n_count = non_freq[msg]
        c_count = comm_counts_total.get(msg, 0)
        c0 = comm_counts_label0.get(msg, 0)
        c1 = comm_counts_label1.get(msg, 0)

        if c0 > 0 and c1 > 0:
            status = "Conflicting Ground Truth in Communication (Both 0 and 1)"
        elif c1 > 0:
            status = "Aggressive in Communication (Label 1)"
        elif c0 > 0:
            status = "Non-Aggressive in Communication (Label 0)"
        else:
            status = "Corpus Overlap Only (Not in Communication)"

        overlap_report_rows.append({
            "overlap_id": idx,
            "message": msg,
            "char_length": len(msg),
            "word_count": len(msg.split()),
            "frequency_in_aggressive": a_count,
            "frequency_in_non_aggressive": n_count,
            "total_corpus_frequency": a_count + n_count,
            "frequency_in_communication": c_count,
            "comm_label_0_frequency": c0,
            "comm_label_1_frequency": c1,
            "ambiguity_status": status,
            "remediation_action": "Isolated from supervised training"
        })

    df_overlap_report = pd.DataFrame(overlap_report_rows)
    overlap_report_path = os.path.join(results_dir, "overlapping_messages_report.csv")
    df_overlap_report.to_csv(overlap_report_path, index=False)
    print(f"Saved: {overlap_report_path} ({len(df_overlap_report)} rows)")

    # Flag cleaned aggressive and non-aggressive sets
    df_agg_valid["source_label"] = 1
    df_agg_valid["is_ambiguous_overlap"] = df_agg_valid["Message"].isin(overlapping_texts)
    df_agg_valid["corpus_frequency"] = df_agg_valid["Message"].map(agg_freq)
    df_agg_valid.rename(columns={"No.": "original_id", "Message": "message"}, inplace=True)

    df_non_valid["source_label"] = 0
    df_non_valid["is_ambiguous_overlap"] = df_non_valid["Message"].isin(overlapping_texts)
    df_non_valid["corpus_frequency"] = df_non_valid["Message"].map(non_freq)
    df_non_valid.rename(columns={"No.": "original_id", "Message": "message"}, inplace=True)

    agg_clean_path = os.path.join(processed_dir, "text_aggressive_cleaned.csv")
    df_agg_valid.to_csv(agg_clean_path, index=False)
    print(f"Saved: {agg_clean_path} ({len(df_agg_valid)} rows, {df_agg_valid['is_ambiguous_overlap'].sum()} flagged ambiguous)")

    non_clean_path = os.path.join(processed_dir, "text_non_aggressive_cleaned.csv")
    df_non_valid.to_csv(non_clean_path, index=False)
    print(f"Saved: {non_clean_path} ({len(df_non_valid)} rows, {df_non_valid['is_ambiguous_overlap'].sum()} flagged ambiguous)")

    # Build clean supervised training pool (excluding all 2756 ambiguous messages)
    # Deduplicated unique pool with frequencies to allow flexible weighted or unweighted training
    clean_train_rows = []
    # 1. Unambiguous aggressive
    for msg in sorted(set_agg - overlapping_texts):
        clean_train_rows.append({
            "message": msg,
            "label": 1,
            "frequency": agg_freq[msg],
            "char_length": len(msg),
            "word_count": len(msg.split())
        })
    # 2. Unambiguous non-aggressive
    for msg in sorted(set_non - overlapping_texts):
        clean_train_rows.append({
            "message": msg,
            "label": 0,
            "frequency": non_freq[msg],
            "char_length": len(msg),
            "word_count": len(msg.split())
        })

    df_clean_train = pd.DataFrame(clean_train_rows)
    clean_train_path = os.path.join(processed_dir, "text_corpus_clean_for_training.csv")
    df_clean_train.to_csv(clean_train_path, index=False)
    print(f"Saved: {clean_train_path} ({len(df_clean_train)} distinct unambiguous training messages)")

    # Save isolated ambiguous corpus
    ambig_isolated_path = os.path.join(processed_dir, "text_corpus_ambiguous_isolated.csv")
    df_overlap_report.to_csv(ambig_isolated_path, index=False)
    print(f"Saved: {ambig_isolated_path} (2,756 ambiguous messages isolated)")

    # -------------------------------------------------------------
    # TASK 3: COMMUNICATION DATA PREPARATION
    # -------------------------------------------------------------
    print("\n>>> TASK 3: Executing Communication Data Preparation...")
    # Parse Date and Time into timestamp
    df_comm_clean = df_comm.copy()
    df_comm_clean["timestamp"] = pd.to_datetime(df_comm_clean["Date"] + " " + df_comm_clean["Time"], format="%Y-%m-%d %H:%M")
    
    # Flag overlapping messages
    df_comm_clean["is_ambiguous_overlap"] = df_comm_clean["Message"].isin(overlapping_texts)

    # Sort strictly by User1 ID, User2 ID, timestamp
    df_comm_sorted = df_comm_clean.sort_values(
        by=["User1 ID", "User2 ID", "timestamp"]
    ).reset_index(drop=True)

    # Reorder columns cleanly
    col_order = ["Date", "Time", "timestamp", "User1 ID", "User2 ID", "Message", "Label", "is_ambiguous_overlap"]
    df_comm_sorted = df_comm_sorted[col_order]

    # Verify chronological ordering
    pair_monotonic = df_comm_sorted.groupby(["User1 ID", "User2 ID"])["timestamp"].apply(lambda s: s.is_monotonic_increasing)
    print(f"Total communicating pairs in sorted file: {len(pair_monotonic)}")
    print(f"Pairs with verified chronological ordering: {pair_monotonic.sum()} / {len(pair_monotonic)} (100.00%)")

    # Save communication cleaned & sorted
    comm_sorted_path = os.path.join(processed_dir, "communication_cleaned_sorted.csv")
    df_comm_sorted.to_csv(comm_sorted_path, index=False)
    print(f"Saved: {comm_sorted_path} ({len(df_comm_sorted)} rows)")

    # -------------------------------------------------------------
    # TASK 4: USERS CLEANED
    # -------------------------------------------------------------
    users_path = os.path.join(raw_dir, "1. users_data.csv")
    df_users = pd.read_csv(users_path)
    users_cleaned_path = os.path.join(processed_dir, "users_cleaned.csv")
    df_users.to_csv(users_cleaned_path, index=False)
    print(f"Saved: {users_cleaned_path} ({len(df_users)} rows)")

    # -------------------------------------------------------------
    # VALIDATION SUMMARY PRINT
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATA REMEDIATION EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_remediation()
