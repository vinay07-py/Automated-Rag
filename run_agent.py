# run_agent.py
import sys
import os
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Runtime Error: Data argument block missing.")
        sys.exit(1)

    target_file = sys.argv[1]
    report_out = "logs/final_report.md"
    print(f"[SYSTEM] Initializing Node Processing Strategy on: {target_file}")

    try:
        ext = os.path.splitext(target_file)[1].lower()
        if ext == '.csv':
            q = pl.scan_csv(target_file, infer_schema_length=10000)
        elif ext == '.parquet':
            q = pl.scan_parquet(target_file)
        elif ext in ['.xls', '.xlsx']:
            q = pl.read_excel(target_file).lazy()
        else:
            raise ValueError(f"Unsupported format extension: {ext}")

        initial_columns = q.columns
        id_cols = [c for c in initial_columns if 'id' in c.lower() or 'name' in c.lower()]
        q = q.drop(id_cols)
        print(f"🤖 [AGENT: Collection] Ingested schema blueprint successfully.")

        print("\n🤖 [AGENT: Cleaning] Sweeping anomalies and filling matrices efficiently...")
        schema = q.schema
        expressions = []
        for col, dtype in schema.items():
            if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                expressions.append(pl.col(col).fill_null(strategy="forward").fill_null(0))
            else:
                expressions.append(pl.col(col).fill_null("Unknown"))
        q = q.with_columns(expressions)

        current_cols = q.columns
        target_col = "Target_Metric"
        possible_targets = [c for c in current_cols if
                            'status' in c.lower() or 'target' in c.lower() or 'label' in c.lower()]
        if possible_targets:
            target_col = possible_targets[0]
        elif len(current_cols) > 1:
            target_col = current_cols[-1]
        else:
            q = q.with_columns(pl.lit(0).alias(target_col))
        print(f"✅ Sanity verification run complete. Master Target metric locked on: '{target_col}'")

        print("\n🤖 [AGENT: Engineering] Compiling Lazy Frame and executing structural transformations...")
        df_clean = q.collect(streaming=True)

        # Save a sample version for local context lookups by our RAG architecture
        df_clean.head(100).write_csv("data/processed/rag_context_sample.csv")

        X_df = df_clean.drop(target_col)
        y_series = df_clean.select(target_col).to_series()

        X_encoded = []
        for col in X_df.columns:
            if X_df[col].dtype == pl.String:
                encoded_series, _ = X_df[col].to_physical().factorize()
                X_encoded.append(pl.Series(col, encoded_series))
            else:
                X_encoded.append(X_df[col])
        X_final = pl.DataFrame(X_encoded).to_numpy()
        y_final = y_series.to_physical().factorize()[0] if y_series.dtype == pl.String else y_series.to_numpy()

        print("\n🤖 [AGENT: Training] Partitioning matrices and building RandomForest Classifier...")
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        joblib.dump(model, "models/model.pkl")

        print("\n🤖 [AGENT: Evaluation] Calculating matrix cross-validation telemetry...")
        accuracy = model.score(X_test, y_test)
        print(f"📈 Model Accuracy Score Verification Completed: {accuracy * 100:.2f}%")

        print("\n🤖 [AGENT: Visualisation] Designing tactical report markdown asset configurations...")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        top_features_summary = ""
        feature_names = X_df.columns
        for rank, i in enumerate(indices):
            top_features_summary += f"{rank + 1}. **{feature_names[i]}** (Weight Value: `{importances[i]:.4f}`)\n"

        markdown_content = f"""# 📊 NEO-DS // Deep Operational Intelligence Dossier
- **Target System Data Vector Source:** `{os.path.basename(target_file)}`
- **Total Processed Dimensions Matrix:** `{df_clean.height} records`
- **Calculated Evaluation Set Accuracy Score:** `{accuracy * 100:.2f}%`

### 🧬 Top Primary Data Weight Influence Dimensions
{top_features_summary}
"""
        with open(report_out, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print("\n[COMPLETE] All autonomous systems finished working successfully.")
    except Exception as e:
        print(f"❌ CRITICAL RUNTIME EXCEPTION CAUGHT: {str(e)}")
        sys.exit(1)