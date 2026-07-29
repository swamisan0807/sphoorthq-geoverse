"""Returns the platform's architecture/dataflow diagram as Mermaid source -
kept as code (not just a static doc) so notebooks can render it inline via
IPython and it never drifts out of sync with docs/architecture.md, which
embeds the same string."""

ARCHITECTURE_DIAGRAM = """
flowchart TB
    subgraph SOURCES["Cloud Sources"]
        S3["AWS S3"]
        ADLS["Azure ADLS Gen2"]
        GCS["Google Cloud Storage"]
        HTTP["Public HTTPS bucket"]
    end

    subgraph INGEST["src/ingestion"]
        CONN["IngestionConnector\\n(s3 / adls / gcs / http / local)"]
    end

    RAW[("datasets/raw/\\nlocal, analysis-ready")]

    subgraph PROCESS["notebooks/02_process\\n+ src/processing"]
        CAL["Lee speckle filter\\n+ linear-to-dB"]
    end

    subgraph FEAT["notebooks/03_feature_engineering"]
        TEX["Texture / polarimetric\\nfeatures + NaN masking"]
        VEC["Per-pixel feature vectors\\n(vv/vh/ratio/diff/local stats)"]
    end

    subgraph CLASSICAL["Classical ML"]
        RF["Random Forest\\n(notebook 04, pixel-wise)"]
        UNET["U-Net\\n(notebook 09, patch-wise)"]
    end

    subgraph QUANTUM["Quantum ML - src/qml\\n(FORCE_SIMULATION default)"]
        IBM["IBM Quantum Runtime\\n(QiskitRuntimeService / AerSimulator)"]
        BRAKET["AWS Braket\\n(AwsDevice / LocalSimulator)"]
        QKERNEL["Quantum kernel SVM\\n(batched gram-matrix jobs)"]
    end

    HYBRID["Hybrid ensemble\\n(notebook 06, classical + QML vote)"]
    EVAL["Evaluation\\n(src/ai/objectives)"]
    ROBUST["Robustness sweep\\n(notebook 08: 5 perturbations x 5 events)"]

    subgraph OBS["Observability - src/observability"]
        LOG["RunLogger\\n(per-stage timing + metrics)"]
        RUNS[("datasets/reports/runs/*.json")]
    end

    S3 --> CONN
    ADLS --> CONN
    GCS --> CONN
    HTTP --> CONN
    CONN --> RAW
    RAW --> CAL --> TEX --> VEC
    VEC --> RF
    VEC --> UNET
    VEC --> IBM --> QKERNEL
    VEC --> BRAKET --> QKERNEL
    RF --> HYBRID
    QKERNEL --> HYBRID
    HYBRID --> EVAL
    RF --> ROBUST
    EVAL --> ROBUST

    CONN -.logs.-> LOG
    CAL -.logs.-> LOG
    VEC -.logs.-> LOG
    RF -.logs.-> LOG
    UNET -.logs.-> LOG
    QKERNEL -.logs.-> LOG
    EVAL -.logs.-> LOG
    ROBUST -.logs.-> LOG
    LOG --> RUNS
""".strip()


def print_diagram() -> None:
    print(ARCHITECTURE_DIAGRAM)
