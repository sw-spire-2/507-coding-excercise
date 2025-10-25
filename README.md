`.
├── 5D-data-for-linear-classification.csv
├── H3O+.csv
├── main.py
├── models
│   ├── basic_gp_models
│   │   ├── X_scaler.joblib
│   │   ├── evaluation_summary.csv
│   │   ├── gp_DotProduct.joblib
│   │   ├── gp_Matern.joblib
│   │   ├── gp_RBF.joblib
│   │   ├── gp_RationalQuadratic.joblib
│   │   ├── meta.json
│   │   ├── test_i.npy
│   │   ├── train_i.npy
│   │   ├── val_i.npy
│   │   └── y_scaler.joblib
│   ├── compare_krr_nn
│   │   ├── compare_fit_time_vs_n.png
│   │   └── compare_rmse_vs_n.png
│   ├── gp_complex_checkpoints
│   │   ├── level0_summary.csv
│   │   ├── level1_summary.csv
│   │   └── level2_summary.csv
│   ├── krr_h3o_models
│   │   ├── X_scaler_n1000.joblib
│   │   ├── X_scaler_n2000.joblib
│   │   ├── X_scaler_n60.joblib
│   │   ├── X_scaler_n600.joblib
│   │   ├── krr_n1000.joblib
│   │   ├── krr_n2000.joblib
│   │   ├── krr_n60.joblib
│   │   ├── krr_n600.joblib
│   │   ├── meta.json
│   │   ├── plots
│   │   │   ├── krr_fit_time_vs_n.png
│   │   │   ├── krr_r2_vs_n.png
│   │   │   ├── krr_rmse_vs_n.png
│   │   │   └── krr_scaling_combined.png
│   │   ├── summary.json
│   │   ├── test_i.npy
│   │   ├── train_i_n1000.npy
│   │   ├── train_i_n2000.npy
│   │   ├── train_i_n60.npy
│   │   ├── train_i_n600.npy
│   │   ├── val_i.npy
│   │   ├── y_scaler_n1000.joblib
│   │   ├── y_scaler_n2000.joblib
│   │   ├── y_scaler_n60.joblib
│   │   └── y_scaler_n600.joblib
│   ├── level2_gp_models
│   │   ├── X_scaler.joblib
│   │   ├── bic_vs_complexity.png
│   │   ├── evaluation_summary.csv
│   │   ├── gp_L1: (RQ) * LIN.joblib
│   │   ├── gp_L1: (RQ) * RBF.joblib
│   │   ├── gp_L1: (RQ) + MAT.joblib
│   │   ├── gp_L1: (RQ) + RBF.joblib
│   │   ├── gp_L1: (RQ) + RQ.joblib
│   │   ├── gp_L2: ((RQ) * LIN) * LIN.joblib
│   │   ├── gp_L2: ((RQ) * LIN) * RBF.joblib
│   │   ├── gp_L2: ((RQ) * LIN) + MAT.joblib
│   │   ├── gp_L2: ((RQ) * LIN) + RBF.joblib
│   │   ├── gp_L2: ((RQ) * LIN) + RQ.joblib
│   │   ├── gp_L2: ((RQ) * RBF) * LIN.joblib
│   │   ├── gp_L2: ((RQ) * RBF) * RBF.joblib
│   │   ├── gp_L2: ((RQ) * RBF) + MAT.joblib
│   │   ├── gp_L2: ((RQ) * RBF) + RBF.joblib
│   │   ├── gp_L2: ((RQ) * RBF) + RQ.joblib
│   │   ├── gp_L2: ((RQ) + MAT) * LIN.joblib
│   │   ├── gp_L2: ((RQ) + MAT) * RBF.joblib
│   │   ├── gp_L2: ((RQ) + MAT) + MAT.joblib
│   │   ├── gp_L2: ((RQ) + MAT) + RBF.joblib
│   │   ├── gp_L2: ((RQ) + MAT) + RQ.joblib
│   │   ├── gp_L2: ((RQ) + RBF) * LIN.joblib
│   │   ├── gp_L2: ((RQ) + RBF) * RBF.joblib
│   │   ├── gp_L2: ((RQ) + RBF) + MAT.joblib
│   │   ├── gp_L2: ((RQ) + RBF) + RBF.joblib
│   │   ├── gp_L2: ((RQ) + RBF) + RQ.joblib
│   │   ├── gp_L2: ((RQ) + RQ) * LIN.joblib
│   │   ├── gp_L2: ((RQ) + RQ) * RBF.joblib
│   │   ├── gp_L2: ((RQ) + RQ) + MAT.joblib
│   │   ├── gp_L2: ((RQ) + RQ) + RBF.joblib
│   │   ├── gp_L2: ((RQ) + RQ) + RQ.joblib
│   │   ├── gp_RQ.joblib
│   │   ├── meta.json
│   │   ├── rmse_vs_complexity.png
│   │   ├── selected_models_plotting.csv
│   │   ├── test_evaluation_summary.csv
│   │   ├── test_i.npy
│   │   ├── train_i.npy
│   │   ├── val_i.npy
│   │   └── y_scaler.joblib
│   ├── nn_h3o_models
│   │   ├── X_scaler_n1000.joblib
│   │   ├── X_scaler_n2000.joblib
│   │   ├── X_scaler_n60.joblib
│   │   ├── X_scaler_n600.joblib
│   │   ├── meta.json
│   │   ├── nn_n1000.keras
│   │   ├── nn_n2000.keras
│   │   ├── nn_n60.keras
│   │   ├── nn_n600.keras
│   │   ├── summary.json
│   │   ├── test_i.npy
│   │   ├── train_i_n1000.npy
│   │   ├── train_i_n2000.npy
│   │   ├── train_i_n60.npy
│   │   ├── train_i_n600.npy
│   │   ├── val_i.npy
│   │   ├── y_scaler_n1000.joblib
│   │   ├── y_scaler_n2000.joblib
│   │   ├── y_scaler_n60.joblib
│   │   └── y_scaler_n600.joblib
│   ├── pca_model
│   │   ├── Sigma.npy
│   │   ├── cumulative_evr.npy
│   │   ├── explained_variance_ratio.npy
│   │   ├── meta.json
│   │   ├── pca_cumulative_evr.png
│   │   ├── pca_model.joblib
│   │   └── pca_scree_plot.png
│   └── svm_model
│       ├── best_model.joblib
│       ├── cv_results.csv
│       ├── linear_intercept.npy
│       ├── linear_weights.npy
│       ├── summary.json
│       ├── svm_class_counts.png
│       └── svm_test_metrics.png
└── modules
    ├── __pycache__
    │   ├── evaluation.cpython-310.pyc
    │   ├── gp_train.cpython-310.pyc
    │   ├── krr_train.cpython-310.pyc
    │   ├── load.cpython-310.pyc
    │   ├── nn_train.cpython-310.pyc
    │   ├── pca_train.cpython-310.pyc
    │   ├── plot.cpython-310.pyc
    │   ├── save.cpython-310.pyc
    │   ├── svm_train.cpython-310.pyc
    │   └── synthesize.cpython-310.pyc
    ├── evaluation.py
    ├── gp_train.py
    ├── krr_train.py
    ├── load.py
    ├── nn_train.py
    ├── pca_train.py
    ├── plot.py
    ├── save.py
    ├── svm_train.py
    └── synthesize.py
`
