# IPO Language & Aftermarket Performance — Evaluation Report

**Cross-validation:** 5-fold stratified  


---

## Target: `label_1w`

Naive accuracy baseline: **52.2%**

| Variant | Model | ROC-AUC | ± | Bal-Accuracy | ± | Pred +% | N |
|---------|-------|---------|---|-------------|---|---------|---|
| M1_text | logistic_regression | 0.516 | 0.032 | 0.511 | 0.030 | 52.9% | 425 |
| M1_text | ridge | 0.530 | 0.037 | 0.529 | 0.053 | 55.8% | 425 |
| M1_text | random_forest | 0.607 | 0.054 | 0.601 | 0.059 | 56.9% | 425 |
| M1_text | xgboost | 0.586 | 0.033 | 0.573 | 0.038 | 57.4% | 425 |
| M2_multiples | logistic_regression | 0.586 | 0.044 | 0.560 | 0.031 | 57.6% | 517 |
| M2_multiples | ridge | 0.593 | 0.052 | 0.563 | 0.047 | 51.3% | 517 |
| M2_multiples | random_forest | 0.592 | 0.072 | 0.573 | 0.066 | 56.7% | 517 |
| M2_multiples | xgboost | 0.575 | 0.055 | 0.554 | 0.033 | 51.8% | 517 |
| M3_combined | logistic_regression | 0.510 | 0.032 | 0.489 | 0.033 | 55.1% | 425 |
| M3_combined | ridge | 0.508 | 0.053 | 0.525 | 0.039 | 54.4% | 425 |
| M3_combined | random_forest | 0.593 | 0.056 | 0.580 | 0.077 | 56.7% | 425 |
| M3_combined | xgboost | 0.584 | 0.054 | 0.569 | 0.041 | 56.0% | 425 |

### Top Features


### M1_text

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_089` | 0.4352 |
| 2 | `emb_026` | 0.3624 |
| 3 | `emb_187` | 0.3601 |
| 4 | `emb_104` | 0.3584 |
| 5 | `emb_315` | 0.3144 |
| 6 | `emb_076` | 0.3068 |
| 7 | `emb_108` | 0.3059 |
| 8 | `emb_225` | 0.2993 |
| 9 | `emb_094` | 0.2978 |
| 10 | `loss_keyword_density` | 0.2957 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_378` | 0.8397 |
| 2 | `emb_383` | 0.8207 |
| 3 | `emb_071` | 0.7431 |
| 4 | `emb_250` | 0.7199 |
| 5 | `emb_076` | 0.7167 |
| 6 | `emb_262` | 0.7140 |
| 7 | `emb_369` | 0.6936 |
| 8 | `emb_304` | 0.6148 |
| 9 | `emb_011` | 0.5981 |
| 10 | `emb_217` | 0.5887 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_189` | 0.0132 |
| 2 | `emb_302` | 0.0104 |
| 3 | `emb_009` | 0.0101 |
| 4 | `emb_379` | 0.0094 |
| 5 | `emb_177` | 0.0092 |
| 6 | `emb_200` | 0.0087 |
| 7 | `emb_102` | 0.0082 |
| 8 | `emb_158` | 0.0073 |
| 9 | `emb_355` | 0.0073 |
| 10 | `emb_256` | 0.0068 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_189` | 0.0113 |
| 2 | `emb_247` | 0.0097 |
| 3 | `emb_324` | 0.0097 |
| 4 | `emb_141` | 0.0086 |
| 5 | `emb_187` | 0.0086 |
| 6 | `emb_123` | 0.0085 |
| 7 | `emb_010` | 0.0083 |
| 8 | `emb_302` | 0.0083 |
| 9 | `emb_094` | 0.0082 |
| 10 | `emb_231` | 0.0081 |


### M2_multiples

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.4277 |
| 2 | `is_hot_ipo_year` | 0.3401 |
| 3 | `cash_burn_proxy` | 0.2009 |
| 4 | `vix_30d_avg` | 0.1844 |
| 5 | `total_assets` | 0.1834 |
| 6 | `price_range_revised_up` | 0.1743 |
| 7 | `sector_etf_ret_90d` | 0.1676 |
| 8 | `cash` | 0.1510 |
| 9 | `revenue_prior` | 0.1500 |
| 10 | `has_insider_selling` | 0.1481 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `cash` | 0.2421 |
| 2 | `ipo_year` | 0.2169 |
| 3 | `is_hot_ipo_year` | 0.2103 |
| 4 | `net_income` | 0.1939 |
| 5 | `vix_30d_avg` | 0.1378 |
| 6 | `sector_etf_ret_90d` | 0.1040 |
| 7 | `ipos_same_quarter` | 0.1032 |
| 8 | `vix_on_ipo_date` | 0.0953 |
| 9 | `price_range_revised_up` | 0.0759 |
| 10 | `has_insider_selling` | 0.0741 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `sector_etf_ret_30d` | 0.1048 |
| 2 | `ipo_year` | 0.1041 |
| 3 | `sp500_ret_30d` | 0.0950 |
| 4 | `ipos_same_quarter` | 0.0849 |
| 5 | `sector_etf_ret_90d` | 0.0800 |
| 6 | `sp500_ret_90d` | 0.0762 |
| 7 | `sector_vs_sp500_30d` | 0.0730 |
| 8 | `vix_30d_avg` | 0.0705 |
| 9 | `vix_on_ipo_date` | 0.0702 |
| 10 | `ipos_same_month` | 0.0618 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.1440 |
| 2 | `has_insider_selling` | 0.0557 |
| 3 | `sector_etf_ret_30d` | 0.0521 |
| 4 | `sp500_ret_90d` | 0.0502 |
| 5 | `vix_30d_avg` | 0.0453 |
| 6 | `gross_profit` | 0.0436 |
| 7 | `sector_etf_ret_90d` | 0.0429 |
| 8 | `ebitda` | 0.0426 |
| 9 | `vix_on_ipo_date` | 0.0398 |
| 10 | `net_income` | 0.0394 |


### M3_combined

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_089` | 0.4259 |
| 2 | `emb_026` | 0.3928 |
| 3 | `emb_104` | 0.3612 |
| 4 | `emb_187` | 0.3590 |
| 5 | `is_hot_ipo_year` | 0.3401 |
| 6 | `loss_keyword_density` | 0.3141 |
| 7 | `emb_147` | 0.2999 |
| 8 | `ipo_year` | 0.2908 |
| 9 | `emb_315` | 0.2900 |
| 10 | `emb_076` | 0.2843 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_071` | 0.8936 |
| 2 | `emb_383` | 0.8569 |
| 3 | `emb_250` | 0.7986 |
| 4 | `emb_378` | 0.7802 |
| 5 | `emb_136` | 0.7789 |
| 6 | `emb_317` | 0.6601 |
| 7 | `emb_262` | 0.6177 |
| 8 | `emb_181` | 0.6001 |
| 9 | `emb_175` | 0.5954 |
| 10 | `emb_229` | 0.5891 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_189` | 0.0176 |
| 2 | `emb_200` | 0.0134 |
| 3 | `emb_302` | 0.0104 |
| 4 | `emb_158` | 0.0090 |
| 5 | `emb_064` | 0.0084 |
| 6 | `emb_256` | 0.0083 |
| 7 | `emb_061` | 0.0077 |
| 8 | `emb_080` | 0.0071 |
| 9 | `emb_006` | 0.0068 |
| 10 | `emb_288` | 0.0065 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_189` | 0.0104 |
| 2 | `emb_010` | 0.0103 |
| 3 | `emb_042` | 0.0099 |
| 4 | `emb_247` | 0.0098 |
| 5 | `emb_324` | 0.0097 |
| 6 | `emb_358` | 0.0096 |
| 7 | `ipo_year` | 0.0086 |
| 8 | `emb_187` | 0.0085 |
| 9 | `emb_302` | 0.0085 |
| 10 | `emb_171` | 0.0082 |


### Signal Interpretation


**xgboost:**

- M1 (text only): ROC-AUC = 0.586
- M2 (fundamentals only): ROC-AUC = 0.575
- M3 (combined): ROC-AUC = 0.584

**Mixed result:** Text adds marginal lift (+0.009 AUC). With a larger corpus the signal may become clearer.
Best variant achieves 0.586 AUC — modest lift above random. Consider expanding the IPO universe for stronger signal.

**logistic_regression:**

- M1 (text only): ROC-AUC = 0.516
- M2 (fundamentals only): ROC-AUC = 0.586
- M3 (combined): ROC-AUC = 0.510

**Text does not help:** M3 underperforms M2 by 0.077 AUC, suggesting text may be adding noise or overfitting with this sample size.
Best variant achieves 0.586 AUC — modest lift above random. Consider expanding the IPO universe for stronger signal.

### Notable Findings

- **`emb_089`** is the #1 feature in **M1_text/logistic_regression** (importance: 0.4352)
- **`emb_378`** is the #1 feature in **M1_text/ridge** (importance: 0.8397)
- **`emb_189`** is the #1 feature in **M1_text/random_forest** (importance: 0.0132)
- **`ipo_year`** is the #1 feature in **M2_multiples/logistic_regression** (importance: 0.4277)
- **`cash`** is the #1 feature in **M2_multiples/ridge** (importance: 0.2421)
- **`sector_etf_ret_30d`** is the #1 feature in **M2_multiples/random_forest** (importance: 0.1048)
- **`emb_071`** is the #1 feature in **M3_combined/ridge** (importance: 0.8936)

### SHAP Plots

**M1_text/logistic_regression**

![SHAP](plots/shap_label_1w_M1_text_logistic_regression.png)

**M1_text/ridge**

![SHAP](plots/shap_label_1w_M1_text_ridge.png)

**M1_text/random_forest**

![SHAP](plots/shap_label_1w_M1_text_random_forest.png)

**M1_text/xgboost**

![SHAP](plots/shap_label_1w_M1_text_xgboost.png)

**M2_multiples/logistic_regression**

![SHAP](plots/shap_label_1w_M2_multiples_logistic_regression.png)

**M2_multiples/ridge**

![SHAP](plots/shap_label_1w_M2_multiples_ridge.png)

**M2_multiples/random_forest**

![SHAP](plots/shap_label_1w_M2_multiples_random_forest.png)

**M2_multiples/xgboost**

![SHAP](plots/shap_label_1w_M2_multiples_xgboost.png)

**M3_combined/logistic_regression**

![SHAP](plots/shap_label_1w_M3_combined_logistic_regression.png)

**M3_combined/ridge**

![SHAP](plots/shap_label_1w_M3_combined_ridge.png)

**M3_combined/random_forest**

![SHAP](plots/shap_label_1w_M3_combined_random_forest.png)

**M3_combined/xgboost**

![SHAP](plots/shap_label_1w_M3_combined_xgboost.png)


---

## Target: `label_1m`

Naive accuracy baseline: **50.6%**

| Variant | Model | ROC-AUC | ± | Bal-Accuracy | ± | Pred +% | N |
|---------|-------|---------|---|-------------|---|---------|---|
| M1_text | logistic_regression | 0.612 | 0.029 | 0.581 | 0.044 | 52.2% | 425 |
| M1_text | ridge | 0.583 | 0.052 | 0.558 | 0.050 | 49.9% | 425 |
| M1_text | random_forest | 0.620 | 0.019 | 0.612 | 0.032 | 54.4% | 425 |
| M1_text | xgboost | 0.607 | 0.035 | 0.577 | 0.042 | 56.0% | 425 |
| M2_multiples | logistic_regression | 0.565 | 0.061 | 0.550 | 0.053 | 53.4% | 517 |
| M2_multiples | ridge | 0.551 | 0.062 | 0.542 | 0.067 | 46.4% | 517 |
| M2_multiples | random_forest | 0.619 | 0.054 | 0.578 | 0.044 | 49.5% | 517 |
| M2_multiples | xgboost | 0.597 | 0.075 | 0.566 | 0.055 | 47.6% | 517 |
| M3_combined | logistic_regression | 0.621 | 0.045 | 0.598 | 0.055 | 53.4% | 425 |
| M3_combined | ridge | 0.588 | 0.059 | 0.574 | 0.060 | 52.0% | 425 |
| M3_combined | random_forest | 0.632 | 0.026 | 0.615 | 0.025 | 53.6% | 425 |
| M3_combined | xgboost | 0.618 | 0.025 | 0.612 | 0.036 | 52.5% | 425 |

### Top Features


### M1_text

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_315` | 0.4775 |
| 2 | `emb_380` | 0.3421 |
| 3 | `emb_120` | 0.3367 |
| 4 | `emb_076` | 0.3354 |
| 5 | `emb_329` | 0.3271 |
| 6 | `emb_369` | 0.3184 |
| 7 | `emb_153` | 0.3177 |
| 8 | `emb_256` | 0.3085 |
| 9 | `emb_383` | 0.3015 |
| 10 | `emb_047` | 0.2944 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_232` | 0.8086 |
| 2 | `emb_383` | 0.8021 |
| 3 | `emb_378` | 0.7184 |
| 4 | `emb_367` | 0.6763 |
| 5 | `emb_356` | 0.6167 |
| 6 | `emb_380` | 0.6092 |
| 7 | `emb_250` | 0.6016 |
| 8 | `emb_152` | 0.5898 |
| 9 | `emb_233` | 0.5862 |
| 10 | `emb_005` | 0.5801 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_006` | 0.0116 |
| 2 | `emb_149` | 0.0112 |
| 3 | `emb_200` | 0.0112 |
| 4 | `emb_064` | 0.0096 |
| 5 | `emb_189` | 0.0094 |
| 6 | `emb_310` | 0.0085 |
| 7 | `emb_037` | 0.0077 |
| 8 | `emb_009` | 0.0076 |
| 9 | `emb_152` | 0.0076 |
| 10 | `emb_120` | 0.0076 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_073` | 0.0111 |
| 2 | `emb_002` | 0.0108 |
| 3 | `emb_092` | 0.0098 |
| 4 | `emb_175` | 0.0085 |
| 5 | `emb_038` | 0.0082 |
| 6 | `emb_284` | 0.0082 |
| 7 | `emb_189` | 0.0081 |
| 8 | `emb_011` | 0.0078 |
| 9 | `emb_064` | 0.0075 |
| 10 | `emb_174` | 0.0075 |


### M2_multiples

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `is_profitable` | 0.2121 |
| 2 | `cash_burn_proxy` | 0.1852 |
| 3 | `ipo_year` | 0.1765 |
| 4 | `total_assets` | 0.1653 |
| 5 | `net_income_pct_revenue` | 0.1615 |
| 6 | `cash` | 0.1475 |
| 7 | `proceeds_to_revenue_ratio` | 0.1455 |
| 8 | `revenue_prior` | 0.1337 |
| 9 | `gross_margin_pct` | 0.1312 |
| 10 | `gross_profit` | 0.1290 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `cash` | 0.2796 |
| 2 | `net_income` | 0.2345 |
| 3 | `ipo_year` | 0.0851 |
| 4 | `is_profitable` | 0.0722 |
| 5 | `vix_30d_avg` | 0.0689 |
| 6 | `cash_burn_proxy` | 0.0620 |
| 7 | `ipos_same_quarter` | 0.0618 |
| 8 | `proceeds_to_revenue_ratio` | 0.0608 |
| 9 | `net_income_pct_revenue` | 0.0554 |
| 10 | `total_assets` | 0.0527 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `sector_etf_ret_30d` | 0.1047 |
| 2 | `sp500_ret_90d` | 0.1005 |
| 3 | `vix_30d_avg` | 0.0957 |
| 4 | `ipos_same_month` | 0.0880 |
| 5 | `sector_vs_sp500_30d` | 0.0849 |
| 6 | `vix_on_ipo_date` | 0.0785 |
| 7 | `sp500_ret_30d` | 0.0774 |
| 8 | `sector_etf_ret_90d` | 0.0753 |
| 9 | `ipos_same_quarter` | 0.0722 |
| 10 | `net_income` | 0.0399 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipos_same_month` | 0.0946 |
| 2 | `vix_30d_avg` | 0.0664 |
| 3 | `ebitda` | 0.0655 |
| 4 | `sp500_ret_90d` | 0.0651 |
| 5 | `ipo_year` | 0.0619 |
| 6 | `ipos_same_quarter` | 0.0612 |
| 7 | `sector_etf_ret_30d` | 0.0606 |
| 8 | `vix_on_ipo_date` | 0.0595 |
| 9 | `sector_vs_sp500_30d` | 0.0581 |
| 10 | `sector_etf_ret_90d` | 0.0549 |


### M3_combined

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_315` | 0.4374 |
| 2 | `emb_369` | 0.3613 |
| 3 | `emb_329` | 0.3520 |
| 4 | `emb_380` | 0.3252 |
| 5 | `emb_256` | 0.3166 |
| 6 | `emb_383` | 0.3085 |
| 7 | `emb_076` | 0.2993 |
| 8 | `emb_153` | 0.2920 |
| 9 | `emb_047` | 0.2896 |
| 10 | `emb_184` | 0.2887 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_378` | 0.8228 |
| 2 | `emb_250` | 0.6867 |
| 3 | `emb_383` | 0.6774 |
| 4 | `emb_367` | 0.6472 |
| 5 | `emb_380` | 0.6271 |
| 6 | `emb_175` | 0.5998 |
| 7 | `emb_232` | 0.5976 |
| 8 | `emb_047` | 0.5701 |
| 9 | `emb_304` | 0.5654 |
| 10 | `emb_181` | 0.5644 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_006` | 0.0137 |
| 2 | `emb_064` | 0.0119 |
| 3 | `emb_253` | 0.0113 |
| 4 | `emb_013` | 0.0101 |
| 5 | `emb_258` | 0.0096 |
| 6 | `emb_189` | 0.0087 |
| 7 | `emb_200` | 0.0086 |
| 8 | `emb_149` | 0.0079 |
| 9 | `emb_250` | 0.0077 |
| 10 | `emb_158` | 0.0077 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_206` | 0.0116 |
| 2 | `emb_301` | 0.0114 |
| 3 | `emb_092` | 0.0099 |
| 4 | `emb_038` | 0.0091 |
| 5 | `emb_163` | 0.0079 |
| 6 | `emb_064` | 0.0079 |
| 7 | `emb_334` | 0.0074 |
| 8 | `emb_200` | 0.0074 |
| 9 | `emb_175` | 0.0071 |
| 10 | `emb_149` | 0.0067 |


### Signal Interpretation


**xgboost:**

- M1 (text only): ROC-AUC = 0.607
- M2 (fundamentals only): ROC-AUC = 0.597
- M3 (combined): ROC-AUC = 0.618

**Text adds signal:** M3 outperforms M2 by +0.020 AUC, suggesting filing language contains predictive information beyond financials.
Best variant achieves 0.618 AUC — modest lift above random. Consider expanding the IPO universe for stronger signal.

**logistic_regression:**

- M1 (text only): ROC-AUC = 0.612
- M2 (fundamentals only): ROC-AUC = 0.565
- M3 (combined): ROC-AUC = 0.621

**Text adds signal:** M3 outperforms M2 by +0.056 AUC, suggesting filing language contains predictive information beyond financials.
Best variant achieves 0.621 AUC — modest lift above random. Consider expanding the IPO universe for stronger signal.

### Notable Findings

- **`emb_315`** is the #1 feature in **M1_text/logistic_regression** (importance: 0.4775)
- **`emb_232`** is the #1 feature in **M1_text/ridge** (importance: 0.8086)
- **`emb_006`** is the #1 feature in **M1_text/random_forest** (importance: 0.0116)
- **`emb_073`** is the #1 feature in **M1_text/xgboost** (importance: 0.0111)
- **`is_profitable`** is the #1 feature in **M2_multiples/logistic_regression** (importance: 0.2121)
- **`cash`** is the #1 feature in **M2_multiples/ridge** (importance: 0.2796)
- **`sector_etf_ret_30d`** is the #1 feature in **M2_multiples/random_forest** (importance: 0.1047)
- **`ipos_same_month`** is the #1 feature in **M2_multiples/xgboost** (importance: 0.0946)
- **`emb_378`** is the #1 feature in **M3_combined/ridge** (importance: 0.8228)
- **`emb_206`** is the #1 feature in **M3_combined/xgboost** (importance: 0.0116)

### SHAP Plots

**M1_text/logistic_regression**

![SHAP](plots/shap_label_1m_M1_text_logistic_regression.png)

**M1_text/ridge**

![SHAP](plots/shap_label_1m_M1_text_ridge.png)

**M1_text/random_forest**

![SHAP](plots/shap_label_1m_M1_text_random_forest.png)

**M1_text/xgboost**

![SHAP](plots/shap_label_1m_M1_text_xgboost.png)

**M2_multiples/logistic_regression**

![SHAP](plots/shap_label_1m_M2_multiples_logistic_regression.png)

**M2_multiples/ridge**

![SHAP](plots/shap_label_1m_M2_multiples_ridge.png)

**M2_multiples/random_forest**

![SHAP](plots/shap_label_1m_M2_multiples_random_forest.png)

**M2_multiples/xgboost**

![SHAP](plots/shap_label_1m_M2_multiples_xgboost.png)

**M3_combined/logistic_regression**

![SHAP](plots/shap_label_1m_M3_combined_logistic_regression.png)

**M3_combined/ridge**

![SHAP](plots/shap_label_1m_M3_combined_ridge.png)

**M3_combined/random_forest**

![SHAP](plots/shap_label_1m_M3_combined_random_forest.png)

**M3_combined/xgboost**

![SHAP](plots/shap_label_1m_M3_combined_xgboost.png)


---

## Target: `label_6m`

Naive accuracy baseline: **65.9%**

| Variant | Model | ROC-AUC | ± | Bal-Accuracy | ± | Pred +% | N |
|---------|-------|---------|---|-------------|---|---------|---|
| M1_text | logistic_regression | 0.592 | 0.022 | 0.551 | 0.032 | 39.5% | 425 |
| M1_text | ridge | 0.558 | 0.033 | 0.509 | 0.040 | 40.9% | 425 |
| M1_text | random_forest | 0.608 | 0.044 | 0.549 | 0.018 | 17.6% | 425 |
| M1_text | xgboost | 0.630 | 0.055 | 0.582 | 0.044 | 27.8% | 425 |
| M2_multiples | logistic_regression | 0.633 | 0.059 | 0.596 | 0.048 | 45.8% | 517 |
| M2_multiples | ridge | 0.629 | 0.055 | 0.564 | 0.021 | 12.8% | 517 |
| M2_multiples | random_forest | 0.682 | 0.061 | 0.626 | 0.022 | 33.8% | 517 |
| M2_multiples | xgboost | 0.631 | 0.034 | 0.619 | 0.032 | 34.8% | 517 |
| M3_combined | logistic_regression | 0.620 | 0.033 | 0.570 | 0.012 | 36.2% | 425 |
| M3_combined | ridge | 0.563 | 0.032 | 0.535 | 0.043 | 39.5% | 425 |
| M3_combined | random_forest | 0.694 | 0.048 | 0.639 | 0.029 | 20.9% | 425 |
| M3_combined | xgboost | 0.732 | 0.042 | 0.676 | 0.047 | 32.0% | 425 |

### Top Features


### M1_text

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_026` | 0.4144 |
| 2 | `emb_235` | 0.3532 |
| 3 | `risk_factor_count_approx` | 0.3313 |
| 4 | `emb_172` | 0.3168 |
| 5 | `emb_118` | 0.3129 |
| 6 | `emb_345` | 0.3047 |
| 7 | `emb_353` | 0.3047 |
| 8 | `emb_323` | 0.2973 |
| 9 | `emb_371` | 0.2964 |
| 10 | `emb_011` | 0.2909 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_250` | 0.7369 |
| 2 | `emb_221` | 0.6769 |
| 3 | `emb_279` | 0.6738 |
| 4 | `emb_353` | 0.6733 |
| 5 | `emb_217` | 0.6455 |
| 6 | `emb_182` | 0.6169 |
| 7 | `emb_041` | 0.5555 |
| 8 | `emb_095` | 0.5301 |
| 9 | `emb_311` | 0.5134 |
| 10 | `emb_174` | 0.5127 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_288` | 0.0119 |
| 2 | `emb_367` | 0.0110 |
| 3 | `emb_189` | 0.0108 |
| 4 | `emb_279` | 0.0093 |
| 5 | `emb_198` | 0.0090 |
| 6 | `emb_280` | 0.0084 |
| 7 | `emb_006` | 0.0079 |
| 8 | `emb_004` | 0.0078 |
| 9 | `emb_323` | 0.0075 |
| 10 | `emb_166` | 0.0072 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_288` | 0.0099 |
| 2 | `emb_155` | 0.0099 |
| 3 | `emb_358` | 0.0093 |
| 4 | `emb_143` | 0.0093 |
| 5 | `emb_214` | 0.0092 |
| 6 | `emb_367` | 0.0090 |
| 7 | `emb_189` | 0.0087 |
| 8 | `emb_121` | 0.0083 |
| 9 | `emb_218` | 0.0079 |
| 10 | `emb_223` | 0.0078 |


### M2_multiples

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `vix_30d_avg` | 0.3645 |
| 2 | `ipo_year` | 0.3550 |
| 3 | `has_insider_selling` | 0.2069 |
| 4 | `ipos_same_quarter` | 0.1751 |
| 5 | `total_assets` | 0.1734 |
| 6 | `revenue_prior` | 0.1539 |
| 7 | `proceeds_to_revenue_ratio` | 0.1507 |
| 8 | `sp500_ret_90d` | 0.1367 |
| 9 | `ebitda` | 0.1349 |
| 10 | `gross_profit` | 0.1335 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `cash` | 0.2711 |
| 2 | `net_income` | 0.2368 |
| 3 | `vix_30d_avg` | 0.1920 |
| 4 | `ipo_year` | 0.1601 |
| 5 | `has_insider_selling` | 0.0964 |
| 6 | `ipos_same_quarter` | 0.0849 |
| 7 | `total_assets` | 0.0821 |
| 8 | `proceeds_to_revenue_ratio` | 0.0731 |
| 9 | `sp500_ret_90d` | 0.0726 |
| 10 | `gross_profit` | 0.0603 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `vix_30d_avg` | 0.1301 |
| 2 | `ipo_year` | 0.1128 |
| 3 | `vix_on_ipo_date` | 0.1104 |
| 4 | `ipos_same_quarter` | 0.0902 |
| 5 | `sector_etf_ret_90d` | 0.0881 |
| 6 | `sp500_ret_90d` | 0.0762 |
| 7 | `sector_vs_sp500_30d` | 0.0725 |
| 8 | `sector_etf_ret_30d` | 0.0725 |
| 9 | `sp500_ret_30d` | 0.0632 |
| 10 | `ipos_same_month` | 0.0578 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.2289 |
| 2 | `vix_30d_avg` | 0.0621 |
| 3 | `is_hot_ipo_year` | 0.0582 |
| 4 | `sp500_ret_90d` | 0.0500 |
| 5 | `ipos_same_quarter` | 0.0495 |
| 6 | `sector_etf_ret_90d` | 0.0476 |
| 7 | `insider_proceeds_pct` | 0.0462 |
| 8 | `sp500_ret_30d` | 0.0452 |
| 9 | `sector_etf_ret_30d` | 0.0389 |
| 10 | `sector_vs_sp500_30d` | 0.0378 |


### M3_combined

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `vix_30d_avg` | 0.4622 |
| 2 | `emb_026` | 0.4153 |
| 3 | `ipo_year` | 0.3975 |
| 4 | `emb_165` | 0.3410 |
| 5 | `emb_345` | 0.3367 |
| 6 | `uncertainty_in_risk` | 0.3083 |
| 7 | `risk_factor_count_approx` | 0.3045 |
| 8 | `emb_340` | 0.3033 |
| 9 | `emb_160` | 0.3018 |
| 10 | `emb_235` | 0.2985 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_250` | 0.8135 |
| 2 | `emb_353` | 0.7134 |
| 3 | `emb_095` | 0.5778 |
| 4 | `emb_217` | 0.5765 |
| 5 | `emb_232` | 0.5459 |
| 6 | `emb_084` | 0.5380 |
| 7 | `emb_191` | 0.5113 |
| 8 | `emb_369` | 0.5024 |
| 9 | `emb_311` | 0.4964 |
| 10 | `emb_124` | 0.4951 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `vix_30d_avg` | 0.0246 |
| 2 | `ipo_year` | 0.0188 |
| 3 | `vix_on_ipo_date` | 0.0174 |
| 4 | `ipos_same_quarter` | 0.0120 |
| 5 | `emb_367` | 0.0094 |
| 6 | `emb_288` | 0.0091 |
| 7 | `emb_279` | 0.0078 |
| 8 | `emb_004` | 0.0074 |
| 9 | `emb_118` | 0.0072 |
| 10 | `emb_261` | 0.0072 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.0184 |
| 2 | `emb_362` | 0.0121 |
| 3 | `emb_127` | 0.0101 |
| 4 | `emb_223` | 0.0094 |
| 5 | `emb_016` | 0.0094 |
| 6 | `emb_110` | 0.0092 |
| 7 | `emb_070` | 0.0090 |
| 8 | `vix_30d_avg` | 0.0089 |
| 9 | `emb_243` | 0.0088 |
| 10 | `emb_266` | 0.0082 |


### Signal Interpretation


**xgboost:**

- M1 (text only): ROC-AUC = 0.630
- M2 (fundamentals only): ROC-AUC = 0.631
- M3 (combined): ROC-AUC = 0.732

**Text adds signal:** M3 outperforms M2 by +0.101 AUC, suggesting filing language contains predictive information beyond financials.
Best variant achieves 0.732 AUC — meaningfully above the 0.5 random baseline.

**logistic_regression:**

- M1 (text only): ROC-AUC = 0.592
- M2 (fundamentals only): ROC-AUC = 0.633
- M3 (combined): ROC-AUC = 0.620

**Text does not help:** M3 underperforms M2 by 0.012 AUC, suggesting text may be adding noise or overfitting with this sample size.
Best variant achieves 0.633 AUC — modest lift above random. Consider expanding the IPO universe for stronger signal.

### Notable Findings

- **`emb_026`** is the #1 feature in **M1_text/logistic_regression** (importance: 0.4144)
- **`emb_250`** is the #1 feature in **M1_text/ridge** (importance: 0.7369)
- **`emb_288`** is the #1 feature in **M1_text/random_forest** (importance: 0.0119)
- **`vix_30d_avg`** is the #1 feature in **M2_multiples/logistic_regression** (importance: 0.3645)
- **`cash`** is the #1 feature in **M2_multiples/ridge** (importance: 0.2711)
- **`ipo_year`** is the #1 feature in **M2_multiples/xgboost** (importance: 0.2289)

### SHAP Plots

**M1_text/logistic_regression**

![SHAP](plots/shap_label_6m_M1_text_logistic_regression.png)

**M1_text/ridge**

![SHAP](plots/shap_label_6m_M1_text_ridge.png)

**M1_text/random_forest**

![SHAP](plots/shap_label_6m_M1_text_random_forest.png)

**M1_text/xgboost**

![SHAP](plots/shap_label_6m_M1_text_xgboost.png)

**M2_multiples/logistic_regression**

![SHAP](plots/shap_label_6m_M2_multiples_logistic_regression.png)

**M2_multiples/ridge**

![SHAP](plots/shap_label_6m_M2_multiples_ridge.png)

**M2_multiples/random_forest**

![SHAP](plots/shap_label_6m_M2_multiples_random_forest.png)

**M2_multiples/xgboost**

![SHAP](plots/shap_label_6m_M2_multiples_xgboost.png)

**M3_combined/logistic_regression**

![SHAP](plots/shap_label_6m_M3_combined_logistic_regression.png)

**M3_combined/ridge**

![SHAP](plots/shap_label_6m_M3_combined_ridge.png)

**M3_combined/random_forest**

![SHAP](plots/shap_label_6m_M3_combined_random_forest.png)

**M3_combined/xgboost**

![SHAP](plots/shap_label_6m_M3_combined_xgboost.png)


---

## Target: `label_1y`

Naive accuracy baseline: **69.2%**

| Variant | Model | ROC-AUC | ± | Bal-Accuracy | ± | Pred +% | N |
|---------|-------|---------|---|-------------|---|---------|---|
| M1_text | logistic_regression | 0.644 | 0.064 | 0.578 | 0.050 | 35.1% | 425 |
| M1_text | ridge | 0.587 | 0.050 | 0.550 | 0.037 | 38.8% | 425 |
| M1_text | random_forest | 0.649 | 0.072 | 0.528 | 0.037 | 10.6% | 425 |
| M1_text | xgboost | 0.605 | 0.051 | 0.553 | 0.021 | 20.9% | 425 |
| M2_multiples | logistic_regression | 0.685 | 0.070 | 0.659 | 0.050 | 46.4% | 517 |
| M2_multiples | ridge | 0.688 | 0.067 | 0.574 | 0.019 | 13.9% | 517 |
| M2_multiples | random_forest | 0.680 | 0.038 | 0.615 | 0.026 | 36.6% | 517 |
| M2_multiples | xgboost | 0.676 | 0.039 | 0.618 | 0.021 | 35.4% | 517 |
| M3_combined | logistic_regression | 0.687 | 0.073 | 0.621 | 0.064 | 36.7% | 425 |
| M3_combined | ridge | 0.615 | 0.053 | 0.586 | 0.054 | 36.9% | 425 |
| M3_combined | random_forest | 0.687 | 0.084 | 0.556 | 0.034 | 13.6% | 425 |
| M3_combined | xgboost | 0.661 | 0.057 | 0.618 | 0.031 | 31.8% | 425 |

### Top Features


### M1_text

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_026` | 0.4664 |
| 2 | `n_sections_found` | 0.4507 |
| 3 | `emb_378` | 0.3388 |
| 4 | `emb_118` | 0.3322 |
| 5 | `emb_235` | 0.3108 |
| 6 | `emb_239` | 0.2963 |
| 7 | `emb_266` | 0.2818 |
| 8 | `uncertainty_density` | 0.2758 |
| 9 | `emb_262` | 0.2739 |
| 10 | `emb_264` | 0.2722 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_041` | 0.6163 |
| 2 | `emb_353` | 0.5999 |
| 3 | `emb_050` | 0.5700 |
| 4 | `emb_250` | 0.5673 |
| 5 | `emb_002` | 0.5554 |
| 6 | `emb_198` | 0.5518 |
| 7 | `emb_065` | 0.5462 |
| 8 | `emb_265` | 0.5419 |
| 9 | `emb_285` | 0.5377 |
| 10 | `emb_272` | 0.5186 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_120` | 0.0103 |
| 2 | `emb_030` | 0.0100 |
| 3 | `emb_249` | 0.0084 |
| 4 | `emb_118` | 0.0083 |
| 5 | `emb_282` | 0.0083 |
| 6 | `emb_189` | 0.0080 |
| 7 | `risk_to_total_ratio` | 0.0080 |
| 8 | `emb_291` | 0.0074 |
| 9 | `emb_273` | 0.0071 |
| 10 | `emb_232` | 0.0070 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_074` | 0.0109 |
| 2 | `emb_275` | 0.0103 |
| 3 | `emb_362` | 0.0103 |
| 4 | `emb_280` | 0.0091 |
| 5 | `emb_232` | 0.0080 |
| 6 | `emb_010` | 0.0080 |
| 7 | `emb_269` | 0.0080 |
| 8 | `emb_283` | 0.0079 |
| 9 | `emb_026` | 0.0079 |
| 10 | `emb_036` | 0.0078 |


### M2_multiples

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.4425 |
| 2 | `vix_30d_avg` | 0.4041 |
| 3 | `is_hot_ipo_year` | 0.2958 |
| 4 | `has_insider_selling` | 0.2697 |
| 5 | `sp500_ret_90d` | 0.2509 |
| 6 | `total_assets` | 0.2065 |
| 7 | `ipos_same_month` | 0.1854 |
| 8 | `revenue_prior` | 0.1711 |
| 9 | `ipos_same_quarter` | 0.1675 |
| 10 | `ebitda` | 0.1581 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `cash` | 0.2773 |
| 2 | `net_income` | 0.2499 |
| 3 | `vix_30d_avg` | 0.2320 |
| 4 | `ipo_year` | 0.1930 |
| 5 | `is_hot_ipo_year` | 0.1410 |
| 6 | `sp500_ret_90d` | 0.1225 |
| 7 | `has_insider_selling` | 0.1187 |
| 8 | `vix_on_ipo_date` | 0.1018 |
| 9 | `ipos_same_quarter` | 0.0808 |
| 10 | `total_assets` | 0.0748 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.1180 |
| 2 | `ipos_same_quarter` | 0.1115 |
| 3 | `vix_on_ipo_date` | 0.0909 |
| 4 | `ipos_same_month` | 0.0892 |
| 5 | `vix_30d_avg` | 0.0822 |
| 6 | `sp500_ret_30d` | 0.0760 |
| 7 | `sector_etf_ret_90d` | 0.0725 |
| 8 | `sector_etf_ret_30d` | 0.0721 |
| 9 | `sp500_ret_90d` | 0.0708 |
| 10 | `sector_vs_sp500_30d` | 0.0707 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.2177 |
| 2 | `ipos_same_month` | 0.0574 |
| 3 | `ebitda` | 0.0551 |
| 4 | `has_insider_selling` | 0.0483 |
| 5 | `gross_profit` | 0.0476 |
| 6 | `sp500_ret_30d` | 0.0445 |
| 7 | `ipos_same_quarter` | 0.0410 |
| 8 | `vix_30d_avg` | 0.0407 |
| 9 | `is_hot_ipo_year` | 0.0398 |
| 10 | `vix_on_ipo_date` | 0.0388 |


### M3_combined

**logistic_regression** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_026` | 0.4577 |
| 2 | `is_hot_ipo_year` | 0.4161 |
| 3 | `ipo_year` | 0.3955 |
| 4 | `vix_30d_avg` | 0.3823 |
| 5 | `n_sections_found` | 0.3793 |
| 6 | `profit_loss_ratio` | 0.3389 |
| 7 | `ipos_same_quarter` | 0.3209 |
| 8 | `emb_266` | 0.3205 |
| 9 | `emb_065` | 0.3159 |
| 10 | `emb_264` | 0.3105 |

**ridge** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `emb_250` | 0.8252 |
| 2 | `emb_265` | 0.6172 |
| 3 | `emb_353` | 0.5993 |
| 4 | `emb_111` | 0.5015 |
| 5 | `emb_285` | 0.5011 |
| 6 | `emb_050` | 0.5007 |
| 7 | `emb_373` | 0.4913 |
| 8 | `emb_198` | 0.4825 |
| 9 | `emb_246` | 0.4695 |
| 10 | `emb_242` | 0.4669 |

**random_forest** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.0233 |
| 2 | `ipos_same_quarter` | 0.0209 |
| 3 | `ipos_same_month` | 0.0182 |
| 4 | `emb_282` | 0.0099 |
| 5 | `emb_030` | 0.0097 |
| 6 | `emb_189` | 0.0083 |
| 7 | `vix_30d_avg` | 0.0081 |
| 8 | `emb_028` | 0.0070 |
| 9 | `emb_120` | 0.0069 |
| 10 | `emb_343` | 0.0065 |

**xgboost** — top 10 features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `ipo_year` | 0.0194 |
| 2 | `emb_255` | 0.0124 |
| 3 | `emb_281` | 0.0110 |
| 4 | `emb_348` | 0.0106 |
| 5 | `emb_124` | 0.0102 |
| 6 | `emb_209` | 0.0095 |
| 7 | `emb_199` | 0.0093 |
| 8 | `emb_180` | 0.0091 |
| 9 | `emb_047` | 0.0091 |
| 10 | `emb_310` | 0.0089 |


### Signal Interpretation


**xgboost:**

- M1 (text only): ROC-AUC = 0.605
- M2 (fundamentals only): ROC-AUC = 0.676
- M3 (combined): ROC-AUC = 0.661

**Text does not help:** M3 underperforms M2 by 0.016 AUC, suggesting text may be adding noise or overfitting with this sample size.
Best variant achieves 0.676 AUC — meaningfully above the 0.5 random baseline.

**logistic_regression:**

- M1 (text only): ROC-AUC = 0.644
- M2 (fundamentals only): ROC-AUC = 0.685
- M3 (combined): ROC-AUC = 0.687

**Mixed result:** Text adds marginal lift (+0.003 AUC). With a larger corpus the signal may become clearer.
Best variant achieves 0.687 AUC — meaningfully above the 0.5 random baseline.

### Notable Findings

- **`emb_026`** is the #1 feature in **M1_text/logistic_regression** (importance: 0.4664)
- **`emb_041`** is the #1 feature in **M1_text/ridge** (importance: 0.6163)
- **`emb_120`** is the #1 feature in **M1_text/random_forest** (importance: 0.0103)
- **`emb_074`** is the #1 feature in **M1_text/xgboost** (importance: 0.0109)
- **`ipo_year`** is the #1 feature in **M2_multiples/logistic_regression** (importance: 0.4425)
- **`cash`** is the #1 feature in **M2_multiples/ridge** (importance: 0.2773)
- **`emb_250`** is the #1 feature in **M3_combined/ridge** (importance: 0.8252)

### SHAP Plots

**M1_text/logistic_regression**

![SHAP](plots/shap_label_1y_M1_text_logistic_regression.png)

**M1_text/ridge**

![SHAP](plots/shap_label_1y_M1_text_ridge.png)

**M1_text/random_forest**

![SHAP](plots/shap_label_1y_M1_text_random_forest.png)

**M1_text/xgboost**

![SHAP](plots/shap_label_1y_M1_text_xgboost.png)

**M2_multiples/logistic_regression**

![SHAP](plots/shap_label_1y_M2_multiples_logistic_regression.png)

**M2_multiples/ridge**

![SHAP](plots/shap_label_1y_M2_multiples_ridge.png)

**M2_multiples/random_forest**

![SHAP](plots/shap_label_1y_M2_multiples_random_forest.png)

**M2_multiples/xgboost**

![SHAP](plots/shap_label_1y_M2_multiples_xgboost.png)

**M3_combined/logistic_regression**

![SHAP](plots/shap_label_1y_M3_combined_logistic_regression.png)

**M3_combined/ridge**

![SHAP](plots/shap_label_1y_M3_combined_ridge.png)

**M3_combined/random_forest**

![SHAP](plots/shap_label_1y_M3_combined_random_forest.png)

**M3_combined/xgboost**

![SHAP](plots/shap_label_1y_M3_combined_xgboost.png)
