### Fraudulent Payment Detection

**Author: Tam Le**

#### Executive summary

To detect fraudulent payment attempts, this publication includes or references synthetic data provided by J.P. Morgan. Exploratory data analysis suggested that the fraudulent rate increases with the increase of the transaction amount. The original large and imbalanced dataset was split into two datasets, a **Low Amount dataset**, and a **High Amount dataset**, with **USD 1,000** - as the delinieation between the two datasets; this allows for different strategies of model training and evaluation for each dataset.

Each dataset was balanced to have the same number of fraudulent and non-fraudulent samples before training the models. Accuracy was used to measure the performance of the models trained in both Low and High Amount datasets. In addition, F1 score (a balance of Precision and Recall) was applied to the Low Amount dataset while F2 score (more weight was placed on Recall) was used to measure the performance of the models trained on the High Amount dataset. Additional measurements and criteria such as ROC-AUC (receiver operating characteristic – area under curve), AP (average precision), speed and interpretability were considered when evaluating the models.

On the **Low Amount** dataset, **eXtreme Gradient Boosting (XGB)** was selected based on its speed, performance and its acceptable interpretability. It achieved **92% Accuracy score, 92% F1 score, 98% ROC-AUC score** and **98% AP score**.

On the **High Amount** dataset, **Decision Trees (DT)** was selected with **perfect scores** of **100% for Accuracy, F2, ROC-AUC and AP**.

#### Rationale

An Online Payment Fraud study conducted by Juniper Research in 2023 forecasted USD 362 billion of total merchant losses from 2023 – 2028, with USD 91 billion losses in 2028 – an increase of more than double from USD 38 billion in 2023. The business objective of this study is to detect fraudulent payment attempts so that businesses and consumers will be better protected from financial loss due to fraudulent payment activities.

#### Research Question

To address this business objective, the goal of this project is to identify a predictive classification machine learning model(s) which effectively predicts whether a payment transaction is a fraudulent or legitimate payment transaction.

#### Data Sources

The synthetic payment dataset was provided by an international financial institution -J.P. Morgan. It describes payment information of 1,495,782 observations. The dataset contains transactional information including payment amount, senders and receivers of a large variety of payment transaction types enriched with labels representing legitimate or fraudulent payments.

#### Methodology

The Cross-Industry Standard Process for Data Mining (CRISP-DM) framework is applied to guide this effort. The framework includes six phases: business understanding, data understanding, data preparation, modelling, evaluation, and deployment.

![fig1](images/process.png)

**Phases of the CRISP-DM Process Model for Data Mining**

After understanding the business objectives, the collected data was explored by using visualizations and probability distributions to form initial findings and hypotheses. Then, the data was cleaned and prepared to handle any integrity issues. Features were engineered for modelling. Next, the dataset was split into a Low Amount dataset and a High Amount dataset. Four predictive classification models were built with default parameters with a cross-validation method applied on the Low Amount dataset. They were **Decision Trees (DT), Histogram Gradient Boosting (HGBT), eXtreme Gradient Boosting (XGB), and AdaBoost (AB)** classification models. Two models, **Decision Trees (DT) and Logistic Regression (LR)**, were built on the High Amount dataset. After eliminating one model on Low Amount dataset, the remaining three models were fine-tuned with optimal parameters. Lastly, these models or classifiers were compared so that the best model, based on a set of predefined criteria, would be evaluated and recommended.

#### Results

The synthetic dataset, provided by J.P. Morgan, with approximately 1.5 million observations was explored and split into a Low Amount dataset (below USD 1,000) and a High Amount dataset (at or above USD 1,000). The two datasets were transformed and balanced before training multiple models for selection. These models were evaluated with a set of pre-defined selection criteria. They are **speed, ease of interpretation and performance metrics (Accuracy, F-measure, AP and ROC-AUC)**.

On the balanced Low Amount dataset, **XGB** was selected. It achieved **92% Accuracy, 92% of F1 and 98% of AP and ROC-AUC**. The top three features that influenced the model are **the account and the country of the Sender** ( "Sender_Account”, "Sender_Country" atributes) and **account of the beneficiary** ("Bene_Account" attribute).

On the High Amount dataset, **DT** was selected with a perfect score of **100% Accuracy, F2, AP and ROC-AUC**. The top two features that influenced the selected model the most are the **transactional amount** ("USD_amount” attribute) and the **JPMC Client type of client** ("Sender_Cat_JPMC_Client" attribute).

#### Next steps

The project can be continued by implementing the last step of CRISP-DM which is deployment. The trained eXtreme Gradient Boosting (XGB) model or Decision Trees (DT) model could be used to detect fraudulent attempts depending on whether the transactional amount is below USD 1,000 or greater, equal to USD 1,000. In addition, a business process could be developed to handle transactions that are detected and flagged as fraudulent transactions by the models.

#### Outline of project

- [Fraud Payment Detection - Report](Fraud_Payment_Detection_Report.md)
- [Fraud Payment Detection - EDA](Fraud_Detection_EDA_Final.jpynb)
- [Fraud Payment Detection - Low Amount](Fraud_Detection_Low_Amount_Final.ipynb)
- [Fraud Payment Detection - High Amount](Fraud_Detection_High_Amount_Final.ipynb)

##### Contact and Further Information

Tam Le - tam.th.le@gmail.com - 415 424 6799