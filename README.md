# Mobile-Price-Prediction
Mobile Price Prediction uses machine learning to estimate smartphone price ranges based on features like RAM, battery, camera, storage, and connectivity, helping users and businesses make informed pricing decisions.
## Objective
The objective of the Cellphone Price Prediction project is to build a machine learning model that accurately classifies mobile phones into four price ranges based on their specifications. By analyzing features such as RAM, battery power, camera quality, and connectivity options, the model helps identify how hardware components influence pricing. This prediction system can assist manufacturers, retailers, and consumers in understanding product segmentation. Ultimately, the goal is to develop an efficient and reliable model that supports decision-making in the mobile market.

---

## Introduction
The Cellphone Price Prediction project aims to analyze mobile phone specifications to determine their corresponding price range. With the rapid growth of smartphone technology, understanding how features such as RAM, battery capacity, camera performance, and display quality affect pricing has become essential. By applying machine learning techniques, this project builds models that can classify phones into predefined price categories. Such predictions can help consumers make informed choices and assist companies in designing competitive products based on market trends.

---

## Data Understanding
The dataset used for cellphone price prediction contains 2,000 mobile phone records with 21 features, including both numerical and categorical variables. These features describe key specifications such as battery power, RAM, pixel resolution, memory, camera strength, and network capabilities. The target variable, **price_range**, is evenly distributed into four classes: low, medium, high, and very high. No missing values are present, and all columns are numeric, making the data suitable for machine learning models. Overall, the dataset provides a comprehensive view of hardware characteristics that influence smartphone pricing.

---

## Basic Checks
To understand the dataset structure, several basic exploratory checks were performed. First, the dataset’s **shape and dimensions** confirmed 2,000 rows and 21 columns. Using `head()` and `tail()`, the top and bottom records were inspected to verify data consistency. The `info()` summary showed that all features are numerical and there are no missing values. The `describe()` function provided statistical insights such as mean, median, and standard deviation for each feature. Checks for **null values** returned zero, and analysis of **unique values** ensured that all columns contain valid ranges. No duplicated rows were found in the dataset. The target variable (**price_range**) is perfectly balanced with 500 samples in each class. Finally, a **correlation matrix** revealed that RAM and battery power have the strongest influence on price, helping identify the most important features for model building.

---

## Exploratory Data Analysis
The EDA began by examining the **distribution of the target variable**, which was found to be perfectly balanced across the four price ranges. Histograms of all numerical features were plotted to understand their spread, followed by **boxplots** to detect potential outliers in variables like battery power, RAM, and pixel dimensions. A **correlation heatmap** highlighted strong positive relationships, especially RAM and battery power with price range, and the **top 10 highly correlated features** were extracted for feature importance insights. Pairplots were created for key attributes such as RAM, battery power, and pixel resolution to visualize class separation. **Countplots** were used for categorical features like dual SIM, 4G, 3G, and touch screen support. Additionally, **scatter plots** were generated to explore important relationships such as RAM vs. price and battery vs. price. The **screen dimension distribution** (screen height and width) was also analyzed to understand how display size varies across phone categories.

---

## Data Preprocessing
In the preprocessing stage, the **target variable** was defined as `price_range`, while all remaining columns were treated as features. The dataset was then split into training and testing sets using an **80:20 train_test_split** with stratification to preserve class balance. Since **SVM** is sensitive to feature scales, the input variables were standardized using **StandardScaler** to ensure all features have equal influence during training. After scaling, the final shapes of `x_train_scaled` and `x_test_scaled` were verified to confirm correct preprocessing, making the dataset ready for  model development.

---

## Model Building, Training and Evaluation
1. The model-building phase began with training a **Support Vector Machine (SVM)** classifier, selected for its ability to handle high-dimensional data effectively. After the initial evaluation, **hyperparameter tuning** was applied to enhance its performance, resulting in a noticeable improvement in accuracy.
2. To broaden the comparison, additional models such as **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **XGBoost** were implemented. Logistic Regression served as a simple yet highly accurate baseline model, and tuning key parameters like regularization strength further boosted its performance.
3. The **KNN model**, however, delivered the weakest results among all the models. Its initial accuracy was significantly low, and even after tuning the number of neighbors and distance metrics, the improvement was minimal. This highlighted KNN's sensitivity to feature scaling and its overall inefficiency on this dataset, making it a poor performer compared to the other algorithms.
4. The **Decision Tree model** initially showed moderate performance, but tuning parameters such as maximum depth and minimum samples per split helped improve its generalization and reduce overfitting.
5. Finally, **XGBoost**, known for its robust gradient-boosting framework, was trained and further refined through hyperparameter tuning, achieving strong and reliable predictive performance.

Overall, each model—both before and after tuning—was trained on the preprocessed dataset and evaluated using accuracy scores and confusion matrices. This comprehensive comparison allowed for clear identification of the top-performing models while also revealing the limitations of weaker ones like KNN.

---
## Model Comparison Report: CellPhone Price Prediction

### Model Performance Comparison Report

#### **Performance in Metrics**

Model performance was evaluated using key classification metrics such as **precision, recall, F1-score, and accuracy**. Logistic Regression delivered the strongest overall results with consistently high precision and recall across all four classes, achieving an accuracy of **97.7%** after tuning. The SVM model also showed strong performance, reaching **94% accuracy** with well-balanced F1-scores, indicating reliable generalization. XGBoost performed competitively with an accuracy of **93%**, though it showed slightly lower recall in a few classes, suggesting moderate difficulty in separating some patterns. The Decision Tree model achieved **86% accuracy**, showing improvement after tuning but still displaying signs of overfitting in certain class predictions.KNN demonstrated the weakest performance, with accuracy improving only from **50% to 61.7%** after tuning. Its precision and recall remained comparatively low, indicating that KNN struggled to learn meaningful decision boundaries for this dataset.

| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-----------------------|--------------|---------------|------------|--------------|
| <span style="color:#2E86C1;"><b>Logistic Regression</b></span> | <b>0.977</b> | <b>0.98</b> | <b>0.98</b> | <b>0.98</b> |
| <span style="color:#AF7AC5;"><b>SVM</b></span>                | <b>0.94</b>  | <b>0.94</b> | <b>0.94</b> | <b>0.94</b> |
| <span style="color:#28B463;"><b>XGBoost</b></span>            | <b>0.93</b>  | <b>0.93</b> | <b>0.93</b> | <b>0.93</b> |
| <span style="color:#F39C12;"><b>Decision Tree</b></span>      | <b>0.86</b>  | <b>0.87</b> | <b>0.87</b> | <b>0.87</b> |
| <span style="color:#E74C3C;"><b>KNN</b></span>                | <b>0.617</b> | <b>0.62</b> | <b>0.62</b> | <b>0.62</b> |


### Best Model Selection
Based on the comparison of accuracy, precision, recall, and F1-score, **Logistic Regression** emerged as the best-performing model for cellphone price prediction. It achieved the highest accuracy of **97%** along with consistently strong precision and recall across all classes, indicating excellent generalization. While SVM and XGBoost, Decision Tree also delivered competitive results, their performance was slightly lower compared to Logistic Regression. Therefore, Logistic Regression is selected as the final best model amonst all due to its reliability, simplicity, and superior overall metrics.

---

## Business Analysis and Feature Importance Report**

From a business perspective, understanding which features drive the target variable is crucial for strategic decisions. In my analysis, I found that features like **RAM, battery power, and screen resolution** had the most significant impact on the price range prediction, indicating that customers value performance and display quality the most. Features such as **dual SIM, 3G/4G availability, and touch screen** also contributed, but to a lesser extent. By focusing on the most important features, businesses can optimize product offerings, prioritize key specifications, and make data-driven decisions to target customer preferences effectively.

---

### Conclusion

This project compared several machine learning models to predict mobile phone price ranges. **Logistic Regression** achieved the highest accuracy (**97.7%**), while SVM and XGBoost also performed well, and KNN lagged behind. Classical models like SVM and Logistic Regression offer faster training times and lower computational requirements compared to more complex models like XGBoost. Hyperparameter tuning improved performance across models, particularly for SVM and Logistic Regression. Feature analysis highlighted **RAM, battery power, and screen resolution** as key drivers, providing actionable insights for business decisions.

---

### Future Scope
The Cellphone Price Prediction project can be extended in several ways to enhance its utility and accuracy. Advanced feature engineering, such as combining screen resolution and display size, could improve model performance. Incorporating additional datasets with brand, user ratings, and market trends may allow for more precise price predictions. Deploying the best model as a **real-time web or mobile application** can help consumers and retailers make informed decisions. Finally, exploring deep learning models or ensemble techniques could further increase predictive accuracy and robustness.

---

## Recommendations for Customers
Based on the analysis and price range predictions, customers can make informed choices when buying a mobile phone:

1. **Low-cost phones (Class 0)** – Suitable for users needing basic features, such as calling, messaging, and light apps.
2. **Medium-cost phones (Class 1)** – Ideal for everyday users who want better performance, moderate RAM, and decent cameras.
3. **High-cost phones (Class 2)** – Recommended for users who use heavy apps, multitasking, and high-quality media.
4. **Premium/Very High-cost phones (Class 3)** – Best for power users, gamers, and professionals requiring top-notch performance, camera, and display quality.

These insights help customers select a phone that balances **performance and budget**, while manufacturers and retailers can target products to the right customer segment.

---

# Challenges Faced and Techniques Used

| **Challenge**                                   | **Description**                                                                                       | **Technique Used**                                                                                 |
|-------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Imbalanced Features                             | Some features had widely varying ranges, affecting model performance.                                 | Applied **StandardScaler** to standardize features for models sensitive to scaling like SVM.     |
| High Dimensionality                             | 21 features with some correlations could impact model efficiency.                                     | Conducted **correlation analysis** and focused on top correlated features for feature selection. |
| Model Selection                                 | Choosing the best-performing model among multiple classifiers.                                        | Trained and evaluated **SVM, Logistic Regression, XGBoost, Decision Tree, and KNN**.            |
| Low KNN Accuracy                                | KNN performed poorly due to high-dimensional data and feature scaling issues.                         | Applied **feature scaling** and experimented with different `k` values, but switched to stronger models. |
| Hyperparameter Tuning                            | Default model parameters did not yield the best results.                                             | Used **GridSearchCV** and manual tuning for SVM and XGBoost to improve accuracy.                  |
| Outliers in Features                             | Outliers in features like RAM and battery power could skew predictions.                               | Performed **EDA with boxplots** and verified model robustness; tree-based models handled outliers better. |
| Balanced Target Variable                         | Needed to ensure models learn all classes equally.                                                    | Used **stratified train-test split** to preserve class distribution in training and testing sets. |
| Interpretation and Comparison of Multiple Models | Difficulty in comparing models across multiple metrics.                                              | Created **comparison tables, line graphs, and confusion matrices** for easy visualization.       |

