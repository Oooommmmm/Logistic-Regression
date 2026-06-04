##Steps Performed##
<ul>
<li>Feature Reduction & Cleaning: Ingested the medical diagnostic dataset into Pandas and stripped extraneous attributes (id and Unnamed: 32) to eliminate non-predictive formatting artifacts and structural noise before training.

<li>Target Variable Encoding: Standardized the qualitative categorical output by mapping text-based diagnostics (M / B) into binary integers (1 for Malignant, 0 for Benign) to establish a clean machine-learning target.

<li>Stratified Data Partitioning: Segmented the cleaned attributes and target arrays into an 80% training and 20% testing split, explicitly leveraging stratification (stratify=y) to tightly maintain original class proportions across both sets.

<li>Feature Standardization: Fit and applied a Z-score standardization scaler (StandardScaler) to smooth out continuous variable differences, transforming multiple feature dimensions into a uniform Gaussian distribution with a mean of 0 and variance of 1.

<li>Log-Linear Classifier Ingestion: Instantiated and optimized a LogisticRegression classification model, computing parameter weights and linear log-odds lines directly over the scaled training matrices.

<li>Probability Mapping: Extracted raw continuous prediction probabilities (predict_proba) on the evaluation test subset, allowing for granular adjustments beyond standard discrete binary outputs.

<li>Clinical Performance Benchmarking: Generated a multi-metric diagnostic suite including Precision, Recall, and an extensive classification report to evaluate model alignment at the default decision boundary.

<li>Decision Boundary Tuning: Adjusted the model's classification threshold down from 0.5 to a custom 0.25 restriction, actively prioritizing high clinical sensitivity (Recall) to safely minimize life-threatening False Negatives.

<li>Visual Evaluation Frameworks: Rendered side-by-side performance plots capturing the underlying Confusion Matrix heatmap distribution and the predictive trade-offs of the Receiver Operating Characteristic (ROC) curve.

<li>Functional Mapping: Computed and plotted a standalone visual representation of the continuous Sigmoid Function curve to explicitly track how multi-dimensional log-odds map onto final mathematical probability outcomes.
</ul>
