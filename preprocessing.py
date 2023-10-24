from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# cat_names = features.select_dtypes(include=["category"]).columns
# num_names = features.select_dtypes(exclude=["category"]).columns


# Create a custom transformer for LabelEncoding multiple columns
class MultiLabelEncoder:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in range(X.shape[1]):
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
        return X_encoded


# Create transformers for each feature type
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", MultiLabelEncoder()),
    ]
)

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# # Combine the transformers into a preprocessor
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, num_names),
#         ("cat", categorical_transformer, cat_names),
#     ]
# )

# # Create the main pipeline for the entire data preprocessing
# pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# # Fit and transform the data using the pipeline
# features_preprocessed = pipeline.fit_transform(features)
