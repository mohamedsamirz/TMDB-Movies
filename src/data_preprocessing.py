import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ast import literal_eval
import re
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
class TMDBPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.genre_columns = []
        self.country_columns = []
        self.language_columns = []

    def load_data(self):
        """Load dataset"""
        self.data = pd.read_csv(os.path.join(self.data_path))
        return self

    def drop_columns(self, columns: list):
        """Remove specified columns from dataset"""
        existing_cols = [col for col in columns if col in self.data.columns]
        self.data = self.data.drop(columns=existing_cols, axis=1)
        return self

    def get_columns(self) -> list:
        """Get current dataframe columns"""
        return self.data.columns.tolist()

    def prepare_for_modeling(self, target='revenue'):
        available_cols = self.get_columns()
        num_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()

        if target in num_cols:
            num_cols.remove(target)
        if target in cat_cols:
            cat_cols.remove(target)

        list_cols = [col for col in cat_cols if isinstance(self.data[col].iloc[0], list)]

        hasher = FeatureHasher(n_features=100, input_type='string')

        def hash_column(col):
            hashed_data = hasher.transform(self.data[col].apply(lambda x: ' '.join(x) if isinstance(x, list) else ''))
            hashed_df = pd.DataFrame(hashed_data.toarray(), columns=[f"{col}_{i}" for i in range(hashed_data.shape[1])])
            return hashed_df

        hashed_columns = [hash_column(col) for col in list_cols]
        self.data = pd.concat([self.data] + hashed_columns, axis=1).drop(columns=list_cols)

        num_cols = [col for col in self.data.select_dtypes(include=['number']).columns.tolist() if col != target]
        cat_cols = [col for col in self.data.select_dtypes(include=['object']).columns.tolist()]

        transformers = []
        if num_cols:
            transformers.append(('num', StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols))

        preprocessor = ColumnTransformer(
            transformers,
            remainder='drop'  # Force drop any leftover columns
        )

        if target not in self.data.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        features = [col for col in self.data.columns if col != target]
        X = self.data[features]
        y = self.data[target]

        # Apply preprocessor
        X = preprocessor.fit_transform(X)

        # Ensure numeric
        import numpy as np
        if hasattr(X, "toarray"):
            X = X.toarray()
        if X.dtype == object:
            X = np.array(X, dtype=np.float64)

        # Just to be safe
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                print(f"⚠️ Column '{col}' is still object type.")
        # Now apply PCA
        pca = PCA(n_components=0.95)
        X = pca.fit_transform(X)
        return X, y
    def process_genres(self):
        """Vectorized genre processing"""
        exploded = self.data.explode('genres')
        genre_dummies = pd.crosstab(exploded.index, exploded['genres'])
        genre_dummies.columns = [f'genre_{col}' for col in genre_dummies.columns]
        self.data = self.data.join(genre_dummies.astype('int8'))
        self.genre_columns = genre_dummies.columns.tolist()
        return self

    def process_countries(self):
        """Batch country processing"""
        exploded = self.data.explode('production_countries')
        top_countries = exploded['production_countries'].value_counts().head(5).index

        country_dummies = pd.DataFrame({
            f'prod_{country}': self.data['production_countries'].apply(
                lambda x: 1 if country in x else 0
            ) for country in top_countries
        }).astype('int8')

        self.data = pd.concat([self.data, country_dummies], axis=1)
        self.country_columns = country_dummies.columns.tolist()
        return self

    def process_languages(self):
        """Vectorized language processing"""
        exploded = self.data.explode('spoken_languages')
        lang_dummies = pd.crosstab(exploded.index, exploded['spoken_languages'])
        lang_dummies.columns = [f'lang_{col}' for col in lang_dummies.columns]
        self.data = self.data.join(lang_dummies.astype('int8'))
        self.language_columns = lang_dummies.columns.tolist()
        return self

    def create_financial_features(self):
        """Vectorized financial features"""
        money_cols = ['budget', 'revenue']
        self.data[money_cols] = self.data[money_cols].apply(
            pd.to_numeric, errors='coerce'
        ).replace(0, np.nan)

        self.data['profit'] = self.data['revenue'] - self.data['budget']
        self.data['roi'] = (self.data['profit'] / self.data['budget']).replace([np.inf, -np.inf], np.nan)
        self.data['profitability'] = pd.cut(
            self.data['roi'],
            bins=[-np.inf, 0, 1, 2, np.inf],
            labels=['Loss', 'Low', 'Medium', 'High']
        )
        return self

    def create_temporal_features(self):
        """Vectorized date features"""
        self.data['release_date'] = pd.to_datetime(self.data['release_date'], errors='coerce')
        self.data['release_year'] = self.data['release_date'].dt.year
        self.data['release_month'] = self.data['release_date'].dt.month
        self.data['release_quarter'] = self.data['release_date'].dt.quarter
        return self

    def impute_missing_values(self):
        """Batch dynamic imputation"""
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        num_cols = self.data.select_dtypes(include=["number"]).columns
        cat_cols = self.data.select_dtypes(include=["object"]).columns

        self.data[num_cols] = num_imputer.fit_transform(self.data[num_cols])
        self.data[cat_cols] = cat_imputer.fit_transform(self.data[cat_cols])
        return self

    def handle_outliers(self):
        """Dynamic outlier handling"""
        # Detect numerical columns dynamically (after imputation)
        num_cols = self.data.select_dtypes(include=['number']).columns.tolist()

        # Clip upper outliers for each numerical column
        for col in num_cols:
            upper = self.data[col].quantile(0.95)
            self.data[col] = np.where(self.data[col] > upper, upper, self.data[col])

        return self

    def process_data(self, save_path=None):
        """Optimized end-to-end workflow"""
        (self.process_genres()
            .process_countries()
            .process_languages()
            .create_financial_features()
            .create_temporal_features()
            .impute_missing_values()
            .handle_outliers())

        # Optimize memory usage (only for one-hot encoded features)
        self.data = self.data.astype({
            col: 'int8' for col in self.genre_columns + self.country_columns + self.language_columns
        })

        if save_path:
            self.save_data(save_path)

        return self

    def save_data(self, save_path: str):
        """Save processed dataset"""
        self.data.to_csv(save_path, index=False)
