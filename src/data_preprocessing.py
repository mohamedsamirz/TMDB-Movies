
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
    def remove_stopwords(text):
        words = [word for word in text.split() if word.lower() not in stop_words]
        return " ".join(words)

    def lemmatize_text(text):
        words = [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(words)
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    def prepare_for_modeling(self, target='revenue'):
        """Prepare features and target for modeling."""
        if target not in self.data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        features = [col for col in self.data.columns if col != target]

        X = self.data[features]
        y = self.data[target]

        return X, y

    def convert_catgories_to_numerical(self):
        features = ['keywords', 'genres', 'overview', 'spoken_languages', 'production_countries','prduction_companies']
        self.data[features] = self.data[features].fillna('')
        self.data.drop_duplicates(inplace=True)
        


    def impute_missing_values_local(data, window=8):
        """Impute missing values using local window: mean for numeric, mode for categorical."""
        data = data.copy()
        
        num_cols = data.select_dtypes(include=["number"]).columns
        cat_cols = data.select_dtypes(include=["object"]).columns

        for col in num_cols:
            series = data[col]
            for idx, value in series.items():
                if pd.isna(value):
                    local_values = series[idx+1:idx+1+window].dropna()
                    if not local_values.empty:
                        series.at[idx] = local_values.mean()
            data[col] = series

        for col in cat_cols:
            series = data[col]
            for idx, value in series.items():
                if pd.isna(value):
                    local_values = series[idx+1:idx+1+window].dropna()
                    if not local_values.empty:
                        most_common = Counter(local_values).most_common(1)[0][0]
                        series.at[idx] = most_common
            data[col] = series

        return data

    def clean_data(self):
        """Filter rows with status 'Released' and drop the 'status' column"""
        self.data = self.data[self.data['status'] == 'Released']
        self.data = self.data.drop(columns=['status'], axis=1)
    def correct_data(self):
        """Convert any boolean columns into 0/1 integer columns."""
        bool_cols = self.data.select_dtypes(include=['bool']).columns
        self.data[bool_cols] = self.data[bool_cols].astype(int)
        return self
    def process_date(self):
        """Convert release_date to datetime and extract year, month, and quarter"""
        self.data['release_date'] = pd.to_datetime(self.data['release_date'], errors='coerce')
        self.data['release_year'] = self.data['release_date'].dt.year
        self.data['release_month'] = self.data['release_date'].dt.month
        self.data['release_quarter'] = self.data['release_date'].dt.quarter
        self.data = self.data.drop(columns=['release_date'], axis=1)
        return self
    def process_original_language(self, top_n=5):
        """One-hot encode top N most common original_language entries, group others as 'other'."""

        print("Processing 'original_language' column...")

        # Fill missing with 'other'
        self.data['original_language'] = self.data['original_language'].fillna('other')

        # Find top N most common languages
        top_languages = self.data['original_language'].value_counts().nlargest(top_n).index.tolist()
        print(f"Top {top_n} languages: {top_languages}")

        # Replace rare languages with 'other'
        self.data['original_language'] = self.data['original_language'].apply(
            lambda x: x if x in top_languages else 'other'
        )

        # One-hot encode
        lang_dummies = pd.get_dummies(self.data['original_language'], prefix='lang')
        print(f"Adding {lang_dummies.shape[1]} language columns.")

        # Concatenate back to original data
        self.data = pd.concat([self.data, lang_dummies], axis=1)

        # Drop the original column
        self.data = self.data.drop(columns=['original_language'])

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
        (self.clean_data()
            .process_date()
            .convert_catgories_to_numerical()
            .impute_missing_values_local()
            .handle_outliers())

        if save_path:
            self.save_data(save_path)

        return self

    def save_data(self, save_path: str):
        """Save processed dataset"""
        self.data.to_csv(save_path, index=False)
