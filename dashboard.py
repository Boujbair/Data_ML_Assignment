import time
import streamlit as st
from PIL import Image
import requests
from src.training.train_pipeline import TrainingPipeline
from src.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
import joblib
from sklearn.metrics import confusion_matrix



data = pd.read_csv('data/raw/resume.csv')





# Define the Streamlit app
class ResumeClassificationApp:
    # data = pd.read_csv('data/raw/resume.csv')
    def __init__(self):
        self.sidebar_options = None
        self.pipeline = None

    def run(self):
        st.title("Resume Classification Dashboard")
        self.sidebar_options = st.sidebar.selectbox("Dashboard Modes", ("EDA", "Training", "Inference"))

        if self.sidebar_options == "EDA":
            self.run_eda_mode()
        elif self.sidebar_options == "Training":
            self.run_training_mode()
        else:
            self.run_inference_mode()

    def run_eda_mode(self):
        st.header("Exploratory Data Analysis")
        st.info("In this section, you are invited to create insightful graphs "
            "about the resume dataset that you were provided.")
        
        

        # Display basic statistics
        st.header("Statistical Descriptions")
        st.subheader("Dataset Overview")
        st.write(f"Number of Rows: {data.shape[0]}")
        st.write(f"Number of Columns: {data.shape[1]}")
        st.write("Column Names:", data.columns.tolist())

        st.subheader("Summary Statistics")
        st.write(data.describe())

        # Display data distribution
        st.header("Charts")
        st.subheader("Histogram")
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
        selected_column = st.selectbox("Select a numerical column to visualize", numerical_columns)
        st.write(f"Column: {selected_column}")
        
        # Create a histogram
        fig, ax = plt.subplots()
        sns.histplot(data=data, x=selected_column, kde=True, ax=ax, color='green')
        st.pyplot(fig)

        # Create a box plot
        st.subheader("Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=selected_column, ax=ax, color='blue')
        st.pyplot(fig)
        
        
        
    def create_training_pipeline(self):
        return TrainingPipeline()

    def train_pipeline(self, serialize, model_name):
        with st.spinner('Training pipeline, please wait...'):
            try:
                self.pipeline.train(serialize=serialize, model_name=model_name)
                self.pipeline.render_confusion_matrix()
            except Exception as e:
                st.error('Failed to train the pipeline!')
                st.exception(e)

    def display_training_results(self):
        accuracy, f1 = self.pipeline.get_model_performance()
        col1, col2 = st.columns(2)

        col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
        col2.metric(label="F1 score", value=str(round(f1, 4)))

        st.image(Image.open(CM_PLOT_PATH), width=850)    




    def run_training_mode(self):
        st.header("Pipeline Training")
        st.info("Before you proceed to training your pipeline. Make sure you "
            "have checked your training pipeline code and that it is set properly.")

        self.pipeline = self.create_training_pipeline()
        #accuracy, f1 = self.pipeline.get_model_performance()
        name = st.text_input('Pipeline name', placeholder='Naive Bayes')
        serialize = st.checkbox('Save pipeline')
        train = st.button('Train pipeline')

        data = pd.read_csv('data/raw/resume.csv')

        if train:
            self.train_pipeline(serialize, name)
            self.display_training_results()


        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['resume'], data['label'], test_size=0.2, random_state=42)

        # Choose vectorization method
        vectorization_method = st.selectbox("Select Vectorization Method", ("Count Vectorizer", "TF-IDF Vectorizer"))

        if vectorization_method == "Count Vectorizer":
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = TfidfVectorizer()

        # Choose a model
        model_choice = st.selectbox("Select a Model", ("Naive Bayes", "Random Forest"))

        if model_choice == "Naive Bayes":
            self.model = MultinomialNB()
        else:
            self.model = RandomForestClassifier(n_estimators=100)

        # Train the selected pipeline
        with st.spinner("Training the custom pipeline, please wait..."):
            vectorized_train_data = self.vectorizer.fit_transform(X_train)
            self.model.fit(vectorized_train_data, y_train)

        # Evaluate the pipeline
        vectorized_test_data = self.vectorizer.transform(X_test)
        predictions = self.model.predict(vectorized_test_data)
        f1 = f1_score(y_test, predictions, average='weighted')
        st.success("Custom pipeline trained and evaluated.")
        st.metric("F1 Score", f1)

        # Option to serialize the model
        serialize_model = st.checkbox("Serialize the Trained Model")

        if serialize_model:
            model_name = st.text_input("Enter the model name for serialization")
            if model_name:
                model_filename = f"models/{model_name}.joblib"
                joblib.dump(self.model, model_filename)
                st.success(f"Model serialized as {model_filename}")

        

    

    def run_inference_mode(self):
        st.header("Resume Inference")
        st.info("This section simplifies the inference process. "
            "Choose a test resume and observe the label that your trained pipeline will predict.")

        sample = st.selectbox(
            "Resume samples for inference",
            tuple(LABELS_MAP.values()),
            index=None,
            placeholder="Select a resume sample",
        )
        infer = st.button('Run Inference')

        if infer:
            self.run_inference(sample)


    def run_inference(self, sample):
        with st.spinner('Running inference...'):
            try:
                sample_file = "_".join(sample.upper().split()) + ".txt"
                with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                    sample_text = file.read()

                result = requests.post(
                    'http://localhost:9000/api/inference',
                    json={'text': sample_text}
                )
                st.success('Done!')
                label = LABELS_MAP.get(int(float(result.text)))
                st.metric(label="Status", value=f"Resume label: {label}")
            except Exception as e:
                st.error('Failed to call Inference API!')
                st.exception(e)


    

    

if __name__ == "__main__":
    app = ResumeClassificationApp()
    app.run()
