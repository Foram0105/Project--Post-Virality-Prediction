import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Post Virality Analysis", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.title("Post Virality Analysis")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Analytics", "Prediction"],
        icons=["house", "bar-chart", "activity"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

# Load Data
@st.cache_data

def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = None

# Dashboard Tab
if selected == "Dashboard":
    st.header("Dashboard")
    if df is not None:
        st.subheader("First 5 Rows of the Dataset")
        st.dataframe(df.head())
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    else:
        st.warning("Please upload a CSV file to get started.")

# Analytics Tab
elif selected == "Analytics":
    st.header("Analytics")
    if df is not None:
        st.subheader("Bar Chart - Post Content Types")
        bar_data = df['text_content_type'].value_counts().reset_index()
        bar_data.columns = ['text_content_type', 'count']
        bar_fig = px.bar(bar_data, x='text_content_type', y='count', title="Count of Post Content Types")
        st.plotly_chart(bar_fig)

        st.subheader("Pie Chart - Platform Distribution")
        pie_fig = px.pie(df, names='platform', title="Platform Usage Distribution")
        st.plotly_chart(pie_fig)

        st.subheader("Box Plot - Engagement Metrics")
        for col in ['likes', 'shares', 'comments']:
            box_fig = px.box(df, y=col, title=f"Distribution of {col.title()}")
            st.plotly_chart(box_fig)

        scatter_fig = px.scatter(
            df,
            x='previous_engagement',  # ✅ CORRECT!
            y='virality_score',
            color='viral',
            title="Previous Engagement vs Virality Score"
        )
        st.plotly_chart(scatter_fig)

        st.subheader("Top 10 Posts by Virality Score")
        top_10 = df.sort_values(by='virality_score', ascending=False).head(10)
        st.dataframe(top_10[["meme_id", "text_content_type", "platform", "likes", "shares", "comments", "virality_score"]])

        st.subheader("Average Engagement by Platform")
        avg_engagement = df.groupby('platform')[['likes', 'shares', 'comments', 'virality_score']].mean().reset_index()
        st.dataframe(avg_engagement)
    else:
        st.warning("Please upload a dataset to see analytics.")

# Prediction Tab
elif selected == "Prediction":
    st.header("Post Virality Prediction")
    if df is not None:
        data = df.copy()
        # Encode categorical columns
        cat_cols = ['text_content_type', 'image_type', 'platform']
        encoders = {col: LabelEncoder().fit(data[col]) for col in cat_cols}
        for col, encoder in encoders.items():
            data[col] = encoder.transform(data[col])

        X = data[['text_content_type', 'image_type', 'platform', 'hashtags_used', 'previous_engagement']]
        y = data['viral']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        st.subheader("Enter Post Details")

        tct = st.selectbox("Text Content Type", encoders['text_content_type'].classes_)
        itype = st.selectbox("Image Type", encoders['image_type'].classes_)
        plat = st.selectbox("Platform", encoders['platform'].classes_)
        hashtags = st.slider("Hashtags Used", 0, 20, 5)
        engagement = st.number_input("Previous Engagement", min_value=0, step=1)

        if st.button("Predict"):
            input_df = pd.DataFrame({
                'text_content_type': [encoders['text_content_type'].transform([tct])[0]],
                'image_type': [encoders['image_type'].transform([itype])[0]],
                'platform': [encoders['platform'].transform([plat])[0]],
                'hashtags_used': [hashtags],
                'previous_engagement': [engagement]
            })

            prediction = model.predict(input_df)[0]
            if prediction == 1:
                st.success("This post is likely to go VIRAL 🔥")
            else:
                st.info("This post may not go viral.")
    else:
        st.warning("Please upload a dataset to make predictions.")
