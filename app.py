import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from mlp import MLP, preprocess_data
import os

# Set page config
st.set_page_config(
    page_title="NBA Combine Prediction",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 500;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .page-title {
        color: #1E88E5;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    """Load the preprocessed data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'combined.csv')
    return pd.read_csv(data_path)

def create_prediction_plot():
    """Create and return the prediction visualization using Plotly"""
    # Load and preprocess data
    X_train, X_test, y_train, y_test, names_train, names_test = preprocess_data()
    
    # Create and train the model
    model = MLP(input_dim=X_train.shape[1], hidden_dim1=32, hidden_dim2=16)
    model.train(X_train, y_train, lr=0.01, epochs=5000)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Create a subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Training Data: Predicted vs Actual (R² = {train_r2:.4f})',
            f'Test Data: Predicted vs Actual (R² = {test_r2:.4f})'
        )
    )
    
    # Add training data scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_train.flatten(),
            y=y_pred_train.flatten(),
            mode='markers',
            name='Training Data',
            marker=dict(color='blue', size=8, opacity=0.6),
            text=names_train,
            hovertemplate="<b>%{text}</b><br>" +
                          "Actual: %{x:.2f}<br>" +
                          "Predicted: %{y:.2f}<br>" +
                          "<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add test data scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_test.flatten(),
            y=y_pred_test.flatten(),
            mode='markers',
            name='Test Data',
            marker=dict(color='green', size=8, opacity=0.6),
            text=names_test,
            hovertemplate="<b>%{text}</b><br>" +
                          "Actual: %{x:.2f}<br>" +
                          "Predicted: %{y:.2f}<br>" +
                          "<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add perfect prediction lines
    min_train = min(y_train.min(), y_pred_train.min())
    max_train = max(y_train.max(), y_pred_train.max())
    min_test = min(y_test.min(), y_pred_test.min())
    max_test = max(y_test.max(), y_pred_test.max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_train, max_train],
            y=[min_train, max_train],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[min_test, max_test],
            y=[min_test, max_test],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        title_text="MLP Model: Predicted vs Actual ROOKIE_SCORE",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Actual ROOKIE_SCORE", row=1, col=1)
    fig.update_xaxes(title_text="Actual ROOKIE_SCORE", row=1, col=2)
    fig.update_yaxes(title_text="Predicted ROOKIE_SCORE", row=1, col=1)
    fig.update_yaxes(title_text="Predicted ROOKIE_SCORE", row=1, col=2)
    
    return fig, train_r2, test_r2

def main():
    # Initialize session state for page tracking
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar navigation
    st.sidebar.markdown('<p class="sidebar-title">Navigation</p>', unsafe_allow_html=True)
    
    # Add a logo or icon at the top of the sidebar
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 2.5rem;">🏀</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home", key="home_btn"):
        st.session_state.current_page = "Home"
    
    if st.sidebar.button("📊 Model Visualization", key="viz_btn"):
        st.session_state.current_page = "Model Visualization"
    
    # Add some spacing
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

    
    # Display content based on current page
    if st.session_state.current_page == "Home":
        # Landing page
        st.title("🏀 NBA Combine Prediction")
        st.markdown("""
        ## Project Overview
        This project aims to predict NBA rookie performance using combine data and machine learning techniques.
        
        ### Data Sources
        - NBA Combine measurements (height, weight, wingspan, etc.)
        - Strength and agility metrics
        - Rookie year statistics

                    
        ### Model
        The project uses a Multilayer Perceptron (MLP) neural network to predict rookie performance
        based on combine measurements. The model takes into account:
        - Anthropometric measurements
        - Strength and agility metrics
        
        ### How to Use
        1. Navigate to the "Model Visualization" page using the sidebar
        2. View the model's predictions vs actual performance
        3. Analyze the model's accuracy and performance metrics
        """)
        
        # Display sample data
        st.subheader("Sample Data")
        df = load_data()
        st.dataframe(df.head())
        
        # Display data statistics
        st.subheader("Data Statistics")
        st.write(df.describe())
        
    elif st.session_state.current_page == "Model Visualization":
        st.title("Model Performance Visualization")
        
        # Add a description
        st.markdown("""
        This interactive visualization shows the model's predictions versus actual rookie performance scores.
        - **Blue dots**: Training data predictions
        - **Green dots**: Test data predictions
        - **R² Score**: Indicates how well the model fits the data
        
        **Interactive Features:**
        - Hover over data points to see player details
        - Zoom in/out using the mouse wheel or pinch gestures
        - Pan by clicking and dragging
        """)
        
        # Create and display the plot
        with st.spinner("Generating visualization..."):
            try:
                fig, train_r2, test_r2 = create_prediction_plot()
                st.plotly_chart(fig, use_container_width=True)
                
                # Display model metrics
                st.subheader("Model Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training R² Score", f"{train_r2:.4f}")
                with col2:
                    st.metric("Test R² Score", f"{test_r2:.4f}")
                with col3:
                    st.metric("Model Status", "Trained")
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                st.info("Please make sure the MLP model is properly set up and the data is available.")

if __name__ == "__main__":
    main() 