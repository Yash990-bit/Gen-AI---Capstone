import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="PropAI: Intelligent Real Estate",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
        min-height: 100vh;
    }
    /* Button hover effect */
    .stButton > button {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,255,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)
@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open('models/trained_model.pkl', 'rb'))
        label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
        columns = pickle.load(open('models/columns.pkl', 'rb'))
        # Attempt to load a Decision Tree model if it exists
        try:
            dt_model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
        except FileNotFoundError:
            dt_model = None
        return model, dt_model, label_encoder, columns
    except FileNotFoundError:
        st.error("Model files not found. Please run the training notebook first.")
        return None, None, None, None
model, dt_model, label_encoder, columns = load_artifacts()

if label_encoder is not None and columns is not None:
    locations = list(label_encoder.classes_)
    
    st.title("         🏢 Intelligent Property Price Prediction")
    st.markdown("### AI-Powered Real Estate Analytics", help="Get market value estimates based on Bangalore dataset.")
    
    col1, col_gap, col2 = st.columns([1.2, 0.1, 1.7])
    
    with col1:
        st.markdown("#### 🏠Property Details")
        st.markdown("<p style='color: #888; font-size: 0.9em; margin-top: -10px;'>Adjust the parameters below:</p>", unsafe_allow_html=True)
        # Model selector: show Decision Tree option only if artifact exists
        model_choice = "Random Forest"
        if dt_model is not None:
            model_choice = st.selectbox("� Model", ["Random Forest", "Decision Tree"], index=0)
        else:
            st.caption("Using Random Forest (Decision Tree model not found)")

        location = st.selectbox("�📍 Location", sorted(locations))
        total_sqft = st.number_input("📏 Total Sqft", min_value=300, max_value=10000, value=1000, step=50)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            bhk = st.number_input("🛏 BHK", min_value=1, max_value=10, value=2)
        with col_s2:
            bath = st.number_input("🚿 Bath", min_value=1, max_value=10, value=2)
            
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

    with col2:
        tab1, tab2 = st.tabs(["💵 Valuation", "📊 Insights"])
        with tab1:
            if predict_clicked:
                try:
                    if location in label_encoder.classes_:
                        loc_encoded = label_encoder.transform([location])[0]
                    else:
                        st.warning("Location not found in training data. Using 'other'.")
                        loc_encoded = label_encoder.transform(['other'])[0]
                except:
                    loc_encoded = 0

                features = np.array([[loc_encoded, total_sqft, bath, bhk]])
                # Choose model based on user selection
                selected_model = model
                if 'model_choice' in locals() and model_choice == 'Decision Tree':
                    if dt_model is not None:
                        selected_model = dt_model
                    else:
                        st.warning("Decision Tree model not found on disk. Falling back to Random Forest.")

                prediction = selected_model.predict(features)[0]
                
                st.markdown(
                    f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 25px; border-radius: 12px; border: 1px solid rgba(0, 255, 255, 0.2); text-align: center;">
                        <h4 style="margin: 0; color: #aaa;">Estimated Value</h4>
                        <h1 style="color: #00e676; margin: 10px 0; font-size: 3em;">₹ {prediction:.2f} <span style="font-size: 0.5em; color: #e0e0e0; font-weight: 500;">Lakhs</span></h1>
                        <p style="margin: 0; color: #ccc;"><strong>Market Range:</strong> ₹ {prediction * 0.9:.2f}L – ₹ {prediction * 1.1:.2f}L</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Analysis: Why this price?", expanded=True):
                    st.write(f"**Price per Sqft:** ₹ {prediction*100000/total_sqft:,.0f} / sqft")
                    
                    sens_data = []
                    for b in range(max(1, bhk-1), min(11, bhk+2)):
                        for ba in range(max(1, bath-1), min(11, bath+2)):
                            f_sens = np.array([[loc_encoded, total_sqft, ba, b]])
                            p_sens = selected_model.predict(f_sens)[0]
                            sens_data.append({"BHK": b, "Bath": ba, "Price": p_sens})
                    
                    sens_df = pd.DataFrame(sens_data)
                    
                    st.write("---")
                    st.markdown("##### Price Sensitivity")
                    st.caption("How changing BHK/Bath affects the price for this Square Footage:")
                    # Improved interactive sensitivity chart using Plotly
                    custom_colors = ["#034625", "#0d9696", "#c808ea", "#a99a16", "#A40909", "#0f6cbd"]
                    # Ensure Bath is treated as a category for separate lines
                    sens_df['Bath'] = sens_df['Bath'].astype(str)
                    fig_sens = px.line(
                        sens_df,
                        x='BHK',
                        y='Price',
                        color='Bath',
                        markers=True,
                        color_discrete_sequence=custom_colors,
                        labels={'Price': 'Estimated Price (Lakhs)', 'BHK': 'Bedrooms (BHK)', 'Bath': 'Bathrooms'}
                    )
                    fig_sens.update_layout(
                        title='Price Sensitivity by BHK and Bath',
                        legend_title='Bathrooms',
                        template='plotly_dark',
                        hovermode='x unified'
                    )
                    # Format hover to show currency and nicer numbers
                    fig_sens.update_traces(hovertemplate='BHK: %{x}<br>Bath: %{legendgroup}<br>Price: ₹ %{y:.2f} Lakhs')
                    st.plotly_chart(fig_sens, use_container_width=True)
                    st.caption("Note: Square Footage and Location have the highest impact. BHK/Bath shifts are more subtle.")
            else:
                st.markdown(
                    """
                    <div style="padding: 40px; text-align: center; color: #666; border: 2px dashed rgba(255,255,255,0.1); border-radius: 12px; margin-top: 20px;">
                        <h3>Ready for Valuation</h3>
                        <p>Adjust the property parameters on the left and click <b>Predict Price</b>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab2:
            st.markdown("#### Model Performance")
            col_m1, col_m2, col_m3 = st.columns(3)
            # Show architecture depending on which model is selected (or available)
            arch_display = "Random Forest"
            if 'model_choice' in locals() and model_choice == 'Decision Tree':
                arch_display = "Decision Tree"
            col_m1.metric("Architecture", arch_display)
            col_m2.metric("Accuracy (R²)", "76.0%")
            col_m3.metric("Avg Error", "₹ 23.2 Lakhs")
            
            st.divider()
            
            st.markdown("#### Factor Influence")
            # Use the selected model for feature importance when available
            selected_for_importance = model
            if 'model_choice' in locals() and model_choice == 'Decision Tree' and dt_model is not None:
                selected_for_importance = dt_model

            if hasattr(selected_for_importance, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Factor': columns,
                    'Weight (%)': selected_for_importance.feature_importances_ * 100
                }).sort_values(by='Weight (%)', ascending=False)
                
                # Interactive horizontal bar chart for feature importances
                importance_df_sorted = importance_df.sort_values('Weight (%)', ascending=True)
                fig_imp = px.bar(
                    importance_df_sorted,
                    x='Weight (%)',
                    y='Factor',
                    orientation='h',
                    color='Weight (%)',
                    color_continuous_scale='Viridis',
                    labels={'Weight (%)': 'Importance (%)', 'Factor': 'Feature'}
                )
                fig_imp.update_layout(title='Feature Importances', template='plotly_dark', xaxis_title='Importance (%)', yaxis_title='')
                fig_imp.update_traces(marker_line_color='rgba(0,0,0,0.3)', marker_line_width=1, hovertemplate='%{y}: %{x:.1f}%')
                st.plotly_chart(fig_imp, use_container_width=True)

                with st.expander("Impact Breakdown"):
                    for _, row in importance_df.iterrows():
                        st.write(f"- **{row['Factor'].replace('_', ' ').title()}**: {row['Weight (%)']:.1f}% impact")
                
                st.info("💡 **Insight:** Total Square ft is the most critical factor. This is why price doesn't change drastically when only BHK or Bathrooms are adjusted without changing the area.")
            else:
                st.info("Feature importance is not available.")
