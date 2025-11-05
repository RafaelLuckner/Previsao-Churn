import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="🎯 Previsão de Churn",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CSS SIMPLES ====================
st.markdown("""
<style>
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CACHE DE MODELOS ====================
@st.cache_resource(show_spinner=False)
def load_models():
    """Carrega todos os modelos de ML com cache otimizado"""
    try:
        LR_pipeline = joblib.load('models/logistic_regression_pipeline.pkl')
        RF_pipeline = joblib.load('models/random_forest_pipeline.pkl')
        XGB_pipeline = joblib.load('models/xgboost_pipeline.pkl')
        eclf1 = joblib.load('models/ensemble_pipeline.pkl')
        return LR_pipeline, RF_pipeline, XGB_pipeline, eclf1
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelos: {str(e)}")
        return None, None, None, None

@st.cache_data(show_spinner=False)
def load_data():
    """Carrega dados com cache"""
    try:
        df = pd.read_csv('data/raw/Telco_Customer_Churn.csv')
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        return None

# ==================== FUNÇÕES AUXILIARES ====================
def create_gauge_chart(probability, title):
    """Cria gráfico de gauge para probabilidade"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 52}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ffffff"},
            'steps': [
                {'range': [0, 33], 'color': "#00ff15"},
                {'range': [33, 66], 'color': "#CE9F03"},
                {'range': [66, 100], 'color': "#d1001f"}
            ],

        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_risk_level_chart(probability):
    """Cria gráfico de barras para nível de risco"""
    risk_level = "ALTO" if probability > 0.66 else "MÉDIO" if probability > 0.33 else "BAIXO"
    color = "#f44336" if probability > 0.66 else "#ff9800" if probability > 0.33 else "#4caf50"
    
    return risk_level, color

def get_retention_strategies(input_data, probability):
    """Gera estratégias de retenção baseadas nos dados do cliente"""
    strategies = []
    
    if input_data['Contract'].values[0] == 'Month-to-month':
        strategies.append("📝 Oferecer desconto para contrato anual")
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    missing_services = [s for s in services if input_data[s].values[0] == 'No']
    
    if len(missing_services) >= 2:
        strategies.append(f"🛡️ Oferecer pacote de {len(missing_services)} serviços adicionais")
    
    if input_data['PaymentMethod'].values[0] == 'Electronic check':
        strategies.append("💳 Incentivar débito automático")
    
    if input_data['tenure'].values[0] < 12:
        strategies.append("🎁 Programa de fidelidade para novos clientes")
    
    return strategies

def create_comparison_chart(LR_prob, RF_prob, XGB_prob, ensemble_prob):
    """Cria gráfico de comparação entre modelos"""
    models = ['LR', 'RF', 'XGB', 'Ensemble']
    probabilities = [LR_prob * 100, RF_prob * 100, XGB_prob * 100, ensemble_prob * 100]
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=probabilities, text=[f'{p:.1f}%' for p in probabilities], textposition='auto')
    ])
    
    fig.update_layout(
        title='Comparação entre Modelos',
        xaxis_title='Modelo',
        yaxis_title='Probabilidade de Churn (%)',
        yaxis_range=[0, 100],
        height=350
    )
    
    return fig

# ==================== CARREGAMENTO INICIAL ====================
with st.spinner('Carregando...'):
    LR_pipeline, RF_pipeline, XGB_pipeline, eclf1 = load_models()
    df = load_data()

if df is None or eclf1 is None:
    st.stop()

# ==================== HEADER ====================
st.title("🎯 Previsão de Churn de Clientes")
st.caption("Sistema de Machine Learning para identificação de clientes em risco")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configurações")
    use_example = st.checkbox("Carregar exemplo", value=False)
    
    if use_example:
        example_customer = df.sample(1).iloc[0]
        st.success("Cliente carregado!")
    if st.button('Novo Exemplo'):
        example_customer = df.sample(1).iloc[0]
        st.success("Novo cliente carregado!")
    
    st.divider()
    st.info(f"""
    **Modelos:** 4  
    **Recall:** 83%  
    **Atualização:** {datetime.now().strftime('%d/%m/%Y')}
    """)

# ==================== ABAS PRINCIPAIS ====================
tab1, tab2, tab3 = st.tabs([" Análise", " Modelos", " Estratégias"])

with tab1:
    st.subheader("Dados do Cliente")
    
    # ==================== FORMULÁRIO DE ENTRADA ====================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Perfil**")
        gender = st.selectbox("Gênero", df['gender'].unique(), 
                             index=int(example_customer['gender'] == 'Female') if use_example else 0)
        senior = st.selectbox("Idoso", df['SeniorCitizen'].unique(),
                             index=int(example_customer['SeniorCitizen']) if use_example else 0)
        partner = st.selectbox("Parceiro", df['Partner'].unique(),
                              index=int(example_customer['Partner'] == 'Yes') if use_example else 0)
        dependents = st.selectbox("Dependentes", df['Dependents'].unique(),
                                 index=int(example_customer['Dependents'] == 'Yes') if use_example else 0)
    
    with col2:
        st.markdown("**Serviços**")
        phone_service = st.selectbox("Telefone", df['PhoneService'].unique(),
                                    index=int(example_customer['PhoneService'] == 'Yes') if use_example else 0)
        multiple_lines = st.selectbox("Múltiplas Linhas", df['MultipleLines'].unique(),
                                     index=list(df['MultipleLines'].unique()).index(example_customer['MultipleLines']) if use_example else 0)
        internet_service = st.selectbox("Internet", df['InternetService'].unique(),
                                       index=list(df['InternetService'].unique()).index(example_customer['InternetService']) if use_example else 0)
        streaming_tv = st.selectbox("TV", df['StreamingTV'].unique(),
                                   index=list(df['StreamingTV'].unique()).index(example_customer['StreamingTV']) if use_example else 0)
        streaming_movies = st.selectbox("Filmes", df['StreamingMovies'].unique(),
                                       index=list(df['StreamingMovies'].unique()).index(example_customer['StreamingMovies']) if use_example else 0)
    
    with col3:
        st.markdown("**Segurança**")
        online_security = st.selectbox("Segurança Online", df['OnlineSecurity'].unique(),
                                      index=list(df['OnlineSecurity'].unique()).index(example_customer['OnlineSecurity']) if use_example else 0)
        online_backup = st.selectbox("Backup", df['OnlineBackup'].unique(),
                                    index=list(df['OnlineBackup'].unique()).index(example_customer['OnlineBackup']) if use_example else 0)
        device_protection = st.selectbox("Proteção", df['DeviceProtection'].unique(),
                                        index=list(df['DeviceProtection'].unique()).index(example_customer['DeviceProtection']) if use_example else 0)
        tech_support = st.selectbox("Suporte", df['TechSupport'].unique(),
                                   index=list(df['TechSupport'].unique()).index(example_customer['TechSupport']) if use_example else 0)
    
    with col4:
        st.markdown("**Financeiro**")
        contract = st.selectbox("Contrato", df['Contract'].unique(),
                               index=list(df['Contract'].unique()).index(example_customer['Contract']) if use_example else 0)
        paperless = st.selectbox("Fatura Digital", df['PaperlessBilling'].unique(),
                                index=int(example_customer['PaperlessBilling'] == 'Yes') if use_example else 0)
        payment = st.selectbox("Pagamento", df['PaymentMethod'].unique(),
                              index=list(df['PaymentMethod'].unique()).index(example_customer['PaymentMethod']) if use_example else 0)
        tenure = st.number_input("Meses", min_value=0, max_value=100, 
                                value=int(example_customer['tenure']) if use_example else 1)
        monthly_charges = st.number_input("Cobrança Mensal ($)", min_value=0.0, max_value=2000.0, 
                                         value=float(example_customer['MonthlyCharges']) if use_example else 50.0,
                                         step=5.0)
        total_charges = st.number_input("Total ($)", min_value=0.0, max_value=100000.0, 
                                       value=float(example_customer['TotalCharges']) if use_example and pd.notna(example_customer['TotalCharges']) else monthly_charges * tenure,
                                       step=10.0)
    
    # ==================== PREDIÇÃO ====================
    st.divider()
    
    # Criar DataFrame de input
    input_data = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }])
    
    # Fazer predições
    try:
        prediction = eclf1.predict(input_data)[0]
        prediction_proba = eclf1.predict_proba(input_data)[0, 1]
        lr_proba = LR_pipeline.predict_proba(input_data)[0, 1] if LR_pipeline else 0
        rf_proba = RF_pipeline.predict_proba(input_data)[0, 1] if RF_pipeline else 0
        xgb_proba = XGB_pipeline.predict_proba(input_data)[0, 1] if XGB_pipeline else 0
        
        # ==================== RESULTADOS ====================
        st.subheader("Resultado")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Probabilidade de Churn", f"{prediction_proba*100:.1f}%",
                     delta=f"{(prediction_proba - 0.5)*100:.1f}% vs média", delta_color="inverse")
        
        with col2:
            risk_level, risk_color = create_risk_level_chart(prediction_proba)
            st.metric("Nível de Risco", risk_level)
        
        with col3:
            customer_value = monthly_charges * 12
            st.metric("Valor Anual", f"${customer_value:.2f}")
        
        # Gauge chart
        gauge_fig = create_gauge_chart(prediction_proba, "Risco de Churn")
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Alerta
        if prediction == 1:
            st.error(f"⚠️ Cliente com risco {risk_level} de churn - Ação recomendada")
        else:
            st.success(f"✅ Cliente com risco {risk_level} de churn")
        
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")

with tab2:
    st.subheader("Comparação de Modelos")
    
    try:
        # Gráfico
        comparison_fig = create_comparison_chart(lr_proba, rf_proba, xgb_proba, prediction_proba)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Tabela
        comparison_df = pd.DataFrame({
            'Modelo': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble'],
            'Probabilidade': [f'{lr_proba*100:.1f}%', f'{rf_proba*100:.1f}%', 
                             f'{xgb_proba*100:.1f}%', f'{prediction_proba*100:.1f}%'],
            'Predição': ['Churn' if p > 0.5 else 'Não Churn' 
                        for p in [lr_proba, rf_proba, xgb_proba, prediction_proba]],
            'Recall': ['79%', '70%', '83%', '83%']
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Erro: {str(e)}")

with tab3:
    st.subheader("Estratégias de Retenção")
    
    strategies = get_retention_strategies(input_data, prediction_proba)
    
    if strategies:
        for strategy in strategies:
            st.info(strategy)
    else:
        st.success("✅ Cliente estável - manter relacionamento atual")

# ==================== FOOTER ====================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Recall", "83%")
with col2:
    st.metric("Modelos", "4")
with col3:
    st.caption("Desenvolvido por Rafael Luckner")