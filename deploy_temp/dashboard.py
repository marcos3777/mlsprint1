import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Mottu - Previsão de Demanda",
    page_icon="🏍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8b4513;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b4423;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f5ebe0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #a0826d;
    }
    .prediction-box {
        background: linear-gradient(135deg, #c19a6b 0%, #8b7355 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #8b4513;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #654321;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models_dir = Path('models')
    
    try:
        model_saida = joblib.load(models_dir / 'model_saida.pkl')
        model_volta = joblib.load(models_dir / 'model_volta.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')
        features = joblib.load(models_dir / 'features.pkl')
        metricas = joblib.load(models_dir / 'metricas.pkl')
        
        return model_saida, model_volta, scaler, features, metricas
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        st.info("Execute o notebook ml.ipynb primeiro para treinar e salvar os modelos!")
        st.stop()

model_saida, model_volta, scaler, features, metricas = load_models()

st.markdown('<p class="main-header">Sistema de Previsão de Demanda - Mottu</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Previsão de saídas e retornos de motocicletas</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Previsão", "Métricas do Modelo", "Sobre"])

with tab1:
    st.markdown("### Fazer Previsão")
    
    st.sidebar.markdown("## Parâmetros de Entrada")
    st.sidebar.markdown("---")
    
    col_sidebar1, col_sidebar2 = st.sidebar.columns(2)
    
    with col_sidebar1:
        dia_semana = st.selectbox(
            "Dia da Semana",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"][x]
        )
        
        motos_em_uso = st.number_input(
            "Motos em Uso",
            min_value=0,
            max_value=100,
            value=20,
            step=1
        )
        
        motos_disponiveis = st.number_input(
            "Motos Disponíveis",
            min_value=0,
            max_value=100,
            value=80,
            step=1
        )
        
        total_motos = st.number_input(
            "Total de Motos",
            min_value=1,
            max_value=200,
            value=100,
            step=1
        )
    
    with col_sidebar2:
        tipo_dia = st.selectbox(
            "Tipo de Dia",
            options=[0, 1],
            format_func=lambda x: "Dia Útil" if x == 0 else "Fim de Semana"
        )
        
        choveu = st.selectbox(
            "Condição Climática",
            options=[0, 1],
            format_func=lambda x: "Sem Chuva" if x == 0 else "Com Chuva"
        )
        
        feriado = st.selectbox(
            "Feriado",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim"
        )
        
        saldo_dia = st.number_input(
            "Saldo do Dia Anterior",
            min_value=-50,
            max_value=50,
            value=0,
            step=1
        )
    
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("Fazer Previsão", use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            galpao = 0
            taxa_ocupacao = motos_em_uso / total_motos
            choveu_fds = choveu * tipo_dia
            feriado_fds = feriado * tipo_dia
            
            input_data = pd.DataFrame([[
                galpao, dia_semana, motos_em_uso, motos_disponiveis,
                choveu, total_motos, feriado, tipo_dia, saldo_dia,
                taxa_ocupacao, choveu_fds, feriado_fds
            ]], columns=features)
            
            input_scaled = scaler.transform(input_data)
            
            pred_saida = model_saida.predict(input_scaled)[0]
            pred_volta = model_volta.predict(input_scaled)[0]
            saldo_previsto = pred_saida - pred_volta
            
            st.session_state['ultima_predicao'] = {
                'saida': pred_saida,
                'volta': pred_volta,
                'saldo': saldo_previsto,
                'timestamp': datetime.now()
            }
    
    if 'ultima_predicao' in st.session_state:
        pred = st.session_state['ultima_predicao']
        
        with col1:
            st.markdown("### Resultados da Previsão")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #c19a6b 0%, #a0826d 100%); 
                            padding: 1.5rem; border-radius: 1rem; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Saídas</h3>
                    <h1 style='color: white; margin: 0.5rem 0;'>{:.0f}</h1>
                    <p style='color: rgba(255,255,255,0.8); margin: 0;'>motos previstas</p>
                </div>
                """.format(pred['saida']), unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #d4a574 0%, #b8956a 100%); 
                            padding: 1.5rem; border-radius: 1rem; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Retornos</h3>
                    <h1 style='color: white; margin: 0.5rem 0;'>{:.0f}</h1>
                    <p style='color: rgba(255,255,255,0.8); margin: 0;'>motos previstas</p>
                </div>
                """.format(pred['volta']), unsafe_allow_html=True)
            
            with result_col3:
                saldo_color = "#8b7355" if pred['saldo'] >= 0 else "#a0522d"
                st.markdown("""
                <div style='background: linear-gradient(135deg, {} 0%, {} 100%); 
                            padding: 1.5rem; border-radius: 1rem; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>Saldo</h3>
                    <h1 style='color: white; margin: 0.5rem 0;'>{:+.0f}</h1>
                    <p style='color: rgba(255,255,255,0.8); margin: 0;'>diferença</p>
                </div>
                """.format(saldo_color, saldo_color, pred['saldo']), unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("#### Visualização Comparativa")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Saídas', 'Retornos'],
                y=[pred['saida'], pred['volta']],
                marker_color=['#c19a6b', '#d4a574'],
                text=[f"{pred['saida']:.1f}", f"{pred['volta']:.1f}"],
                textposition='outside',
                textfont=dict(size=14, color='#654321')
            ))
            
            fig.update_layout(
                title="Comparação: Saídas vs Retornos",
                yaxis_title="Número de Motos",
                template="plotly_white",
                height=400,
                showlegend=False,
                paper_bgcolor='#f5ebe0',
                plot_bgcolor='#f5ebe0'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Interpretação")
            if pred['saldo'] > 5:
                st.info(f"**Saldo Positivo**: Espera-se que {abs(pred['saldo']):.0f} motos a mais saiam do que retornem.")
            elif pred['saldo'] < -5:
                st.warning(f"**Saldo Negativo**: Espera-se que {abs(pred['saldo']):.0f} motos a mais retornem do que saiam.")
            else:
                st.success("**Saldo Equilibrado**: Saídas e retornos previstos estão equilibrados.")
    
    with col2:
        st.markdown("### Resumo dos Parâmetros")
        
        dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        
        st.markdown(f"""
        <div class='metric-card'>
            <strong>Dia:</strong> {dias[dia_semana]}<br>
            <strong>Tipo:</strong> {"Dia Útil" if tipo_dia == 0 else "Fim de Semana"}<br>
            <strong>Clima:</strong> {"Sem Chuva" if choveu == 0 else "Com Chuva"}<br>
            <strong>Feriado:</strong> {"Não" if feriado == 0 else "Sim"}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card'>
            <strong>Em Uso:</strong> {motos_em_uso}<br>
            <strong>Disponíveis:</strong> {motos_disponiveis}<br>
            <strong>Total:</strong> {total_motos}<br>
            <strong>Saldo Anterior:</strong> {saldo_dia:+d}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if motos_em_uso + motos_disponiveis > 0:
            fig_pizza = go.Figure(data=[go.Pie(
                labels=['Em Uso', 'Disponíveis'],
                values=[motos_em_uso, motos_disponiveis],
                marker=dict(colors=['#8b4513', '#c19a6b']),
                hole=0.4
            )])
            
            fig_pizza.update_layout(
                title="Status Atual da Frota",
                template="plotly_white",
                height=300,
                showlegend=True,
                paper_bgcolor='#f5ebe0'
            )
            
            st.plotly_chart(fig_pizza, use_container_width=True)

with tab2:
    st.markdown("### Performance dos Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Modelo: Motos que Saíram")
        
        metrics_saida = metricas['model_saida']
        
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("R² Score", f"{metrics_saida['r2']:.4f}")
        mcol2.metric("MAE", f"{metrics_saida['mae']:.2f}")
        mcol3.metric("RMSE", f"{metrics_saida['rmse']:.2f}")
        
        st.markdown("""
        **Interpretação:**
        - **R² Score**: Indica o quanto o modelo explica a variação dos dados (quanto mais próximo de 1, melhor)
        - **MAE**: Erro médio absoluto das predições
        - **RMSE**: Raiz do erro quadrático médio (penaliza erros maiores)
        """)
    
    with col2:
        st.markdown("#### Modelo: Motos que Voltaram")
        
        metrics_volta = metricas['model_volta']
        
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("R² Score", f"{metrics_volta['r2']:.4f}")
        mcol2.metric("MAE", f"{metrics_volta['mae']:.2f}")
        mcol3.metric("RMSE", f"{metrics_volta['rmse']:.2f}")
        
        st.markdown("""
        **Interpretação:**
        - **R² Score**: Indica o quanto o modelo explica a variação dos dados (quanto mais próximo de 1, melhor)
        - **MAE**: Erro médio absoluto das predições
        - **RMSE**: Raiz do erro quadrático médio (penaliza erros maiores)
        """)
    
    st.markdown("---")
    
    st.markdown("#### Comparação de Métricas")
    
    fig_metrics = go.Figure()
    
    metricas_nomes = ['R² Score', 'MAE', 'RMSE']
    saida_vals = [metrics_saida['r2'], metrics_saida['mae'], metrics_saida['rmse']]
    volta_vals = [metrics_volta['r2'], metrics_volta['mae'], metrics_volta['rmse']]
    
    fig_metrics.add_trace(go.Bar(
        name='Saídas',
        x=metricas_nomes,
        y=saida_vals,
        marker_color='#8b4513'
    ))
    
    fig_metrics.add_trace(go.Bar(
        name='Retornos',
        x=metricas_nomes,
        y=volta_vals,
        marker_color='#c19a6b'
    ))
    
    fig_metrics.update_layout(
        title="Métricas dos Modelos",
        yaxis_title="Valor",
        barmode='group',
        template="plotly_white",
        height=400,
        paper_bgcolor='#f5ebe0',
        plot_bgcolor='#f5ebe0'
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Sobre os Modelos")
    st.markdown("""
    **Configuração dos Modelos:**
    - Algoritmo: Random Forest Regressor
    - Número de árvores: 300
    - Profundidade máxima: 10
    - Features: 9 variáveis
    - Divisão treino/teste: 70% / 30%
    - Normalização: MinMaxScaler
    """)

with tab3:
    st.markdown("### Sobre o Sistema")
    
    st.markdown("""
    #### Objetivo
    
    Sistema de previsão de demanda de motocicletas para galpões de delivery utilizando Machine Learning.
    
    #### Variáveis do Modelo
    
    - Dia da Semana (0-6)
    - Tipo de Dia (Útil ou Fim de Semana)
    - Condições Climáticas
    - Feriado
    - Motos em Uso
    - Motos Disponíveis
    - Total de Motos
    - Saldo do Dia Anterior
    - Galpão
    
    #### Tecnologias
    
    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    - RandomForest
    
    ---
    
    #### Equipe
    
    - Marcos Vinicius Pereira de Oliveira - RM 557252
    - Ruan Lima Silva - RM 558775
    - Richardy Borges Santana - RM 557883
    
    ---
    
    **Versão:** 1.0.0
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; font-size: 0.8rem; color: #666;'>
    <p><strong>Mottu ML</strong></p>
    <p>v1.0.0</p>
</div>
""", unsafe_allow_html=True)

