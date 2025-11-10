# 🏍️ Sistema de Previsão de Demanda - Mottu

Sistema de Machine Learning para previsão de demanda de motocicletas em galpões de delivery.

## 👥 Equipe

- **Marcos Vinicius Pereira de Oliveira** - RM 557252
- **Ruan Lima Silva** - RM 558775
- **Richardy Borges Santana** - RM 557883

## 🚀 Acesso ao Sistema

- **Dashboard Interativo:** http://162.240.161.80:8501/
- **API (Documentação):** http://162.240.161.80:8502/docs#

## 🎯 Objetivo

Prever a quantidade de motocicletas que:
1. Sairão do galpão
2. Retornarão ao galpão

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Machine Learning:** scikit-learn (RandomForest Regressor)
- **Análise de Dados:** pandas, numpy
- **Visualização:** matplotlib, seaborn
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Persistência:** joblib

## 📊 Resultados

### Modelo - Motos que Saíram
- **R²:** 0.5561
- **MAE:** 3.14
- **RMSE:** 4.70

### Modelo - Motos que Voltaram
- **R²:** 0.3779
- **MAE:** 3.09
- **RMSE:** 4.64

### Features Utilizadas (12)
- Básicas: galpão, dia da semana, motos em uso, disponíveis, chuva, total, feriado, tipo de dia, saldo
- **Derivadas:** taxa de ocupação, chuva em FDS, feriado em FDS

## 📁 Estrutura do Projeto

```
Sprint3/
├── ml-improved.ipynb          # Notebook completo com análise e treinamento
├── dados_mottu_corrigido.csv  # Dataset
├── models/                    # Modelos treinados (.pkl)
├── deploy_temp/
│   ├── app.py                # API FastAPI
│   └── dashboard.py          # Dashboard Streamlit
└── requirements.txt          # Dependências
```

## 🔧 Instalação Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar API
cd deploy_temp
uvicorn app:app --reload --port 8502

# Rodar Dashboard (outro terminal)
streamlit run dashboard.py --server.port 8501
```

## 📈 Como Usar

1. **Via Dashboard:** Acesse o link do dashboard e preencha os campos
2. **Via API:** Use a documentação interativa para fazer requisições POST

## 📝 Detalhes

Para análise exploratória completa, processo de feature engineering e métricas detalhadas, consulte o notebook `ml-improved.ipynb`.

