from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pandas as pd
import joblib
from pathlib import Path

# Paths
DATA_PATH = Path("dados_mottu_corrigido.csv")
MODELS_DIR = Path("models")

tags_metadata = [
    {
        "name": "health",
        "description": "Endpoints para verificar o status da API e obter informações sobre os modelos carregados."
    },
    {
        "name": "prediction",
        "description": "Endpoint principal para realizar predições de saídas e retornos de motocicletas."
    },
]

app = FastAPI(
    title="Mottu ML - API de Previsão de Demanda",
    description="""
### Funcionalidades
- Previsão de quantas motos sairão do galpão
- Previsão de quantas motos retornarão ao galpão
- Cálculo do saldo previsto (diferença entre saídas e retornos)
- Métricas de performance dos modelos (R², MAE, RMSE)

### Equipe
- Marcos Vinicius Pereira de Oliveira - RM 557252
- Ruan Lima Silva - RM 558775
- Richardy Borges Santana - RM 557883
    """,
    version="1.0.0",
    contact={
        "name": "Equipe Mottu ML",
        "email": "rm557252@fiap.com.br"
    },
    openapi_url="/openapi.json",  
    docs_url="/docs",             
    redoc_url="/redoc",           
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "tryItOutEnabled": True,
        "defaultModelsExpandDepth": 2,
        "defaultModelExpandDepth": 2
    },
)

FEATURES = [
    "galpao","dia_semana","motos_em_uso","motos_disponiveis",
    "choveu","total_motos","feriado","tipo_dia","saldo_dia",
    "taxa_ocupacao","choveu_fds","feriado_fds"
]

galpao_map = {}
tipo_dia_map = {"UTIL": 0, "FIM_DE_SEMANA": 1}

class InputPayload(BaseModel):
    """Modelo de entrada para previsão de demanda de motocicletas"""
   
    galpao: Optional[int] = Field(
        None,
        description="Código numérico do galpão (0 para BUTANTAN)",
        example=0
    )
    galpao_str: Optional[str] = Field(
        None,
        description="Nome do galpão em texto (ex: 'BUTANTAN')",
        example="BUTANTAN"
    )

    dia_semana: int = Field(
        ...,
        ge=0,
        le=6,
        description="Dia da semana: 0=Segunda, 1=Terça, 2=Quarta, 3=Quinta, 4=Sexta, 5=Sábado, 6=Domingo",
        example=6
    )
    
    motos_em_uso: float = Field(
        ...,
        ge=0,
        description="Quantidade de motos atualmente em uso/operação",
        example=18
    )
    
    motos_disponiveis: float = Field(
        ...,
        ge=0,
        description="Quantidade de motos disponíveis no galpão",
        example=82
    )
    
    choveu: int = Field(
        ...,
        ge=0,
        le=1,
        description="Condição climática: 0=Sem chuva, 1=Com chuva",
        example=0
    )
    
    total_motos: float = Field(
        ...,
        ge=1,
        description="Total de motos na frota",
        example=100
    )
    
    feriado: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indica se é feriado: 0=Não, 1=Sim",
        example=1
    )

    tipo_dia: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Tipo de dia (código): 0=Dia útil, 1=Fim de semana",
        example=1
    )
    
    tipo_dia_str: Optional[str] = Field(
        None,
        description="Tipo de dia (texto): 'UTIL' ou 'FIM_DE_SEMANA'",
        example="FIM_DE_SEMANA"
    )

    saldo_dia: float = Field(
        ...,
        description="Saldo do dia anterior (diferença entre saídas e retornos)",
        example=7
    )
   
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "galpao_str": "BUTANTAN",
                    "dia_semana": 6,
                    "motos_em_uso": 18,
                    "motos_disponiveis": 82,
                    "choveu": 0,
                    "total_motos": 100,
                    "feriado": 1,
                    "tipo_dia_str": "FIM_DE_SEMANA",
                    "saldo_dia": 7
                },
                {
                    "galpao": 0,
                    "dia_semana": 0,
                    "motos_em_uso": 20,
                    "motos_disponiveis": 80,
                    "choveu": 0,
                    "total_motos": 100,
                    "feriado": 0,
                    "tipo_dia": 0,
                    "saldo_dia": 0
                },
                {
                    "galpao_str": "BUTANTAN",
                    "dia_semana": 5,
                    "motos_em_uso": 25,
                    "motos_disponiveis": 75,
                    "choveu": 1,
                    "total_motos": 100,
                    "feriado": 0,
                    "tipo_dia_str": "FIM_DE_SEMANA",
                    "saldo_dia": -3
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Modelo de resposta da previsão"""
    
    motos_que_sairam: float = Field(
        ...,
        description="Quantidade prevista de motos que sairão do galpão"
    )
    motos_que_voltaram: float = Field(
        ...,
        description="Quantidade prevista de motos que retornarão ao galpão"
    )
    saldo_previsto: float = Field(
        ...,
        description="Saldo previsto (saídas - retornos). Positivo = mais saídas, Negativo = mais retornos"
    )
    galpao_map: Dict[str, int] = Field(
        ...,
        description="Mapeamento de nomes de galpões para códigos numéricos"
    )
    tipo_dia_map: Dict[str, int] = Field(
        ...,
        description="Mapeamento de tipos de dia para códigos numéricos"
    )
    metricas_modelo: Optional[Dict[str, Any]] = Field(
        None,
        description="Métricas de performance dos modelos (R², MAE, RMSE)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "motos_que_sairam": 45.23,
                "motos_que_voltaram": 38.15,
                "saldo_previsto": 7.08,
                "galpao_map": {"BUTANTAN": 0},
                "tipo_dia_map": {"UTIL": 0, "FIM_DE_SEMANA": 1},
                "metricas_modelo": {
                    "saida": {"r2": 0.8532, "mae": 3.45},
                    "volta": {"r2": 0.8421, "mae": 3.67}
                }
            }
        }
    }


class HealthResponse(BaseModel):
    """Modelo de resposta do health check"""
    
    status: str = Field(..., description="Status da API")
    models_loaded: bool = Field(..., description="Indica se os modelos estão carregados")
    galpao_map: Dict[str, int] = Field(..., description="Mapeamento de galpões disponíveis")
    tipo_dia_map: Dict[str, int] = Field(..., description="Mapeamento de tipos de dia")
    metricas: Optional[Dict[str, Any]] = Field(None, description="Métricas dos modelos")

# Variáveis globais para os modelos
scaler = None
model_saida = None
model_volta = None
metricas = None

@app.on_event("startup")
def _init():
    """Carrega os modelos treinados do disco ao iniciar a API"""
    global galpao_map, scaler, model_saida, model_volta, metricas
    
    try:
        # Carregar modelos salvos
        print("Carregando modelos do disco...")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        model_saida = joblib.load(MODELS_DIR / "model_saida.pkl")
        model_volta = joblib.load(MODELS_DIR / "model_volta.pkl")
        metricas = joblib.load(MODELS_DIR / "metricas.pkl")
        print("Modelos carregados com sucesso!")
        
        # Carregar dados apenas para gerar os mapas de categoria
        df = pd.read_csv(DATA_PATH)
        
        if df["galpao"].dtype == object:
            df["galpao_norm"] = df["galpao"].astype(str).str.upper().str.strip()
            cats = sorted(df["galpao_norm"].unique())
            galpao_map = {name: i for i, name in enumerate(cats)}
        
        print(f"API inicializada! Galpões disponíveis: {list(galpao_map.keys())}")
        
    except FileNotFoundError as e:
        print(f"ERRO: Modelos não encontrados em {MODELS_DIR}")
        print("Execute o notebook ml.ipynb primeiro para treinar e salvar os modelos!")
        raise RuntimeError(
            "Modelos não encontrados. Execute o notebook ml.ipynb para treinar os modelos."
        ) from e
    except Exception as e:
        print(f"ERRO ao carregar modelos: {e}")
        raise

@app.get(
    "/health",
    tags=["health"],
    response_model=HealthResponse,
    summary="Verificar status da API",
    description="""
    Retorna o status atual da API e informações sobre os modelos carregados.
    
    **Informações retornadas:**
    - Status da API (ok/error)
    - Se os modelos estão carregados
    - Mapeamento de galpões disponíveis
    - Mapeamento de tipos de dia
    - Métricas de performance dos modelos (R², MAE, RMSE)
    """
)
def health():
    """Endpoint de health check"""
    if model_saida is None or model_volta is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    return {
        "status": "ok",
        "models_loaded": True,
        "galpao_map": galpao_map,
        "tipo_dia_map": tipo_dia_map,
        "metricas": metricas if metricas else "N/A"
    }

def _normalize_input(inp: InputPayload) -> pd.DataFrame:
    
    if inp.galpao_str is not None:
        key = inp.galpao_str.upper().strip()
        g = galpao_map.get(key, 0)  
    else:
        g = 0 if inp.galpao is None else int(inp.galpao)

  
    if inp.tipo_dia_str is not None:
        key = inp.tipo_dia_str.upper().strip()
        td = tipo_dia_map.get(key, 0)
    else:
        td = 0 if inp.tipo_dia is None else int(inp.tipo_dia)

    row = {
        "galpao": g,
        "dia_semana": inp.dia_semana,
        "motos_em_uso": inp.motos_em_uso,
        "motos_disponiveis": inp.motos_disponiveis,
        "choveu": inp.choveu,
        "total_motos": inp.total_motos,
        "feriado": inp.feriado,
        "tipo_dia": td,
        "saldo_dia": inp.saldo_dia,
        "taxa_ocupacao": inp.motos_em_uso / inp.total_motos,
        "choveu_fds": inp.choveu * td,
        "feriado_fds": inp.feriado * td
    }
    return pd.DataFrame([row])[FEATURES]

@app.post(
    "/predict",
    tags=["prediction"],
    response_model=PredictionResponse,
    summary="Realizar previsão de demanda",
    description="""
    Realiza a previsão de quantas motos sairão e retornarão ao galpão com base nos parâmetros fornecidos.
    
    **Como usar:**
    
    Você pode fornecer os dados de duas formas:
    
    1. **Usando códigos numéricos:**
       - `galpao`: 0 (BUTANTAN)
       - `tipo_dia`: 0 (dia útil) ou 1 (fim de semana)
    
    2. **Usando strings (mais intuitivo):**
       - `galpao_str`: "BUTANTAN"
       - `tipo_dia_str`: "UTIL" ou "FIM_DE_SEMANA"
    
    **Parâmetros obrigatórios:**
    - `dia_semana`: 0 a 6 (0=Segunda, 6=Domingo)
    - `motos_em_uso`: Quantidade de motos em operação
    - `motos_disponiveis`: Quantidade de motos paradas no galpão
    - `choveu`: 0 (sem chuva) ou 1 (com chuva)
    - `total_motos`: Total de motos na frota
    - `feriado`: 0 (não) ou 1 (sim)
    - `saldo_dia`: Saldo do dia anterior (pode ser negativo)
    
    **Resposta:**
    - Quantidade prevista de saídas
    - Quantidade prevista de retornos
    - Saldo previsto (positivo = mais saídas, negativo = mais retornos)
    - Métricas de acurácia dos modelos
    
    **Exemplo de uso:**
    
    ```json
    {
      "galpao_str": "BUTANTAN",
      "dia_semana": 6,
      "motos_em_uso": 18,
      "motos_disponiveis": 82,
      "choveu": 0,
      "total_motos": 100,
      "feriado": 1,
      "tipo_dia_str": "FIM_DE_SEMANA",
      "saldo_dia": 7
    }
    ```
    """,
    responses={
        200: {
            "description": "Previsão realizada com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "motos_que_sairam": 45.23,
                        "motos_que_voltaram": 38.15,
                        "saldo_previsto": 7.08,
                        "galpao_map": {"BUTANTAN": 0},
                        "tipo_dia_map": {"UTIL": 0, "FIM_DE_SEMANA": 1},
                        "metricas_modelo": {
                            "saida": {"r2": 0.8532, "mae": 3.45},
                            "volta": {"r2": 0.8421, "mae": 3.67}
                        }
                    }
                }
            }
        },
        422: {
            "description": "Erro de validação - parâmetros inválidos ou faltando"
        },
        500: {
            "description": "Erro interno durante a previsão"
        },
        503: {
            "description": "Modelos não carregados - execute o notebook ml.ipynb primeiro"
        }
    }
)
def predict(inp: InputPayload):
    """Endpoint principal de previsão"""
    if model_saida is None or model_volta is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelos não carregados. Execute o notebook ml.ipynb primeiro."
        )
    
    try:
        X = _normalize_input(inp)
        Xs = scaler.transform(X)
        saidas = float(model_saida.predict(Xs)[0])
        retornos = float(model_volta.predict(Xs)[0])
        saldo = saidas - retornos
        
        return {
            "motos_que_sairam": round(saidas, 2),
            "motos_que_voltaram": round(retornos, 2),
            "saldo_previsto": round(saldo, 2),
            "galpao_map": galpao_map,
            "tipo_dia_map": tipo_dia_map,
            "metricas_modelo": {
                "saida": {
                    "r2": round(metricas["model_saida"]["r2"], 4),
                    "mae": round(metricas["model_saida"]["mae"], 2)
                },
                "volta": {
                    "r2": round(metricas["model_volta"]["r2"], 4),
                    "mae": round(metricas["model_volta"]["mae"], 2)
                }
            } if metricas else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")