# Documento Técnico - Previsão de Demanda de Motocicletas

## 1. INTRODUÇÃO

### 1.1 Contexto do Problema
Este projeto aborda um problema real de otimização operacional na empresa Mottu, que gerencia uma frota de motocicletas para delivery. O desafio principal é prever a demanda diária de veículos considerando múltiplos fatores.

### 1.2 Problema de Pesquisa
Como prever com precisão o número de motocicletas que sairão e retornarão a um galpão, considerando variáveis como:
- Condições climáticas
- Tipo de dia (útil/fim de semana)
- Histórico operacional
- Feriados

### 1.3 Objetivos
- **Objetivo Geral**: Desenvolver um modelo de machine learning para previsão de demanda
- **Objetivos Específicos**:
  - Implementar redes neurais para regressão
  - Avaliar a performance dos modelos
  - Comparar predições com cenários reais

## 2. METODOLOGIA

### 2.1 Ferramentas Utilizadas

#### 2.1.1 Python
Linguagem escolhida por sua ampla adoção em machine learning e disponibilidade de bibliotecas especializadas.

#### 2.1.2 TensorFlow/Keras
Framework para construção das redes neurais, oferecendo:
- API simplificada para criação de modelos
- Algoritmos de otimização integrados
- Facilidade de implementação

#### 2.1.3 Scikit-learn
Utilizado para:
- Normalização dos dados (MinMaxScaler)
- Divisão treino/teste
- Métricas de avaliação

#### 2.1.4 Pandas e NumPy
- **Pandas**: Manipulação do dataset CSV
- **NumPy**: Operações matemáticas e arrays

#### 2.1.5 Matplotlib
Visualização dos resultados de treinamento e gráficos de perda.

### 2.2 Arquitetura do Modelo

#### 2.2.1 Escolha das Redes Neurais
Optamos por redes neurais densas (fully connected) por serem adequadas para:
- Problemas de regressão com dados tabulares
- Captura de relações não-lineares entre variáveis
- Flexibilidade na arquitetura

#### 2.2.2 Estrutura da Rede
```
Entrada (9 variáveis) → Camada Densa (16 neurônios) → Camada Densa (8 neurônios) → Saída (1 neurônio)
```

**Características**:
- **Duas redes independentes**: Uma para saídas, outra para retornos
- **Função de ativação**: ReLU nas camadas ocultas, Linear na saída
- **Redução gradual**: 16→8→1 neurônios

### 2.3 Configurações de Treinamento

#### 2.3.1 Otimizador Adam
Escolhido por sua eficiência e capacidade de adaptação automática do learning rate.

#### 2.3.2 Função de Perda
MSE (Mean Squared Error) - padrão para problemas de regressão.

#### 2.3.3 Pré-processamento
- **Normalização**: MinMaxScaler para escala [0,1]
- **Codificação**: Variáveis categóricas convertidas para numéricas
- **Divisão**: 70% treino, 30% teste (seed=42)

## 3. DATASET E VARIÁVEIS

### 3.1 Descrição do Dataset
O dataset contém 95 registros do galpão Butantã com as seguintes variáveis:

**Variáveis de Entrada**:
- `galpao`: Localização (BUTANTAN)
- `dia_semana`: Dia da semana (0-6)
- `motos_em_uso`: Motos atualmente em operação
- `motos_disponiveis`: Motos disponíveis no galpão
- `choveu`: Indicador de chuva (0/1)
- `total_motos`: Total da frota (100)
- `feriado`: Indicador de feriado (0/1)
- `tipo_dia`: Útil (0) ou fim de semana (1)
- `saldo_dia`: Diferença entre saídas e retornos

**Variáveis de Saída**:
- `motos_que_sairam`: Número de motos que saíram
- `motos_que_voltaram`: Número de motos que retornaram

### 3.2 Características dos Dados
- **Período**: Dados históricos de operação
- **Granularidade**: Diária
- **Completude**: Dataset sem valores faltantes
- **Balanceamento**: Distribuição equilibrada entre dias úteis e fins de semana

## 4. RESULTADOS E ANÁLISE

### 4.1 Performance dos Modelos

#### 4.1.1 Métricas Obtidas
- **Modelo de Saídas**: MSE = 18.19
- **Modelo de Retornos**: MSE = 19.59

#### 4.1.2 Interpretação
- Ambos os modelos apresentam performance similar
- Erro médio de aproximadamente 4-5 motos por predição
- Modelo de saídas ligeiramente mais preciso

### 4.2 Exemplos de Predições

| Cenário | Saídas | Retornos | Observação |
|---------|--------|----------|------------|
| Dia útil normal | 18.40 | 18.02 | Comportamento equilibrado |
| Fim de semana + chuva | 33.38 | 28.59 | Maior movimentação |
| Segunda-feira | 12.85 | 14.03 | Menor atividade |
| Sexta-feira chuvosa | 29.15 | 25.60 | Alta atividade |

### 4.3 Padrões Identificados

#### 4.3.1 Fatores de Influência
- **Dia da semana**: Segunda-feira com menor movimento
- **Condições climáticas**: Chuva afeta diferentemente cada dia
- **Feriados**: Aumentam significativamente a demanda
- **Fins de semana**: Padrão distinto dos dias úteis

#### 4.3.2 Limitações
- Dataset pequeno (95 registros)
- Dados de apenas um galpão
- Possíveis fatores externos não considerados

## 5. CONCLUSÕES

### 5.1 Resultados Alcançados
O projeto demonstrou a viabilidade de usar redes neurais para previsão de demanda de motocicletas:
- **Implementação bem-sucedida**: Dois modelos funcionais
- **Performance adequada**: Erros dentro do esperado para o problema
- **Padrões capturados**: Modelos identificaram tendências nos dados

### 5.2 Aprendizados
- **Redes neurais**: Adequadas para problemas de regressão com dados tabulares
- **Pré-processamento**: Normalização essencial para convergência
- **Duas redes separadas**: Estratégia eficaz para problemas multi-output

### 5.3 Limitações e Trabalhos Futuros
**Limitações**:
- Dataset pequeno (95 registros)
- Dados de apenas um galpão
- Falta de validação cruzada

**Melhorias Futuras**:
- Incluir mais dados de diferentes galpões
- Implementar validação cruzada
- Testar outros algoritmos (Random Forest, XGBoost)
- Adicionar features temporais (sazonalidade)
- Otimizar hiperparâmetros

## 6. REFERÊNCIAS

### 6.1 Tecnologias
- TensorFlow/Keras: Framework de deep learning
- Scikit-learn: Biblioteca de machine learning
- Pandas: Manipulação de dados
- NumPy: Computação numérica
- Matplotlib: Visualização

### 6.2 Conceitos
- Redes Neurais Artificiais
- Algoritmo Adam
- Mean Squared Error (MSE)
- Normalização MinMax
- Problemas de Regressão 