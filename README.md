# Previsão de Demanda de Motocicletas - Sistema Mottu

## 📋 Visão Geral

Este projeto acadêmico desenvolve um sistema de machine learning para prever a demanda de motocicletas em uma frota de delivery, aplicando conceitos de redes neurais para resolver um problema real de gestão operacional.

## 🎯 Problema

A empresa Mottu precisa otimizar a gestão de sua frota de motocicletas, enfrentando dificuldades em:

- Prever quantas motos sairão e retornarão diariamente
- Considerar fatores como clima, dia da semana e feriados
- Otimizar a distribuição de veículos nos galpões

## 🧠 Solução

Desenvolvemos duas redes neurais que preveem:
1. Número de motocicletas que sairão do galpão
2. Número de motocicletas que retornarão ao galpão

### Dataset
O dataset contém informações sobre:
- Localização do galpão
- Dia da semana e tipo de dia (útil/fim de semana)
- Condições climáticas (chuva)
- Estado atual da frota (motos em uso, disponíveis)
- Histórico do dia anterior

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem de programação
- **TensorFlow/Keras**: Construção das redes neurais
- **scikit-learn**: Pré-processamento e divisão dos dados
- **pandas**: Manipulação de dados
- **matplotlib**: Visualização dos resultados

## 🏗️ Metodologia

### Arquitetura da Rede Neural
Cada rede neural possui:
- **Entrada**: 9 variáveis (features)
- **Camada oculta 1**: 16 neurônios com ativação ReLU
- **Camada oculta 2**: 8 neurônios com ativação ReLU  
- **Saída**: 1 neurônio com ativação linear

### Configurações
- **Otimizador**: Adam
- **Função de perda**: MSE (Mean Squared Error)
- **Épocas**: 100
- **Divisão dos dados**: 70% treino, 30% teste

### Pré-processamento
- Normalização das variáveis com MinMaxScaler
- Codificação de variáveis categóricas
- Divisão aleatória dos dados (seed=42)

## 📊 Resultados

### Performance dos Modelos
- **Erro médio (Saídas)**: 18.19 MSE
- **Erro médio (Retornos)**: 19.59 MSE

### Exemplos de Predições
| Cenário | Saídas Previstas | Retornos Previstos |
|---------|------------------|-------------------|
| Dia útil, sem chuva | 18.40 | 18.02 |
| Fim de semana, com chuva | 33.38 | 28.59 |
| Segunda-feira normal | 12.85 | 14.03 |
| Sexta-feira chuvosa | 29.15 | 25.60 |

## 🚀 Como Executar

### Instalação
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

### Execução
1. Baixe os arquivos do projeto
2. Execute o notebook `ml.ipynb`

### Arquivos do Projeto
- `ml.ipynb` - Código principal
- `dados_mottu_corrigido.csv` - Dataset
- `README.md` - Documentação

## 📈 Justificativas

### Por que Redes Neurais?
- Adequadas para problemas de regressão
- Capazes de capturar relações não-lineares
- Flexíveis para diferentes tipos de dados

### Escolhas de Design
- **Duas redes separadas**: Cada uma se especializa em um tipo de predição
- **Arquitetura 16→8→1**: Redução gradual permite melhor aprendizado
- **Normalização**: Melhora a convergência do modelo

## 👥 Equipe

- Marcos Vinicius Pereira de Oliveira - RM 557252
- Ruan Lima Silva - RM 558775  
- Richardy Borges Santana - RM 557883

## 🎥 Apresentação

**Vídeo do Projeto**: [YouTube](https://www.youtube.com/watch?v=5nN9AZ1_jY4)

## 🔮 Melhorias Futuras

- Incluir mais dados de diferentes galpões
- Implementar validação cruzada
- Testar outros algoritmos de machine learning
- Adicionar features temporais (sazonalidade)
- Otimizar hiperparâmetros

## 📝 Nota

Projeto desenvolvido para fins acadêmicos. 