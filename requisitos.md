# Descricao

desafio é criar um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações.

# Coleta de Dados

utilize um dataset de preços históricos de ações, como o Yahoo Finance.

# Desenvolvimento do Modelo LSTM

• Construção do Modelo: implemente um modelo de deep learning utilizando LSTM para capturar padrões temporais nos dados de preços das ações.

• Treinamento: treine o modelo utilizando uma parte dos dados e ajuste os hiperparâmetros para otimizar o desempenho.

• Avaliação: avalie o modelo utilizando dados de validação e utilize métricas como MAE (Mean Absolute Error), RMSE (Root Mean Square Error), MAPE (Erro Percentual Absoluto Médio) ou outra métrica apropriada para medir a precisão das previsões.

# Salvamento e Exportação do Modelo

• Salvar o Modelo: após atingir um desempenho satisfatório, salve o modelo treinado em um formato que possa ser utilizado para inferência.

# Deploy do Modelo

• Criação da API: desenvolva uma API RESTful utilizando Flask ou FastAPI para servir o modelo. A API deve permitir que o usuário forneça dados históricos de preços e receba previsões dos preços futuros.

# Escalabilidade e Monitoramento

• Monitoramento: configure ferramentas de monitoramento para rastrear a performance do modelo em produção, incluindo tempo de resposta e utilização de recursos.