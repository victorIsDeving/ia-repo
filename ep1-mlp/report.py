# ALUNO VICTOR AUGUSTO COSTA MONTEIRO
# NUSP 8942937
import numpy as np


def markdown_report(nome_arquivo, infos, conf_matrix=None):
    with open(nome_arquivo, 'w') as arquivo:
        arquivo.write("# Relatório do Treinamento e Teste do Modelo de Rede Neural\n\n")

        arquivo.write("## Arquitetura da Rede Neural\n")
        arquivo.write(f"- Total de perceptrons na camada escondida: {infos['perceptrons_camada_escondida']}\n")
        arquivo.write(f"- Total de perceptrons na camada de saída: {infos['perceptrons_camada_saida']}\n\n")

        arquivo.write("## Treinamento\n")
        arquivo.write("### Infos Gerais\n")
        arquivo.write(f"- Função de ativação: {infos['funcao_ativacao']}\n")
        arquivo.write(f"- Aprendizado: {infos['aprendizado']}\n")
        arquivo.write(f"- Total de épocas rodadas: {infos['total_epocas']} (de {infos['maximo_epocas']} máximo épocas)\n")
        arquivo.write(f"- Com parada antecipada: {infos['parada_antecipada']}\n")
        if infos['parada_antecipada']:
            arquivo.write(f"- Paciência parada antecipada: {infos['parada_antecipada_paciencia']}\n")
            arquivo.write(f"- Parou antecipadamente: {infos['parou_antecipadamente']}\n")
            arquivo.write(f"- Melhor erro de validação: {infos['melhor_erro_validacao']}\n")
        arquivo.write(f"- Tempo de processamento: {infos['tempo_termino'] - infos['tempo_inicio']} seconds\n")
        arquivo.write("### Pesos Resultado\n")
        arquivo.write("- Pesos Iniciais:\n")
        arquivo.write(f"  - Camada Escondida: arquivo 'pesos_iniciais_camada_escondida.npy'\n")
        arquivo.write(f"  - Camada de Saída: arquivo 'pesos_iniciais_camada_saida.npy'\n")
        arquivo.write("- Pesos Finais:\n")
        arquivo.write(f"  - Camada Escondida: arquivo 'pesos_finais_camada_escondida.npy'\n")
        arquivo.write(f"  - Camada de Saída: arquivo 'pesos_finais_camada_saida.npy\n\n")

        arquivo.write("## Teste\n")
        arquivo.write("### Infos Gerais\n")
        arquivo.write(f"- Função de ativação: {infos['funcao_ativacao']}\n")
        arquivo.write(f"- Threshold: {infos['threshold']}\n")
        arquivo.write(f"- Cross Validation: {infos['cross_validacao']}\n")
        if infos['cross_validacao']:
            arquivo.write(f"    - Folds totais: {infos['cross_validacao_folds']}\n")
            arquivo.write("- Holdout: False\n")
        else:
            arquivo.write("- Holdout: True\n")
        arquivo.write(f"- Resultado final:\n")
        arquivo.write(f"    - Saída final do teste: arquivo 'output_final.npy\n")
        arquivo.write(f"    - Saída camada escondida do teste: arquivo 'output_escondida.npy'\n\n")

        arquivo.write("### Matriz de Confusão\n")
        if conf_matrix is not None:
            arquivo.write(f"- Linhas são valores previstos\n")
            arquivo.write(f"- Colunas são valores verdade\n\n")
            int_letter = {
                0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Err'
            }
            true_positive = np.diag(conf_matrix)
            false_positive = np.sum(conf_matrix, axis=0) - true_positive
            arquivo.write(
                "|   | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X "
                "| Y | Z | Err |\n")
            arquivo.write(
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---"
                "|---|---|-------|\n")
            for i in range(len(conf_matrix)):
                arquivo.write(f"| {int_letter[i]} |")
                for j in range(len(conf_matrix[0])):
                    if i == j:
                        arquivo.write(f" **<u>{conf_matrix[i][j]}</u>** |")
                    else:
                        arquivo.write(f" {conf_matrix[i][j]} |")
                arquivo.write("\n")
            arquivo.write("Acertos e erros por letra:\n\n")
            arquivo.write(
                "|   | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X "
                "| Y | Z | Err |\n")
            arquivo.write(
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---"
                "|---|---|-------|\n")
            arquivo.write("| TP |")
            for i in range(len(true_positive)):
                arquivo.write(f" {true_positive[i]} |")
            arquivo.write("\n| FP |")
            for i in range(len(false_positive)):
                arquivo.write(f" {false_positive[i]} |")
            arquivo.write("\n")
        else:
            arquivo.write("- Matriz de confusão não disponibilizada.\n")


def dict_resume(hidden_perceptrons, output_perceptrons,
                learn_rate, act_func, early_stop_bool,
                early_stop_patience, stopped_early, threshold,
                cross_validation_bool, cross_validation_folds,
                training_results, testing_results, output_loc,
                start_time, end_time, max_epochs):
    info = {
        'perceptrons_camada_escondida': hidden_perceptrons,
        'perceptrons_camada_saida': output_perceptrons,
        'aprendizado': learn_rate,
        'funcao_ativacao': act_func,
        'pesos_iniciais_camada_escondida': training_results[0][0],
        'pesos_iniciais_camada_saida': training_results[0][1],
        'pesos_finais_camada_escondida': training_results[1][0],
        'pesos_finais_camada_saida': training_results[1][1],
        'total_epocas': training_results[2],
        'maximo_epocas': max_epochs,
        'melhor_erro_validacao': training_results[3],
        'parada_antecipada': early_stop_bool,
        'parada_antecipada_paciencia': early_stop_patience,
        'parou_antecipadamente': stopped_early,
        'threshold': threshold,
        'output_final': testing_results[1],
        'output_escondida': testing_results[0],
        'cross_validacao': cross_validation_bool,
        'cross_validacao_folds': cross_validation_folds,
        'tempo_inicio': start_time,
        'tempo_termino': end_time
    }
    for key in info:
        if isinstance(info[key], np.ndarray):
            np.save(f"{output_loc}/{key}.npy", info[key])
    return info
