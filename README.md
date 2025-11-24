# Proyecto Final Fundamentos de IA
# SmartMCTS Connect4 Policy

Este repositorio contiene una policy híbrida para Connect4 basada en:
- Capa táctica (ganar/bloquear inmediato).
- Reglas tipo Allis (odd threats / claim-even / follow-ups).
- Poda de movimientos con `useful_cols`.
- Minimax rápido para evitar trampas.
- Aprendizaje online con GPI/MCTS interno y tabla Q persistente en Parquet.

La versión actual permite evaluar aprendizaje en batches locales.

---


## Integrantes
 - David Esteban Diaz Vargas
 - Diego Norberto Diaz Algarin
 - Juan Pablo Moreno Patarroyo

## Archivos principales

- `policyv2.py` / `policy.py`: implementación de la clase **SmartMCTS** (policy final).
- `helpers.py`: utilidades Allis + heurística minimax + poda.
- `mctsnode.py`: nodo MCTS.
- `board_utils.py`: reglas del juego (winner, legales, transición).
- `tactical_search.py`: minimax rápido con alpha-beta.
- `gpi_core.py`: núcleo de GPI (trials, update Q, mini-MCTS, selección final).
- `test_q_logs.py`: script local para correr partidas por batches y visualizar aprendizaje.

> Nota: SmartMCTS guarda la Q-table en `q_values.parquet`.

---

## Requisitos

### Python
- Python 3.9+ recomendado.

### Librerías externas
Instala con pip:

```bash
pip install numpy pandas tqdm matplotlib pyarrow
´´´

## Pruebas
Para realizar las pruebas por favor dirigirse a la carpeta tournaments y hacer python  test_q_logs.py
