# Churn Hunters
### Sistema de Prediccion de Churn — Arca Continental

> *"El 1 de febrero, antes de que empiece el mes, ya sabemos quien no va a comprar."*

---

## El problema vale $12,000,000 MXN al mes

Cada mes, cientos de tienditas del canal tradicional dejan de comprarle a Arca Continental sin aviso. El equipo comercial se entera cuando ya es demasiado tarde para actuar.

**El costo de reaccionar tarde: $12 MM MXN perdidos cada mes.**

Churn Hunters resuelve eso.

---

## Que hace el sistema

El 1 de cada mes, el sistema analiza el historial de compras de cada tiendita y entrega una lista con tres niveles de alerta:

| Nivel | Que significa | Accion recomendada |
|---|---|---|
| 🔴 Alto riesgo | Va a dejar de comprar este mes | Llamada urgente esta semana |
| 🟡 Riesgo medio | Señales de deterioro — vigilar | Visita comercial este mes |
| 🟢 Bajo riesgo | Cliente estable | Seguimiento normal |

**El equipo comercial deja de apagar incendios y empieza a prevenirlos.**

---

## Resultados — validados con datos reales

El modelo se entreno con 24 meses de historia y se probo en meses que nunca habia visto. Sin trampa. Sin ver el futuro.

| Metrica | Resultado | Que significa |
|---|---|---|
| PR-AUC | **0.9262** | 107 veces mejor que no hacer nada |
| ROC-AUC | **0.9995** | Casi perfecto separando riesgo real de ruido |
| Recall | **100%** | No se escapa ningun cliente en riesgo |
| Tiempo de ejecucion | **2 minutos** | Listo para correr cada mes |

---

## Los 3 hallazgos que cambian la estrategia comercial

### 1. El sistema detecta el deterioro antes de que sea evidente

Un cliente no deja de comprar de un dia para otro. Primero compra menos. Luego mucho menos. Luego desaparece. El modelo identifica ese patron de caida progresiva **1 mes antes de que llegue a cero** — cuando todavia hay tiempo de actuar.

### 2. La zona norte necesita atencion urgente

No todos los territorios pierden clientes igual. Hay zonas con hasta **4.8 veces mas churn** que otras.

| Territorio | Riesgo | vs la media nacional |
|---|---|---|
| Monclova | 🔴 1.35% | 1.6 veces mas alto |
| Reynosa | 🔴 1.31% | 1.5 veces mas alto |
| Guadalajara | 🟡 1.13% | 1.3 veces mas alto |
| Durango | 🟢 0.28% | 3 veces mas bajo |

**El sistema le dice al equipo comercial donde enfocar sus recursos.**

### 3. Un cooler instalado vale mas que mil llamadas

Este es el hallazgo mas poderoso del analisis:

| Coolers en el punto de venta | Probabilidad de perder al cliente |
|---|---|
| Sin coolers | 3.34% — riesgo critico |
| 1 cooler | 0.73% — 78% menos riesgo |
| 2 coolers | 0.30% — 91% menos riesgo |
| 3 o mas coolers | Menos de 0.17% — riesgo minimo |

**Instalar un cooler es la inversion de retencion con mayor retorno que tiene Arca.**
Cada cooler instalado es un ancla que ata al cliente a la marca.

---

## Por que este sistema es diferente

La mayoria de los modelos de churn detectan cuando el cliente ya se fue. Este predice antes de que pase.

- **Predice 1 mes antes** — usando solo datos historicos, sin ver el mes a predecir
- **Validado con rigor** — entrenado en el pasado, probado en meses que nunca vio
- **Listo para produccion** — corre en 2 minutos, genera resultados automaticamente
- **Escalable** — funciona con cualquier mes nuevo de datos sin modificar el codigo

---

## Impacto potencial para Arca

Con una tasa de retencion del 30% sobre los clientes detectados:

**$11,800,000 MXN recuperables cada mes.**

En un ano: **$141,600,000 MXN** que hoy se pierden y con este sistema se pueden recuperar.

---

## Listo para produccion en 30 dias

```
Semana 1 — Integracion de datos con los sistemas de Arca
Semana 2 — Prueba piloto con el equipo comercial de 2 territorios
Semana 3 — Ajuste fino con retroalimentacion del equipo
Semana 4 — Lanzamiento a todos los territorios
Mes 2+   — Operacion automatica mensual
```

El equipo comercial recibe su lista de accion el primer dia de cada mes.

---

## Como correr el modelo

```bash
# Instalar dependencias (una sola vez)
pip install -r requirements.txt

# Agregar los 4 archivos de datos de Arca en la carpeta
# Correr
python3 churn_model_final.py
```

El sistema genera automaticamente:
- Lista de clientes en riesgo con nivel Alto / Medio / Bajo
- Probabilidad de churn por cliente
- Ranking de las variables mas importantes ese mes

---

## Archivos

| Archivo | Descripcion |
|---|---|
| `churn_model_final.py` | El sistema completo |
| `preds_submission_final.csv` | Predicciones de febrero 2026 |
| `scoring_comercial.csv` | Lista de accion para el equipo de ventas |
| `feature_importance.csv` | Variables mas predictivas |
| `requirements.txt` | Dependencias |

---

**Churn Hunters** — Analytics · Hack 4 Her · Tec de Monterrey

*Construido para Arca Continental con datos reales del canal tradicional de Mexico.*

---

## Detalle Tecnico

### Datos utilizados

4 tablas reales de Arca Continental — canal tradicional, Mexico, 2024-2026:

| Archivo | Contenido | Registros |
|---|---|---|
| `sales_churn_train.csv` | Ventas por cliente y mes con etiqueta de churn | 5,030,534 |
| `sales_churn_test.csv` | Clientes de febrero 2026 a predecir | 199,923 |
| `Coolers.csv` | Refrigeradores por cliente y mes | 4,636,676 |
| `Clientes.csv` | Territorio, subcanal y tamano de cliente | 371,727 |

### Decisiones de limpieza

| Problema | Decision | Justificacion |
|---|---|---|
| 79 filas con ventas negativas | Convertir a NaN, no eliminar | Son devoluciones contables. Eliminar la fila corromperia el lag del mes siguiente. El cliente conserva su historial completo |
| 148 filas duplicadas en Coolers | Sumar los valores | Son distintos tipos de refrigerador en el mismo punto de venta. La suma refleja el total real de equipos instalados |
| 15,438 clientes sin clasificacion de tamano | Categoria "Desconocido" | No se imputa con la moda — los sin clasificar pueden tener su propio patron de churn |

### Features del modelo (40 variables)

Todas las variables usan **solo el historial previo** al mes a predecir. Nada del mes de febrero.

- **Patron de desenganche:** cuantos meses consecutivos lleva el cliente por debajo de su propio promedio historico
- **Lags de ventas:** ventas de los ultimos 6 meses (lag1 a lag6)
- **Tendencia:** pendiente de caida en los ultimos 3 y 6 meses
- **Alertas criticas:** flag si el mes anterior tuvo menos de 5 cajas, menos de 10, o cero
- **Coolers:** coolers actuales, perdidos en el ultimo mes, perdidos en los ultimos 3 meses
- **Contexto geografico:** ventas vs mediana del territorio, percentil dentro de la zona
- **Estacionalidad:** ventas del mismo mes del ano anterior
- **Cliente:** antiguedad, tamano, subcanal comercial, territorio

### Modelo y validacion

**Algoritmo:** XGBoost con hiperparametros optimizados por Optuna (30 pruebas automaticas)

**Validacion Walk-Forward:**
- Entrenamiento: enero 2024 → diciembre 2025 (24 meses, 4.59 millones de registros)
- Validacion: enero 2026 (mes que el modelo nunca vio durante el entrenamiento)
- Prediccion: febrero 2026

**Manejo del desbalance:** `scale_pos_weight = 110` — por cada churner hay 110 clientes normales. El modelo penaliza 110 veces mas equivocarse en un churner que en un cliente normal.

**Metricas:**

| Metrica | Valor | Por que importa |
|---|---|---|
| PR-AUC | 0.9262 | Metrica correcta para clases desbalanceadas. Un modelo aleatorio tendria 0.009 |
| ROC-AUC | 0.9995 | Capacidad de discriminar entre churners y no-churners |
| Recall | 100% | No se escapa ningun churner en la validacion |
| Precision | 78% | De cada 100 alertas, 78 son churns reales |

**Hiperparametros optimos (Optuna):**
```
n_estimators:     597
max_depth:        7
learning_rate:    0.031
subsample:        0.95
colsample_bytree: 0.75
min_child_weight: 16
gamma:            1.85
reg_alpha:        0.37
reg_lambda:       3.27
scale_pos_weight: 110
```

### Por que XGBoost y no otro modelo

Comparamos XGBoost contra LightGBM, Random Forest y Regresion Logistica en los mismos datos con la misma validacion temporal. XGBoost obtuvo el mayor PR-AUC en todos los experimentos. Ademas:

- Maneja el desbalance 110:1 de forma nativa con `scale_pos_weight`
- `tree_method=hist` permite entrenar 5 millones de filas en menos de 2 minutos
- Interpretable con feature importance — cada prediccion es explicable
- `early_stopping_rounds=50` evita overfitting automaticamente

### Reproducibilidad

El pipeline es completamente reproducible:
1. Los mismos datos de entrada siempre producen los mismos resultados
2. La semilla aleatoria esta fijada (`random_state=42`)
3. El script detecta automaticamente los archivos de datos aunque tengan prefijos diferentes
4. Genera todos los archivos de salida en una sola ejecucion de 2 minutos
5. 
