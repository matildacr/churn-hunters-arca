# Churn Hunters — Arca Continental | Hack 4 Her

Sistema de prediccion de churn para el canal tradicional de Arca Continental.
Desarrollado en Analytics · Hack 4 Her · Tec de Monterrey.

---

## El problema que resolvemos

Arca Continental pierde **$12 millones MXN al mes** porque sus clientes del canal tradicional (tienditas) dejan de comprar sin aviso. El equipo comercial se entera cuando ya es demasiado tarde.

**Nuestra solucion:** un sistema que predice 1 mes antes que clientes van a dejar de comprar, para que el equipo comercial pueda actuar a tiempo.

---

## Como funciona en 3 pasos simples

**1. Aprende del pasado**
El modelo analiza 25 meses de historial de compras de 241,805 tienditas.

**2. Detecta patrones de riesgo**
Identifica senales de alerta: clientes que llevan meses comprando cada vez menos, que perdieron refrigeradores, o que estan por debajo de su zona geografica.

**3. Genera una lista de accion**
Cada mes, el equipo comercial recibe una lista con los clientes en riesgo, su nivel (Alto/Medio/Bajo) y por que estan en peligro.

---

## Resultados del modelo

| Metrica | Resultado |
|---|---|
| PR-AUC | 0.9262 |
| ROC-AUC | 0.9995 |
| Mejora vs no hacer nada | 107 veces mejor |
| Clientes evaluados en feb 2026 | 199,923 |
| Clientes detectados en riesgo alto | 1,724 |

---

## Las 3 preguntas clave — respondidas con datos

### Que variables influyen mas en el churn?

La senal mas poderosa es el **patron de desenganche progresivo del cliente** — el modelo detecta cuando un cliente lleva meses comprando cada vez menos, por debajo de su propio promedio historico. No espera a que deje de comprar: identifica la tendencia antes de que llegue a cero. Un cliente que venia comprando 100 cajas y bajo a 40, luego a 15, esta dando una senal clara. El modelo la detecta en enero para que el equipo actue en febrero antes de que ese cliente desaparezca.

Las 5 senales mas importantes:
1. Patron de desenganche progresivo — 72% (cuantos meses lleva por debajo de su propio promedio)
2. Posicion relativa en su territorio — 9% (si vende menos que sus vecinos de zona)
3. Ventas vs mediana de su zona — 8% (que tan lejos esta del tipico cliente de su region)
4. Coolers perdidos en los ultimos 3 meses — 2% (perder un cooler es senal de deterioro en la relacion)
5. Ventas del mes anterior — 2% (el historial reciente mas inmediato)

### El territorio influye en la perdida de clientes?

Si. Hay territorios con hasta **4.8 veces mas churn** que otros.

| Territorio | Tasa de churn | Riesgo |
|---|---|---|
| Monclova | 1.35% | Alto — 1.6x la media |
| Reynosa | 1.31% | Alto — 1.5x la media |
| Guadalajara | 1.13% | Alto — 1.3x la media |
| Durango | 0.28% | Bajo — 0.3x la media |

**Recomendacion:** priorizar visitas del equipo comercial en la zona norte.

### Los coolers afectan el riesgo de churn?

Es el hallazgo mas poderoso del analisis:

| Coolers instalados | Tasa de churn | Reduccion de riesgo |
|---|---|---|
| 0 coolers | 3.34% | — |
| 1 cooler | 0.73% | 78% menos riesgo |
| 2 coolers | 0.30% | 91% menos riesgo |
| 3 coolers | 0.17% | 95% menos riesgo |
| 6 o mas | 0.05% | 99% menos riesgo |

**Instalar un solo cooler reduce el riesgo de perder al cliente en un 78%.**

---

## Como se valido el modelo

El modelo entrena con datos de enero 2024 a diciembre 2025. Luego predice enero 2026 sin haber visto ese mes. Se compara la prediccion contra lo que realmente paso.

Esto garantiza que las metricas son reales — el modelo no hace trampa viendo el futuro.

---

## Archivos del proyecto

| Archivo | Que es |
|---|---|
| `churn_model_final.py` | El modelo completo — se corre con python3 churn_model_final.py |
| `preds_submission_final.csv` | Predicciones 0/1 por cliente para febrero 2026 |
| `scoring_comercial.csv` | Lista de clientes con nivel de riesgo Alto/Medio/Bajo |
| `feature_importance.csv` | Que variables son mas importantes para predecir el churn |
| `requirements.txt` | Librerias necesarias para correr el modelo |

---

## Como correr el modelo

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Poner los 4 CSVs de Arca en la misma carpeta
#    sales_churn_train.csv
#    sales_churn_test.csv
#    Coolers.csv
#    Clientes.csv
#    preds_submission.csv

# 3. Correr
python3 churn_model_final.py
```

El modelo genera los archivos de resultados automaticamente en 2 minutos.

---

## Impacto esperado

Con este sistema, el equipo comercial puede:

- Llamar a los **1,724 clientes en riesgo** antes de que dejen de comprar
- Priorizar instalacion de coolers en clientes sin refrigeradores
- Enfocar esfuerzos comerciales en la zona norte del pais
- Recuperar hasta **$11.8 MM MXN al mes** con una tasa de retencion del 30%

---

## Equipo

**Churn Hunters**
Analytics · Hack 4 Her · Tec de Monterrey
Reto de Arca Continental — Digital Nest
