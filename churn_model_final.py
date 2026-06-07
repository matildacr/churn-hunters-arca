"""
CHURN HUNTERS - Arca Continental - Hack 4 Her
==============================================
Modelo de prediccion de churn - Version produccion

La logica del negocio confirma:
- target=1 SIEMPRE corresponde a uni_boxes_sold_m=0 (100% de los casos)
- El modelo usa las ventas del mes actual + historial para predecir churn
- En produccion: correr a fin de mes con ventas del mes en curso

Correr: python3 churn_model_final.py
"""

import pandas as pd
import numpy as np
import gc, time, warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
t0 = time.time()

def find(names):
    for name in names:
        for f in Path('.').glob(f'*{name}*'):
            return str(f)
    raise FileNotFoundError(f'No encontre: {names}')

# ============================================================
# PASO 1 - CARGA
# ============================================================
print('\n' + '='*55)
print('PASO 1: Cargando datos...')
print('='*55)

train = pd.read_csv(find(['sales_churn_train']),
    dtype={'customer_id':'str','calmonth':'int32','target':'int8'},
    usecols=['customer_id','calmonth','num_transacciones','uni_boxes_sold_m','target'])
coolers = pd.read_csv(find(['Coolers','coolers']),
    dtype={'customer_id':'str','calmonth':'int32'},
    usecols=['customer_id','calmonth','num_coolers','num_doors'])
clientes = pd.read_csv(find(['Clientes','clientes']), dtype='str',
    usecols=['customer_id','territory_d','comercial_subchannel_d','rtm_customer_size_d'])
test = pd.read_csv(find(['sales_churn_test']),
    dtype={'customer_id':'str','calmonth':'int32'},
    usecols=['customer_id','calmonth'])
# No cargamos uni_boxes_sold_m del test - predecimos sin ver febrero
sub = pd.read_csv(find(['preds_submission']), usecols=['customer_id'])

CHURN_RATE = train['target'].mean()
print(f'  Train:    {len(train):,} filas | {train["customer_id"].nunique():,} clientes')
print(f'  Churn rate historico: {CHURN_RATE*100:.2f}%')
print(f'  Confirmado: target=1 siempre tiene ventas=0')

# ============================================================
# PASO 2 - LIMPIEZA
# ============================================================
print('\n' + '='*55)
print('PASO 2: Limpieza...')
print('='*55)

# Negativos -> NaN (devoluciones contables, no eliminar)
neg = (train['uni_boxes_sold_m'] < 0).sum()
train.loc[train['uni_boxes_sold_m'] < 0, 'uni_boxes_sold_m'] = np.nan
print(f'  [1] {neg} devoluciones -> NaN')

# Coolers duplicados -> sumar (distintos tipos de refrigerador)
coolers = coolers.groupby(['customer_id','calmonth']).agg(
    num_coolers=('num_coolers','sum'),
    num_doors=('num_doors','sum')
).reset_index()
print(f'  [2] Coolers consolidados: {len(coolers):,} filas')

clientes['rtm_customer_size_d'] = clientes['rtm_customer_size_d'].fillna('Desconocido')
print(f'  [3] Nulos tamano -> Desconocido')

# ============================================================
# PASO 3 - FEATURE ENGINEERING
# Incluye ventas del mes actual + historial completo
# ============================================================
print('\n' + '='*55)
print('PASO 3: Feature engineering...')
print('='*55)

train = train.sort_values(['customer_id','calmonth']).reset_index(drop=True)
gv = train.groupby('customer_id')['uni_boxes_sold_m']
gt = train.groupby('customer_id')['num_transacciones']

# Lags de ventas (historial)
for i in range(1, 7):
    train[f'lag{i}'] = gv.shift(i).astype('float32')
train['t_lag1'] = gt.shift(1).astype('float32')
train['t_lag2'] = gt.shift(2).astype('float32')

# Estadisticas historicas
train['mean3']   = ((train['lag1']+train['lag2']+train['lag3'])/3).astype('float32')
train['mean6']   = sum(train[f'lag{i}'] for i in range(1,7)).div(6).astype('float32')
train['std3']    = gv.shift(1).rolling(3, min_periods=2).std().astype('float32')
train['min3']    = train[['lag1','lag2','lag3']].min(axis=1).astype('float32')
train['trend3']  = ((train['lag1']-train['lag3'])/(train['lag3']+1)).astype('float32')
train['trend6']  = ((train['lag1']-train['lag6'])/(train['lag6']+1)).astype('float32')
train['accel']   = (train['lag1']-train['lag2']).astype('float32')
train['ventas_max']   = gv.cummax().shift(1).astype('float32')
train['caida_vs_max'] = ((train['ventas_max']-train['lag1'])/(train['ventas_max']+1)).astype('float32')

# Features del mes ACTUAL (la señal mas fuerte)
# target=1 siempre tiene uni_boxes_sold_m=0
train['ventas_cero']   = (train['uni_boxes_sold_m']==0).astype('int8')
train['ventas_lt5']    = ((train['uni_boxes_sold_m']>0)&(train['uni_boxes_sold_m']<5)).astype('int8')
train['ventas_lt20']   = ((train['uni_boxes_sold_m']>=5)&(train['uni_boxes_sold_m']<20)).astype('int8')
train['caida_actual']  = ((train['lag1']-train['uni_boxes_sold_m'])/(train['lag1']+1)).astype('float32')
train['ratio_vs_mean'] = (train['uni_boxes_sold_m']/(train['mean3']+1)).astype('float32')
train['ratio_vs_lag1'] = (train['uni_boxes_sold_m']/(train['lag1']+1)).astype('float32')

# Alertas historicas
train['lag1_cero']  = (train['lag1']==0).astype('int8')
train['lag1_lt5']   = ((train['lag1']>0)&(train['lag1']<5)).astype('int8')
train['lag1_lt10']  = ((train['lag1']>=5)&(train['lag1']<10)).astype('int8')
train['cayendo2']   = ((train['lag1']<train['lag2'])&(train['lag2']<train['lag3'])).astype('int8')
train['meses_lt10'] = sum((train[f'lag{i}']<10).astype(int) for i in range(1,5)).astype('int8')
train['months_active'] = train.groupby('customer_id').cumcount().astype('int16')

# Meses sin comprar
ultima = train[train['uni_boxes_sold_m']>0].groupby('customer_id')['calmonth'].transform('max')
train['meses_sin_comprar'] = (train['calmonth']-ultima).fillna(99).clip(upper=24).astype('float32')

print('  Ventas actuales + historial OK')

# Coolers
train = train.merge(coolers, on=['customer_id','calmonth'], how='left')
train['num_coolers'] = train['num_coolers'].fillna(0).astype('float32')
train['num_doors']   = train['num_doors'].fillna(0).astype('float32')
gc2 = train.groupby('customer_id')['num_coolers']
for i in range(1,4):
    train[f'cool_lag{i}'] = gc2.shift(i).fillna(0).astype('float32')
train['sin_coolers']  = (train['num_coolers']==0).astype('int8')
train['cool_lost']    = (train['cool_lag1']-train['num_coolers']).clip(lower=0).astype('float32')
train['cool_lost_3m'] = (train['cool_lag3']-train['num_coolers']).clip(lower=0).astype('float32')
train['perdio_cooler']= (train['cool_lost']>0).astype('int8')
print('  Coolers OK')

# Clientes y contexto
train = train.merge(clientes, on='customer_id', how='left')
train['rtm_customer_size_d'] = train['rtm_customer_size_d'].fillna('Desconocido')
med_terr = train.groupby(['territory_d','calmonth'])['lag1'].transform('median')
train['vs_med_terr'] = (train['lag1']/(med_terr+1)).astype('float32')
train['pct_terr']    = train.groupby(['territory_d','calmonth'])['lag1'].rank(pct=True).astype('float32')

# Estacionalidad
train['calmonth_ant'] = train['calmonth'] - 100
est = train[['customer_id','calmonth','uni_boxes_sold_m']].rename(
    columns={'calmonth':'calmonth_ant','uni_boxes_sold_m':'v_anio_ant'})
train = train.merge(est, on=['customer_id','calmonth_ant'], how='left')
train['v_anio_ant']   = train['v_anio_ant'].astype('float32')
train['ratio_anio']   = (train['lag1']/(train['v_anio_ant']+1)).astype('float32')
train.drop(columns=['calmonth_ant'], inplace=True)

le_dict = {}
for col in ['territory_d','comercial_subchannel_d','rtm_customer_size_d']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    le_dict[col] = le

print('  Territorio y estacionalidad OK')

# ============================================================
# FEATURES - ventas actuales + historial completo
# ============================================================
FEATURES = [
    # Historial de ventas (NO usamos ventas de febrero)
    # Solo lo que sabemos ANTES de que empiece febrero
    # Historial de ventas
    'lag1','lag2','lag3','lag4','lag5','lag6',
    't_lag1','t_lag2',
    # Estadisticas historicas
    'mean3','mean6','std3','min3',
    'trend3','trend6','accel',
    'caida_vs_max','ventas_max',
    # Alertas historicas
    'lag1_cero','lag1_lt5','lag1_lt10',
    'cayendo2','meses_lt10',
    'months_active','meses_sin_comprar',
    # Coolers
    'num_coolers','num_doors',
    'cool_lag1','cool_lag2','cool_lag3',
    'sin_coolers','cool_lost','cool_lost_3m','perdio_cooler',
    # Contexto
    'vs_med_terr','pct_terr',
    'territory_d','comercial_subchannel_d','rtm_customer_size_d',
    # Estacionalidad
    'v_anio_ant','ratio_anio',
]

print(f'\n  {len(FEATURES)} features totales')

# Guardar historial para test
last = train.sort_values('calmonth').groupby('customer_id').last().reset_index()
prev = train.sort_values('calmonth').groupby('customer_id').nth(-2).reset_index()
ante = train.sort_values('calmonth').groupby('customer_id').nth(-3).reset_index()
p4   = train.sort_values('calmonth').groupby('customer_id').nth(-4).reset_index()
p5   = train.sort_values('calmonth').groupby('customer_id').nth(-5).reset_index()
p6   = train.sort_values('calmonth').groupby('customer_id').nth(-6).reset_index()

# ============================================================
# PASO 4 - WALK-FORWARD VALIDATION
# ============================================================
print('\n' + '='*55)
print('PASO 4: Walk-Forward Validation...')
print('='*55)

VAL = [202601]  # Solo enero 2026 - mas datos de entrenamiento
df_tr = train[~train['calmonth'].isin(VAL)].dropna(subset=['lag1','target'])
df_v  = train[train['calmonth'].isin(VAL)].dropna(subset=['lag1','target'])

X_tr = df_tr[FEATURES].fillna(0); y_tr = df_tr['target'].astype(int)
X_v  = df_v[FEATURES].fillna(0);  y_v  = df_v['target'].astype(int)

ratio = (y_tr==0).sum()/(y_tr==1).sum()
print(f'  Entrenamiento: ene 2024 -> dic 2025 | {len(X_tr):,} registros')
print(f'  Validacion:    enero 2026 solamente   | {len(X_v):,} registros')
print(f'  Churns reales en validacion: {y_v.sum():,}')
print(f'  Desbalance: {ratio:.0f}:1')

del train, df_tr, df_v; gc.collect()

# ============================================================
# PASO 5 - ENTRENAMIENTO
# ============================================================
print('\n' + '='*55)
print('PASO 5: Entrenamiento XGBoost...')
print('='*55)

model = xgb.XGBClassifier(
    n_estimators          = 597,
    max_depth             = 7,
    learning_rate         = 0.031,
    subsample             = 0.95,
    colsample_bytree      = 0.75,
    colsample_bylevel     = 0.75,
    min_child_weight      = 16,
    gamma                 = 1.85,
    reg_alpha             = 0.37,
    reg_lambda            = 3.27,
    scale_pos_weight      = ratio,
    early_stopping_rounds = 50,
    random_state          = 42,
    n_jobs                = -1,
    eval_metric           = 'aucpr',
    verbosity             = 0,
    tree_method           = 'hist',
)

print('  Entrenando... (3-8 min)')
model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=50)

# ============================================================
# PASO 6 - METRICAS
# ============================================================
print('\n' + '='*55)
print('PASO 6: Metricas...')
print('='*55)

p_val  = model.predict_proba(X_v)[:,1]
pr_auc = average_precision_score(y_v, p_val)
roc    = roc_auc_score(y_v, p_val)

print(f'\n  PR-AUC  (metrica principal): {pr_auc:.4f}')
print(f'  ROC-AUC:                     {roc:.4f}')
print(f'  Mejor iteracion:             {model.best_iteration}')
print(f'  Mejora vs baseline:          {pr_auc/CHURN_RATE:.0f}x')

print(f'\n  Analisis por umbral:')
print(f'  {"Umbral":>6} | {"Detectados":>15} | {"Prec":>6} | {"Recall":>6} | {"F1":>6}')
print('  ' + '-'*50)
for thr in [0.1, 0.3, 0.5, 0.7]:
    p = (p_val>=thr).astype(int)
    tp=((p==1)&(y_v==1)).sum(); fp=((p==1)&(y_v==0)).sum(); fn=((p==0)&(y_v==1)).sum()
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
    print(f'  {thr:>6.1f} | {tp:>6}/{y_v.sum():<7} | {prec:>6.3f} | {rec:>6.3f} | {f1:>6.3f}')

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f'\n  Top 15 variables:')
for i,(f,v) in enumerate(fi.head(15).items(),1):
    bar = '=' * int(v*150)
    print(f'  {i:2d}. {f:28s} {v*100:.2f}% {bar}')

fi.reset_index().rename(columns={'index':'feature',0:'importance'}).to_csv(
    'feature_importance.csv', index=False)

del X_tr, y_tr; gc.collect()

# ============================================================
# PASO 7 - FEATURES PARA TEST
# ============================================================
print('\n' + '='*55)
print('PASO 7: Features para test (feb 2026)...')
print('='*55)

COLS_LAST = ['customer_id','uni_boxes_sold_m','num_transacciones',
             'num_coolers','num_doors','cool_lag1','cool_lag2','cool_lag3',
             'territory_d','comercial_subchannel_d','rtm_customer_size_d',
             'ventas_max','sin_coolers','v_anio_ant','vs_med_terr','pct_terr',
             'meses_sin_comprar']
COLS_LAST = [c for c in COLS_LAST if c in last.columns]

test = test.merge(
    last[COLS_LAST].rename(columns={
        'uni_boxes_sold_m':'lag1',
        'num_transacciones':'t_lag1'}),
    on='customer_id', how='left')

# Las ventas reales de feb 2026 ya vienen en el test csv
# se mantienen como uni_boxes_sold_m y num_transacciones

for nth, col in [(prev,'lag2'),(ante,'lag3'),(p4,'lag4'),(p5,'lag5'),(p6,'lag6')]:
    test = test.merge(
        nth[['customer_id','uni_boxes_sold_m']].rename(
            columns={'uni_boxes_sold_m':col}),
        on='customer_id', how='left')
test = test.merge(
    prev[['customer_id','num_transacciones']].rename(
        columns={'num_transacciones':'t_lag2'}),
    on='customer_id', how='left')

for c in ['lag1','lag2','lag3','lag4','lag5','lag6','t_lag1','t_lag2',
          'num_coolers','num_doors','cool_lag1','cool_lag2','cool_lag3',
          'ventas_max','v_anio_ant','sin_coolers','vs_med_terr','pct_terr',
          'meses_sin_comprar']:
    if c not in test.columns: test[c] = 0
    test[c] = test[c].fillna(0).astype('float32')

# Recalcular derivadas con ventas actuales de febrero
test['mean3']        = ((test['lag1']+test['lag2']+test['lag3'])/3).astype('float32')
test['mean6']        = sum(test[f'lag{i}'] for i in range(1,7)).div(6).astype('float32')
test['std3']         = test[['lag1','lag2','lag3']].std(axis=1).astype('float32')
test['min3']         = test[['lag1','lag2','lag3']].min(axis=1).astype('float32')
test['trend3']       = ((test['lag1']-test['lag3'])/(test['lag3']+1)).astype('float32')
test['trend6']       = ((test['lag1']-test['lag6'])/(test['lag6']+1)).astype('float32')
test['accel']        = (test['lag1']-test['lag2']).astype('float32')
test['caida_vs_max'] = ((test['ventas_max']-test['lag1'])/(test['ventas_max']+1)).astype('float32')

# Features clave del mes actual
# test['ventas_cero']   = (test['uni_boxes_sold_m']==0).astype('int8')
# test['ventas_lt5']    = ((test['uni_boxes_sold_m']>0)&(test['uni_boxes_sold_m']<5)).astype('int8')
# test['ventas_lt20']   = ((test['uni_boxes_sold_m']>=5)&(test['uni_boxes_sold_m']<20)).astype('int8')
# test['caida_actual']  = ((test['lag1']-test['uni_boxes_sold_m'])/(test['lag1']+1)).astype('float32')
# test['ratio_vs_mean'] = (test['uni_boxes_sold_m']/(test['mean3']+1)).astype('float32')
# test['ratio_vs_lag1'] = (test['uni_boxes_sold_m']/(test['lag1']+1)).astype('float32')
test['lag1_cero']     = (test['lag1']==0).astype('int8')
test['lag1_lt5']      = ((test['lag1']>0)&(test['lag1']<5)).astype('int8')
test['lag1_lt10']     = ((test['lag1']>=5)&(test['lag1']<10)).astype('int8')
test['cayendo2']      = ((test['lag1']<test['lag2'])&(test['lag2']<test['lag3'])).astype('int8')
test['meses_lt10']    = sum((test[f'lag{i}']<10).astype(int) for i in range(1,5)).astype('int8')
test['months_active'] = test.get('months_active', pd.Series(12, index=test.index)).fillna(12).astype('int16')
test['meses_sin_comprar'] = test['meses_sin_comprar'].fillna(1).astype('float32')
test['cool_lost']     = (test['cool_lag1']-test['num_coolers']).clip(lower=0).astype('float32')
test['cool_lost_3m']  = (test['cool_lag3']-test['num_coolers']).clip(lower=0).astype('float32')
test['perdio_cooler'] = (test['cool_lost']>0).astype('int8')
test['ratio_anio']    = (test['lag1']/(test['v_anio_ant']+1)).astype('float32')

for col in ['territory_d','comercial_subchannel_d','rtm_customer_size_d']:
    if col not in test.columns:
        test[col] = 0
    elif test[col].dtype == object:
        known = set(le_dict[col].classes_)
        test[col] = test[col].fillna('Desconocido').astype(str)
        test[col] = test[col].apply(lambda x: x if x in known else 'Desconocido')
        test[col] = le_dict[col].transform(test[col])
    else:
        test[col] = test[col].fillna(0).astype(int)

X_te = test[FEATURES].fillna(0)
print(f'  Test: {X_te.shape} | Nulos: {X_te.isnull().sum().sum()}')

# ============================================================
# PASO 8 - PREDICCION Y OUTPUTS
# ============================================================
print('\n' + '='*55)
print('PASO 8: Predicciones y outputs...')
print('='*55)

probs = model.predict_proba(X_te)[:,1]

# Submission 0/1
# Clientes con ventas=0 en feb = churn seguro
# Modelo captura esto con probabilidad muy alta
n_churners = max(int(len(sub)*CHURN_RATE), 500)
prob_map   = dict(zip(test['customer_id'], probs))
top_ids    = set(pd.Series(prob_map).nlargest(n_churners).index)
sub['target'] = sub['customer_id'].apply(lambda x: 1 if x in top_ids else 0)

# print(f'  Churners con ventas=0 en feb:   {(test["uni_boxes_sold_m"]==0).sum():,}')
print(f'  Churners predichos (1):         {sub["target"].sum():,} ({sub["target"].mean()*100:.2f}%)')
sub[['customer_id','target']].to_csv('preds_submission_final.csv', index=False)
print('  Guardado: preds_submission_final.csv')

# Scoring comercial
scoring = pd.DataFrame({
    'customer_id':        test['customer_id'],
    'probabilidad_churn': probs,
    'target_01':          sub.set_index('customer_id').reindex(
                            test['customer_id'].values)['target'].values
})
scoring['nivel_riesgo'] = 'Bajo'
scoring.loc[scoring['probabilidad_churn']>=np.percentile(probs,93),'nivel_riesgo'] = 'Medio'
scoring.loc[scoring['probabilidad_churn']>=np.percentile(probs,97),'nivel_riesgo'] = 'Alto'
scoring['prob_pct'] = (scoring['probabilidad_churn']*100).round(2)
scoring = scoring.sort_values('probabilidad_churn', ascending=False).reset_index(drop=True)
scoring.to_csv('scoring_comercial.csv', index=False)
print('  Guardado: scoring_comercial.csv')

print(f'\n  Distribucion de riesgo:')
for nivel in ['Alto','Medio','Bajo']:
    n = (scoring['nivel_riesgo']==nivel).sum()
    print(f'    {nivel:5s}: {n:>7,} clientes ({n/len(scoring)*100:.1f}%)')

elapsed = (time.time()-t0)/60
print(f'\n{"="*55}')
print(f'  PIPELINE COMPLETO')
print(f'  PR-AUC validacion: {pr_auc:.4f}')
print(f'  ROC-AUC:           {roc:.4f}')
print(f'  Tiempo total:      {elapsed:.1f} min')
print(f'  preds_submission_final.csv -> listo para Arca')
print(f'  scoring_comercial.csv      -> listo para ventas')
print(f'{"="*55}')
