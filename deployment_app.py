"""BESSTIE deployment app — task-aware variety-aware sentiment & sarcasm service.

Imported and driven by scripts/run_deployment.py. Holds the deployment registry
builder, route selection logic, lazy model loaders, the predict function, and the
Gradio app factory, in a single self-contained module.

Usage (called once by run_deployment.py):

    import deployment_app
    deployment_app.setup(
        master_df=load_master_results(),
        train_val_df=train_val_df,
        text_col=TEXT_COL,
        sarcasm_col=SARCASM_COL,
        sentiment_col=SENTIMENT_COL,
        varieties=VARIETIES,
        classical_model_keys=CLASSICAL_MODEL_KEYS,
        is_transformer_model_key=is_transformer_model_key,
        is_plain_transformer_model_key=is_plain_transformer_model_key,
        is_transformer_reddit_key=is_transformer_reddit_key,
        is_transformer_crossvariety_key=is_transformer_crossvariety_key,
        transformer_checkpoint_dir=transformer_checkpoint_dir,
        build_classical_model=build_classical_model,
        load_qlora_adapter=load_qlora_adapter,
        clear_vram=clear_vram,
        lora_model=LORA_MODEL,
        device=device,
        enable_live_qwen=env_flag('ENABLE_LIVE_QWEN_DEPLOYMENT'),
        enable_live_transformer=env_flag('ENABLE_LIVE_TRANSFORMER_DEPLOYMENT'),
    )

After setup() completes:
    deployment_app.deployment_registry_df  -> 13-route dataframe
    deployment_app.predict_text_for_ui     -> the Gradio prediction fn
    deployment_app.deployment_demo         -> the Gradio Interface object
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Public-facing constants (used by run_deployment.py and other callers).
UI_VARIETY_TO_CODE = {
    'Australian English (en-AU)': 'en-AU',
    'Indian English (en-IN)': 'en-IN',
    'British English (en-UK)': 'en-UK',
}
CODE_TO_UI_VARIETY = {v: k for k, v in UI_VARIETY_TO_CODE.items()}

UI_TASK_TO_CODE = {
    'Sarcasm detection': 'sarcasm',
    'Sentiment analysis': 'sentiment',
}
CODE_TO_UI_TASK = {v: k for k, v in UI_TASK_TO_CODE.items()}

MODEL_ROUTE_CHOICES = [
    'Auto best live route for selected task',
    'Sentiment: best overall model',
    'Sentiment: best encoder pooled',
    'Sentiment: DeBERTa pooled',
    'Sentiment: classical fast fallback',
    'Sarcasm: Qwen same-variety adapter',
    'Sarcasm: Qwen pooled adapter',
    'Sarcasm: best encoder pooled',
    'Sarcasm: DeBERTa pooled',
    'Sarcasm: classical fast fallback',
]

TASK_LABELS = {
    'sarcasm': {0: 'Not Sarcastic', 1: 'Sarcastic'},
    'sentiment': {0: 'Negative', 1: 'Positive'},
}
TASK_CLASS1_NAME = {'sarcasm': 'Sarcastic', 'sentiment': 'Positive'}


# Module-level state populated by setup(). Functions below read from these.
_deps: dict = {}
deployment_registry_df: pd.DataFrame | None = None
deployment_demo = None
DEPLOYMENT_MODEL_CACHE: dict = {}


def setup(
    master_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    text_col: str,
    sarcasm_col: str,
    sentiment_col: str,
    varieties: list[str],
    classical_model_keys: list[str],
    is_transformer_model_key,
    is_plain_transformer_model_key,
    is_transformer_reddit_key,
    is_transformer_crossvariety_key,
    transformer_checkpoint_dir,
    build_classical_model,
    load_qlora_adapter,
    clear_vram,
    lora_model: str,
    device: str = 'cuda',
    enable_live_qwen: bool = False,
    enable_live_transformer: bool = False,
):
    """Wire pipeline state into this module and build the deployment registry + Gradio app."""
    global _deps, deployment_registry_df, deployment_demo, DEPLOYMENT_MODEL_CACHE
    DEPLOYMENT_MODEL_CACHE = {}
    _deps = {
        'train_val_df': train_val_df,
        'text_col': text_col,
        'sarcasm_col': sarcasm_col,
        'sentiment_col': sentiment_col,
        'varieties': varieties,
        'classical_model_keys': classical_model_keys,
        'is_transformer_model_key': is_transformer_model_key,
        'is_plain_transformer_model_key': is_plain_transformer_model_key,
        'is_transformer_reddit_key': is_transformer_reddit_key,
        'is_transformer_crossvariety_key': is_transformer_crossvariety_key,
        'transformer_checkpoint_dir': transformer_checkpoint_dir,
        'build_classical_model': build_classical_model,
        'load_qlora_adapter': load_qlora_adapter,
        'clear_vram': clear_vram,
        'lora_model': lora_model,
        'device': device,
        'enable_live_qwen': enable_live_qwen,
        'enable_live_transformer': enable_live_transformer,
        'task_label_cols': {'sarcasm': sarcasm_col, 'sentiment': sentiment_col},
    }
    deployment_registry_df = build_task_specific_deployment_registry(master_df)
    deployment_demo = make_gradio_app()
    return deployment_registry_df


# ---------- registry construction ----------
def _numeric_metric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, np.nan), errors='coerce')


def _best_row(df: pd.DataFrame, mask, sort_cols: list[str]):
    subset = df[mask].copy()
    if subset.empty:
        return None
    for col in sort_cols:
        subset[col] = _numeric_metric(subset, col)
    return subset.sort_values(sort_cols, ascending=False).iloc[0]


def _deployment_artifact_path(row: pd.Series, task: str) -> str:
    model_key = str(row.get('model_key', ''))
    raw_path = row.get('adapter_path', np.nan)
    if pd.notna(raw_path) and str(raw_path).strip() not in {'', 'nan', 'None'}:
        return str(raw_path)
    if _deps['is_transformer_model_key'](model_key):
        seed = int(row['seed']) if pd.notna(row.get('seed', np.nan)) else 42
        if _deps['is_transformer_reddit_key'](model_key):
            return str(_deps['transformer_checkpoint_dir'](model_key, task, seed, scope='reddit_only'))
        if _deps['is_transformer_crossvariety_key'](model_key):
            return str(_deps['transformer_checkpoint_dir'](model_key, task, seed, scope='cross_variety',
                                                           train_variety=str(row.get('trained_on', ''))))
        return str(_deps['transformer_checkpoint_dir'](model_key, task, seed, scope='pooled_all_varieties'))
    if model_key in _deps['classical_model_keys']:
        return f'in_memory_refit::{task}::{model_key}'
    return 'not_applicable'


def _checkpoint_status(row: pd.Series, task: str) -> tuple[str, str]:
    model_key = str(row.get('model_key', ''))
    artifact_path = _deployment_artifact_path(row, task)
    if model_key.startswith('qwen2.5-3b-qlora'):
        path = Path(artifact_path)
        status = 'ready' if path.exists() and (path / 'adapter_config.json').exists() else 'needs_adapter_materialization'
        return ('qwen_adapter' if status == 'ready' else 'metadata_only', status)
    if _deps['is_transformer_model_key'](model_key):
        path = Path(artifact_path)
        status = 'ready' if path.exists() and (path / 'config.json').exists() else 'needs_checkpoint_materialization'
        return ('transformer_checkpoint' if status == 'ready' else 'metadata_only', status)
    if model_key in _deps['classical_model_keys']:
        return ('classical_refit', 'ready_refit_on_startup')
    return ('metadata_only', 'not_applicable')


def _registry_entry(row, route_key, route_label, task, variety_code, serving_strategy):
    backend, artifact_status = _checkpoint_status(row, task)
    artifact_path = _deployment_artifact_path(row, task)
    class1_f1_col = 'test_f1_Sarcastic' if task == 'sarcasm' else 'test_f1_Positive'
    threshold = row.get('threshold', np.nan)
    threshold = 0.5 if pd.isna(threshold) else float(threshold)
    class1_f1 = row.get(class1_f1_col, np.nan)
    class1_f1 = 'not_applicable' if pd.isna(class1_f1) else class1_f1
    return {
        'route_key': route_key,
        'route_label': route_label,
        'task': task,
        'ui_task': CODE_TO_UI_TASK[task],
        'ui_variety': CODE_TO_UI_VARIETY.get(variety_code, 'All varieties / pooled'),
        'variety_code': variety_code if variety_code is not None else 'pooled_all_varieties',
        'model_type': row.get('model_type', ''),
        'model_key': row.get('model_key', ''),
        'model_name': row.get('model_name', ''),
        'trained_on': row.get('trained_on', ''),
        'validated_on': row.get('validated_on', ''),
        'tested_on': row.get('tested_on', ''),
        'seed': int(row['seed']) if pd.notna(row.get('seed', np.nan)) else 42,
        'test_macro_f1': row.get('test_macro_f1', 'not_available'),
        'class1_f1': class1_f1,
        'adapter_path': artifact_path,
        'threshold': threshold,
        'live_backend': backend,
        'artifact_status': artifact_status,
        'serving_strategy': serving_strategy,
    }


def build_task_specific_deployment_registry(master_df: pd.DataFrame) -> pd.DataFrame:
    """Build deployment routes for sentiment and sarcasm, including comparable alternatives."""
    df = master_df.copy()
    df['test_macro_f1'] = _numeric_metric(df, 'test_macro_f1')
    df['test_f1_Sarcastic'] = _numeric_metric(df, 'test_f1_Sarcastic')
    df['test_f1_Positive'] = _numeric_metric(df, 'test_f1_Positive')
    plain_encoder_mask = df['model_key'].astype(str).apply(_deps['is_plain_transformer_model_key'])
    routes = []
    cmk = _deps['classical_model_keys']

    def add(row, **kw):
        if row is not None:
            routes.append(_registry_entry(row, **kw))

    add(_best_row(df, df['task'].astype(str).eq('sentiment'), ['test_macro_f1', 'test_f1_Positive']),
        route_key='sentiment_best_overall', route_label='Sentiment: best overall model',
        task='sentiment', variety_code=None,
        serving_strategy='Use the strongest sentiment model from the registry; fall back to classical if checkpoint absent.')
    add(_best_row(df, (df['task'].astype(str).eq('sentiment')) & plain_encoder_mask
                  & (df['trained_on'].astype(str).eq('pooled_all_varieties')),
                  ['test_macro_f1', 'test_f1_Positive']),
        route_key='sentiment_encoder_pooled', route_label='Sentiment: best encoder pooled',
        task='sentiment', variety_code=None,
        serving_strategy='Best pooled encoder checkpoint among DeBERTa, RoBERTa candidates.')
    add(_best_row(df, (df['task'].astype(str).eq('sentiment'))
                  & (df['model_key'].astype(str).eq('deberta-v3-base'))
                  & (df['trained_on'].astype(str).eq('pooled_all_varieties')),
                  ['test_macro_f1', 'test_f1_Positive']),
        route_key='sentiment_deberta_pooled', route_label='Sentiment: DeBERTa pooled',
        task='sentiment', variety_code=None,
        serving_strategy='Original pooled DeBERTa-v3-base sentiment route kept for comparison.')
    add(_best_row(df, (df['task'].astype(str).eq('sentiment'))
                  & (df['model_key'].astype(str).isin(cmk)),
                  ['test_macro_f1', 'test_f1_Positive']),
        route_key='sentiment_classical_fast', route_label='Sentiment: classical fast fallback',
        task='sentiment', variety_code=None,
        serving_strategy='CPU fallback for live demos when encoder checkpoints are unavailable.')

    for variety in _deps['varieties']:
        add(_best_row(df, (df['task'].astype(str).eq('sarcasm'))
                      & (df['model_key'].astype(str).str.startswith('qwen2.5-3b-qlora'))
                      & (df['trained_on'].astype(str).eq(variety))
                      & (df['tested_on'].astype(str).eq(variety)),
                      ['test_macro_f1', 'test_f1_Sarcastic']),
            route_key=f'sarcasm_qwen_same_{variety}', route_label='Sarcasm: Qwen same-variety adapter',
            task='sarcasm', variety_code=variety,
            serving_strategy='Variety-specific LoRA adapter selected directly from the user variety.')
        add(_best_row(df, (df['task'].astype(str).eq('sarcasm'))
                      & (df['model_key'].astype(str).eq('qwen2.5-3b-qlora'))
                      & (df['trained_on'].astype(str).eq('pooled_all_varieties'))
                      & (df['tested_on'].astype(str).eq(variety)),
                      ['test_macro_f1', 'test_f1_Sarcastic']),
            route_key=f'sarcasm_qwen_pooled_{variety}', route_label='Sarcasm: Qwen pooled adapter',
            task='sarcasm', variety_code=variety,
            serving_strategy='One pooled Qwen LoRA adapter trained on all varieties, evaluated for the selected variety.')

    add(_best_row(df, (df['task'].astype(str).eq('sarcasm')) & plain_encoder_mask
                  & (df['trained_on'].astype(str).eq('pooled_all_varieties')),
                  ['test_macro_f1', 'test_f1_Sarcastic']),
        route_key='sarcasm_encoder_pooled', route_label='Sarcasm: best encoder pooled',
        task='sarcasm', variety_code=None,
        serving_strategy='Best pooled encoder checkpoint among DeBERTa, RoBERTa candidates.')
    add(_best_row(df, (df['task'].astype(str).eq('sarcasm'))
                  & (df['model_key'].astype(str).eq('deberta-v3-base'))
                  & (df['trained_on'].astype(str).eq('pooled_all_varieties')),
                  ['test_macro_f1', 'test_f1_Sarcastic']),
        route_key='sarcasm_deberta_pooled', route_label='Sarcasm: DeBERTa pooled',
        task='sarcasm', variety_code=None,
        serving_strategy='Original pooled DeBERTa-v3-base route for sarcasm comparison.')
    add(_best_row(df, (df['task'].astype(str).eq('sarcasm'))
                  & (df['model_key'].astype(str).isin(cmk)),
                  ['test_macro_f1', 'test_f1_Sarcastic']),
        route_key='sarcasm_classical_fast', route_label='Sarcasm: classical fast fallback',
        task='sarcasm', variety_code=None,
        serving_strategy='CPU fallback; useful for latency comparison and demos without GPU.')

    if not routes:
        raise ValueError('No deployment routes could be built from the master registry.')
    return pd.DataFrame(routes).sort_values(['task', 'route_label', 'variety_code']).reset_index(drop=True)


# ---------- route selection + model loading ----------
def select_deployment_route(task: str, variety_code: str, route_choice: str) -> pd.Series:
    task = UI_TASK_TO_CODE.get(task, task)
    route_choice = str(route_choice)
    reg = deployment_registry_df

    if route_choice == 'Auto best live route for selected task':
        if task == 'sentiment':
            live = reg[(reg['task'] == 'sentiment')
                       & (reg['live_backend'].isin(['transformer_checkpoint', 'classical_refit']))].copy()
            if len(live) == 0:
                live = reg[reg['task'] == 'sentiment'].copy()
            return live.sort_values('test_macro_f1', ascending=False).iloc[0]
        same_key = f'sarcasm_qwen_same_{variety_code}'
        rows = reg[reg['route_key'] == same_key]
        if len(rows) > 0:
            return rows.iloc[0]
        return reg[reg['task'] == 'sarcasm'].sort_values('test_macro_f1', ascending=False).iloc[0]

    route_map = {
        'Sentiment: best overall model': 'sentiment_best_overall',
        'Sentiment: best encoder pooled': 'sentiment_encoder_pooled',
        'Sentiment: DeBERTa pooled': 'sentiment_deberta_pooled',
        'Sentiment: classical fast fallback': 'sentiment_classical_fast',
        'Sarcasm: Qwen same-variety adapter': f'sarcasm_qwen_same_{variety_code}',
        'Sarcasm: Qwen pooled adapter': f'sarcasm_qwen_pooled_{variety_code}',
        'Sarcasm: best encoder pooled': 'sarcasm_encoder_pooled',
        'Sarcasm: DeBERTa pooled': 'sarcasm_deberta_pooled',
        'Sarcasm: classical fast fallback': 'sarcasm_classical_fast',
    }
    route_key = route_map.get(route_choice)
    rows = reg[(reg['route_key'] == route_key) & (reg['task'] == task)]
    if len(rows) == 0:
        print(f'Route "{route_choice}" not valid for task={task}; using auto.')
        return select_deployment_route(task, variety_code, 'Auto best live route for selected task')
    return rows.iloc[0]


def load_classical_deployment_model(row: pd.Series):
    cache_key = f'classical::{row["task"]}::{row["model_key"]}'
    if cache_key not in DEPLOYMENT_MODEL_CACHE:
        task = row['task']
        label_col = _deps['task_label_cols'][task]
        class_weight = 'balanced' if task == 'sarcasm' else None
        model = _deps['build_classical_model'](row['model_key'], class_weight=class_weight, seed=int(row['seed']))
        model.fit(_deps['train_val_df'][_deps['text_col']], _deps['train_val_df'][label_col])
        DEPLOYMENT_MODEL_CACHE[cache_key] = model
    return DEPLOYMENT_MODEL_CACHE[cache_key]


def load_deployment_qwen_adapter(row: pd.Series):
    cache_key = f'qwen::{row["adapter_path"]}'
    if cache_key not in DEPLOYMENT_MODEL_CACHE:
        adapter_path = Path(str(row['adapter_path']))
        if not (adapter_path / 'adapter_config.json').exists():
            raise FileNotFoundError(f'Deployment adapter is missing: {adapter_path}')
        _deps['clear_vram'](verbose=True)
        model, tokenizer, _info = _deps['load_qlora_adapter'](
            _deps['lora_model'], adapter_path, num_labels=2, device=_deps['device'])
        DEPLOYMENT_MODEL_CACHE[cache_key] = (model, tokenizer)
    return DEPLOYMENT_MODEL_CACHE[cache_key]


def load_transformer_deployment_checkpoint(row: pd.Series):
    raw = row.get('adapter_path', np.nan)
    checkpoint_dir = (Path(str(raw)) if pd.notna(raw)
                      else Path(_deployment_artifact_path(row, row['task'])))
    cache_key = f'transformer::{checkpoint_dir}'
    if cache_key not in DEPLOYMENT_MODEL_CACHE:
        if not (checkpoint_dir / 'config.json').exists():
            raise FileNotFoundError(f'Encoder checkpoint is missing: {checkpoint_dir}')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(_deps['device'])
        model.eval()
        DEPLOYMENT_MODEL_CACHE[cache_key] = (model, tokenizer)
    return DEPLOYMENT_MODEL_CACHE[cache_key]


def _predict_with_transformer_like(model, tokenizer, text: str, threshold, task: str):
    encoded = tokenizer(str(text), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids=encoded['input_ids'].to(_deps['device']),
                        attention_mask=encoded['attention_mask'].to(_deps['device']))
        probs = torch.softmax(outputs.logits.float(), dim=-1)[0].detach().cpu().numpy()
    cutoff = 0.5 if threshold is None or pd.isna(threshold) else float(threshold)
    pred = int(probs[1] >= cutoff)
    return TASK_LABELS[task][pred], float(probs[1])


# ---------- public predict + Gradio app ----------
def _predict_text_for_ui_inner(text: str, task_choice: str, variety_choice: str, route_choice: str,
                               threshold_override=None):
    """Inner prediction logic; wrapped by predict_text_for_ui with error handling.

    ``threshold_override`` (0..1), when set, overrides the route's decision threshold for the
    positive class (Sarcastic / Positive) on the transformer-like backends. This exposes the
    precision/recall dial from the Q4 error analysis: lower it (~0.3, the en-AU validation-tuned
    value) to catch more of the rare sarcastic class, raise it (~0.7) to suppress false alarms.
    """
    if not str(text).strip():
        return 'Please enter some text.', 0.0, 'No prediction was made.'

    if deployment_registry_df is None:
        return ('Setup not called', 0.0,
                'deployment_app.setup(...) has not run yet. Run scripts/run_deployment.py '
                '(which calls setup) before predicting.')

    task = UI_TASK_TO_CODE.get(task_choice, task_choice)
    variety_code = UI_VARIETY_TO_CODE.get(variety_choice, variety_choice)
    row = select_deployment_route(task, variety_code, route_choice)
    backend = row['live_backend']
    route_threshold = row.get('threshold', np.nan)
    applied_threshold = (float(threshold_override)
                         if threshold_override not in (None, '')
                         else (0.5 if pd.isna(route_threshold) else float(route_threshold)))

    if backend == 'qwen_adapter':
        if not _deps.get('enable_live_qwen', False):
            return ('Live Qwen disabled', 0.0,
                    f'Selected {row["route_label"]}: trained_on={row["trained_on"]}, seed={row["seed"]}. '
                    'Set ENABLE_LIVE_QWEN_DEPLOYMENT=1 and re-run scripts/run_deployment.py to return live Qwen predictions.')
        model, tokenizer = load_deployment_qwen_adapter(row)
        label, p1 = _predict_with_transformer_like(model, tokenizer, text, applied_threshold, task)
    elif backend == 'transformer_checkpoint':
        if not _deps.get('enable_live_transformer', False):
            return ('Live transformer disabled', 0.0,
                    f'Selected {row["route_label"]}: {row["model_key"]}, seed={row["seed"]}. '
                    'Set ENABLE_LIVE_TRANSFORMER_DEPLOYMENT=1 and re-run scripts/run_deployment.py to return live encoder predictions.')
        model, tokenizer = load_transformer_deployment_checkpoint(row)
        label, p1 = _predict_with_transformer_like(model, tokenizer, text, applied_threshold, task)
    elif backend == 'classical_refit':
        model = load_classical_deployment_model(row)
        pred = int(model.predict([text])[0])
        label = TASK_LABELS[task][pred]
        if hasattr(model, 'predict_proba'):
            p1 = float(model.predict_proba([text])[0, 1])
        elif hasattr(model, 'decision_function'):
            score = float(model.decision_function([text])[0])
            p1 = float(1 / (1 + np.exp(-score)))
        else:
            p1 = float(pred)
    else:
        return ('Model artifact unavailable', 0.0,
                f'The registry best route is {row["model_key"]} ({row["route_label"]}), '
                f'but artifact_status={row["artifact_status"]}. Use a classical fallback or rerun training.')

    thr_note = '' if backend == 'classical_refit' else f'decision_threshold={applied_threshold:.2f}; '
    details = (
        f'Task={task}; route={row["route_label"]}; model={row["model_key"]}; '
        f'trained_on={row["trained_on"]}; tested_on={row["tested_on"]}; seed={row["seed"]}; '
        f'{thr_note}{TASK_CLASS1_NAME[task]} probability={p1:.3f}; registry Macro-F1={float(row["test_macro_f1"]):.3f}'
    )
    return label, p1, details


def predict_text_for_ui(text: str, task_choice: str, variety_choice: str, route_choice: str,
                        threshold_override=None):
    """Prediction function used by the deployment UI. Wraps inner logic so unhandled exceptions
    surface a readable error message in the Gradio UI instead of just 'Error'."""
    try:
        return _predict_text_for_ui_inner(text, task_choice, variety_choice, route_choice, threshold_override)
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=4)
        return (
            f'{type(e).__name__}: {str(e)[:140]}',
            0.0,
            f'inputs: task={task_choice!r} variety={variety_choice!r} route={route_choice!r}\n'
            f'traceback (last 4 frames):\n{tb}'
        )


def predict_sarcasm_for_ui(text: str, variety_choice: str):
    """Backwards-compatible alias used by earlier notes/screenshots."""
    return predict_text_for_ui(text, 'Sarcasm detection', variety_choice, 'Sarcasm: Qwen same-variety adapter')


def make_gradio_app():
    """Build the Gradio Interface object. Returns None if gradio is not installed."""
    try:
        import gradio as gr
    except ImportError:
        print('Gradio is not installed. To launch the UI: pip install gradio')
        return None
    return gr.Interface(
        fn=predict_text_for_ui,
        inputs=[
            gr.Textbox(label='Input text', lines=4, placeholder='Type a review/comment here...'),
            gr.Dropdown(choices=list(UI_TASK_TO_CODE.keys()), label='Task', value='Sarcasm detection'),
            gr.Dropdown(choices=list(UI_VARIETY_TO_CODE.keys()), label='English variety', value='British English (en-UK)'),
            gr.Dropdown(choices=MODEL_ROUTE_CHOICES, label='Model route', value='Auto best live route for selected task'),
            gr.Slider(minimum=0.05, maximum=0.95, step=0.05, value=0.5,
                      label='Sarcastic/Positive decision threshold',
                      info='Lower (~0.30, en-AU validation-tuned) catches more sarcasm; higher (~0.70) cuts false alarms. Applies to Qwen/encoder routes.'),
        ],
        outputs=[
            gr.Textbox(label='Prediction'),
            gr.Number(label='P(class 1: Sarcastic/Positive)'),
            gr.Textbox(label='Model route details'),
        ],
        title='BESSTIE Variety-Aware Sentiment & Sarcasm Service',
        description=('Select a task, English variety, model route, and decision threshold. The threshold '
                     'slider exposes the precision/recall dial from the Q4 error analysis: move it and watch '
                     'borderline sarcasm flip between Not Sarcastic and Sarcastic.'),
    )
