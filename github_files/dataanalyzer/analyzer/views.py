from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse
from django.core.paginator import Paginator

from .forms import RegisterForm
from .models import Dataset

import pandas as pd
import os
import json
import re
import logging
import matplotlib.pyplot as plt
import ollama
from .ai_analyzer import profile_dataset, analyze_dataset_with_ai, render_charts, answer_dataset_question

logger = logging.getLogger(__name__)

# Simple in-process cache: {dataset_id: analysis_dict}
_analysis_cache = {}
_ollama_working = None


def _read_file(file_path):
    """Read CSV/Excel with multi-encoding and comment-line fallback."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
            for comment in (None, '#'):
                try:
                    return pd.read_csv(file_path, encoding=enc, comment=comment)
                except Exception:
                    continue
        raise ValueError('Could not decode CSV with any supported encoding.')
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(file_path)
    raise ValueError(f'Unsupported file format: {ext}')


# =========================================================
# HOME
# =========================================================

def home(request):
    return render(request, 'home.html')


# =========================================================
# REGISTER
# =========================================================

def register(request):

    if request.method == 'POST':

        form = RegisterForm(request.POST)

        if form.is_valid():

            user = form.save(commit=False)

            user.set_password(
                form.cleaned_data['password']
            )

            user.save()

            return redirect('login')

    else:

        form = RegisterForm()

    return render(
        request,
        'register.html',
        {'form': form}
    )


# =========================================================
# LOGIN
# =========================================================

def user_login(request):

    if request.method == 'POST':

        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(request, user)

            return redirect('dashboard')

        else:

            return render(
                request,
                'login.html',
                {
                    'error': 'Invalid username or password'
                }
            )

    return render(
        request,
        'login.html'
    )


# =========================================================
# DASHBOARD
# =========================================================

@login_required
def dashboard(request):

    search = request.GET.get('search', '').strip()

    # Get all datasets belonging to the logged-in user
    all_datasets = Dataset.objects.filter(
        user=request.user
    )

    # Search datasets
    datasets_qs = all_datasets

    if search:
        datasets_qs = datasets_qs.filter(
            name__icontains=search
        )

    paginator = Paginator(datasets_qs.order_by('-uploaded_at'), 8)
    page_number = request.GET.get('page', 1)
    datasets = paginator.get_page(page_number)

    # Total number of datasets
    total_datasets = all_datasets.count()

    # Default KPI values
    total_rows = 0
    total_columns = 0

    # Get latest dataset
    latest_dataset = all_datasets.order_by(
        '-uploaded_at'
    ).first()

    # Read latest dataset and calculate rows/columns
    if latest_dataset:
        try:
            df = _read_file(latest_dataset.file.path)
            total_rows = len(df)
            total_columns = len(df.columns)
        except Exception:
            total_rows = 0
            total_columns = 0

    return render(
        request,
        'dashboard.html',
        {
            'datasets': datasets,
            'search': search,
            'total_datasets': total_datasets,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'latest_dataset': latest_dataset,
            'page_obj': datasets,
        }
    )
# =========================================================
# LOGOUT
# =========================================================

@login_required
def user_logout(request):
    if request.method == 'POST':
        logout(request)
    return redirect('login')


# =========================================================
# UPLOAD DATASET
# =========================================================



@login_required
def upload_dataset(request):

    if request.method == 'POST':

        name = request.POST.get('name', '').strip()
        uploaded_file = request.FILES.get('file')

        # Check dataset name
        if not name:
            return render(
                request,
                'upload.html',
                {
                    'error':
                    'Please enter a dataset name.'
                }
            )

        # Check file
        if not uploaded_file:
            return render(
                request,
                'upload.html',
                {
                    'error':
                    'Please select a file.'
                }
            )

        # Check empty file
        if uploaded_file.size == 0:
            return render(
                request,
                'upload.html',
                {
                    'error':
                    'The uploaded file is empty. '
                    'Please select a valid dataset.'
                }
            )

        # Check file size
        max_size = 10 * 1024 * 1024  # 10 MB

        if uploaded_file.size > max_size:
            return render(
                request,
                'upload.html',
                {
                    'error':
                    'File is too large. '
                    'Maximum allowed size is 10 MB.'
                }
            )

        # Allowed extensions
        allowed_extensions = [
            '.csv',
            '.xlsx',
            '.xls'
        ]

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        if file_extension not in allowed_extensions:
            return render(
                request,
                'upload.html',
                {
                    'error':
                    'Invalid file type. '
                    'Please upload a CSV or Excel file.'
                }
            )

        # Validate actual file contents
        try:

            if file_extension == '.csv':
                uploaded_file.seek(0)
                raw = uploaded_file.read()
                df = None
                for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
                    for comment in (None, '#'):
                        try:
                            import io
                            df = pd.read_csv(io.BytesIO(raw), nrows=5, encoding=enc, comment=comment)
                            break
                        except Exception:
                            continue
                    if df is not None:
                        break
                if df is None:
                    raise ValueError('Could not decode CSV.')
            else:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, nrows=5)

            # Check whether the file contains columns
            if len(df.columns) == 0:
                return render(
                    request,
                    'upload.html',
                    {
                        'error':
                        'The uploaded file does not contain '
                        'valid columns.'
                    }
                )

        except Exception:

            return render(
                request,
                'upload.html',
                {
                    'error':
                    'The uploaded file could not be read. '
                    'Please upload a valid CSV or Excel dataset.'
                }
            )

        # Reset file pointer before saving
        uploaded_file.seek(0)

        # Save dataset
        Dataset.objects.create(
            user=request.user,
            name=name,
            file=uploaded_file
        )

        messages.success(request, f'Dataset "{name}" uploaded successfully!')
        return redirect('dashboard')

    return render(
        request,
        'upload.html'
    )


# =========================================================
# ANALYZE DATASET
# =========================================================
@login_required
def analyze_dataset(request, dataset_id):

    # ---------------------------------------------------------
    # GET DATASET SAFELY
    # ---------------------------------------------------------

    dataset = Dataset.objects.filter(
        id=dataset_id,
        user=request.user
    ).first()

    if not dataset:
        return redirect('dashboard')

    file_path = dataset.file.path

    # ---------------------------------------------------------
    # READ CSV / EXCEL
    # ---------------------------------------------------------

    try:
        df = _read_file(file_path)
    except Exception as e:
        return render(request, 'analysis.html', {
            'dataset': dataset, 'error': f'Could not read dataset: {str(e)}'
        })

    # ---------------------------------------------------------
    # CHECK EMPTY DATASET
    # ---------------------------------------------------------

    if df.empty:

        return render(
            request,
            'analysis.html',
            {
                'dataset': dataset,
                'error': (
                    'The dataset is empty. Please upload a dataset '
                    'containing data.'
                )
            }
        )

    # ---------------------------------------------------------
    # BASIC INFORMATION
    # ---------------------------------------------------------

    rows = len(df)

    columns = len(df.columns)

    column_names = list(df.columns)

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    missing_values = (
        df.isnull()
        .sum()
        .to_dict()
    )

    # Total missing values
    total_missing = int(
        df.isnull().sum().sum()
    )

    # ---------------------------------------------------------
    # DUPLICATE ROWS
    # ---------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    # ---------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------

    data_types = (
        df.dtypes
        .astype(str)
        .to_dict()
    )

    # ---------------------------------------------------------
    # STATISTICAL SUMMARY
    # ---------------------------------------------------------

    try:

        # Only numeric columns for clean statistics
        numeric_df = df.select_dtypes(
            include='number'
        )

        if not numeric_df.empty:

            statistics = (
                numeric_df.describe()
                .round(2)
                .to_html(
                    classes='table',
                    border=0
                )
            )

        else:

            statistics = (
                "No numeric columns available "
                "for statistical summary."
            )

    except Exception:

        statistics = "No statistical summary available."

    # ---------------------------------------------------------
    # DATA PREVIEW WITH PAGINATION
    # ---------------------------------------------------------

    ROWS_PER_PAGE = 10
    try:
        preview_page = int(request.GET.get('preview_page', 1))
    except (ValueError, TypeError):
        preview_page = 1

    total_preview_rows = len(df)
    total_preview_pages = max(1, (total_preview_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
    preview_page = max(1, min(preview_page, total_preview_pages))
    preview_start = (preview_page - 1) * ROWS_PER_PAGE
    preview_end = preview_start + ROWS_PER_PAGE

    try:
        preview_df = df.iloc[preview_start:preview_end]
        preview = preview_df.to_html(
            classes='table preview-table',
            index=False,
            border=0
        )
    except Exception:
        preview = "Data preview is not available."
        total_preview_pages = 1
        preview_page = 1
        total_preview_rows = 0

    preview_start_display = preview_start + 1 if total_preview_rows > 0 else 0
    preview_end_display = min(preview_end, total_preview_rows)

    # Window of page numbers to show (up to 7 numbers around current page)
    w_start = max(1, preview_page - 3)
    w_end = min(total_preview_pages, preview_page + 3)
    preview_page_range = list(range(w_start, w_end + 1))

    # ---------------------------------------------------------
    # CONTEXT
    # ---------------------------------------------------------

    context = {

        'dataset': dataset,

        'rows': rows,

        'columns': columns,

        'column_names': column_names,

        'missing_values': missing_values,

        'total_missing': total_missing,

        'duplicate_rows': duplicate_rows,

        'data_types': data_types,

        'statistics': statistics,

        'preview': preview,
        'preview_page': preview_page,
        'total_preview_pages': total_preview_pages,
        'total_preview_rows': total_preview_rows,
        'rows_per_page': ROWS_PER_PAGE,
        'preview_start_display': preview_start_display,
        'preview_end_display': preview_end_display,
        'preview_page_range': preview_page_range,

    }

    return render(
        request,
        'analysis.html',
        context
    )



# =========================================================
# GENERATE CHARTS
# =========================================================


@login_required
def generate_charts(request, dataset_id):

    dataset = Dataset.objects.filter(
        id=dataset_id, user=request.user
    ).first()
    if not dataset:
        return redirect('dashboard')

    # ---------------------------------------------------------
    # READ DATASET
    # ---------------------------------------------------------
    try:
        df = _read_file(dataset.file.path)
    except Exception as e:
        return render(request, 'charts.html', {
            'dataset': dataset, 'charts': [], 'error': f'Could not read dataset: {e}'
        })

    if df.empty:
        return render(request, 'charts.html', {
            'dataset': dataset, 'charts': [],
            'error': 'The dataset is empty.'
        })

    charts_dir = os.path.join(
        settings.BASE_DIR, 'analyzer', 'static', 'charts'
    )
    if dataset_id not in _analysis_cache:
        _analysis_cache[dataset_id] = analyze_dataset_with_ai(df, dataset.name)
    analysis = _analysis_cache[dataset_id]
    charts = render_charts(df, analysis.get('charts', []), charts_dir, dataset_id)

    return render(request, 'charts.html', {
        'dataset': dataset,
        'charts': charts,
        'summary': analysis.get('dataset_summary', {}),
        'error': '' if charts else 'No charts could be generated for this dataset. The columns may not contain enough numeric or categorical variance.',
    })




# =========================================================
# AI INSIGHTS
# =========================================================

@login_required
def ai_insights(request, dataset_id):

    dataset = Dataset.objects.filter(
        id=dataset_id, user=request.user
    ).first()
    if not dataset:
        return redirect('dashboard')

    try:
        df = _read_file(dataset.file.path)
    except Exception as e:
        return render(request, 'ai_insights.html', {
            'dataset': dataset, 'insights': [], 'error': f'Could not read dataset: {e}'
        })

    if dataset_id not in _analysis_cache:
        _analysis_cache[dataset_id] = analyze_dataset_with_ai(df, dataset.name)
    analysis = _analysis_cache[dataset_id]

    return render(request, 'ai_insights.html', {
        'dataset': dataset,
        'summary': analysis.get('dataset_summary', {}),
        'data_quality': analysis.get('data_quality', []),
        'insights': analysis.get('insights', []),
        'charts': analysis.get('charts', []),
        'suggested_followups': analysis.get('suggested_followups', []),
        'row_count': len(df),
        'col_count': len(df.columns),
        'error': ''
    })


# =========================================================
# ASK AI - QWEN3 + OLLAMA
# =========================================================

@login_required
def ask_ai(request, dataset_id):

    dataset = Dataset.objects.filter(
        id=dataset_id,
        user=request.user
    ).first()

    if not dataset:
        return redirect('dashboard')

    file_path = dataset.file.path

    # =========================================================
    # READ DATASET
    # =========================================================

    try:
        df = _read_file(file_path)
    except Exception as e:
        return render(request, 'ask_ai.html', {
            'dataset': dataset, 'error': f'Could not read dataset: {str(e)}'
        })

    answer = None
    initial_question = request.GET.get('q', '').strip()
    question = initial_question

    # =========================================================
    # USER ASKED A QUESTION
    # =========================================================

    if request.method == 'POST':

        question = request.POST.get(
            'question',
            ''
        ).strip()

        if not question:

            answer = "⚠️ Please enter a question."

        else:

            try:

                # =================================================
                # LEAN DATASET SUMMARY — only key facts
                # =================================================

                total_sales = 0
                total_profit = 0
                summary = []

                summary.append(f"Rows: {len(df):,}")
                summary.append(f"Columns: {', '.join(map(str, df.columns))}")

                if 'SalesAmount' in df.columns:
                    total_sales = df['SalesAmount'].sum()
                    avg_sales = df['SalesAmount'].mean()
                    summary.append(f"Total Sales: ₹{total_sales:,.2f}")
                    summary.append(f"Average Sales: ₹{avg_sales:,.2f}")

                if 'Profit' in df.columns:
                    total_profit = df['Profit'].sum()
                    summary.append(f"Total Profit: ₹{total_profit:,.2f}")

                if 'Quantity' in df.columns:
                    summary.append(f"Total Quantity: {df['Quantity'].sum():,}")

                if total_sales != 0 and total_profit != 0:
                    margin = (total_profit / total_sales) * 100
                    summary.append(f"Profit Margin: {margin:.2f}%")

                if 'Category' in df.columns and 'SalesAmount' in df.columns:
                    cat = (
                        df.groupby('Category')['SalesAmount']
                        .sum().sort_values(ascending=False)
                    )
                    summary.append(
                        "Sales by Category: "
                        + ", ".join(f"{k}: ₹{v:,.0f}" for k, v in cat.items())
                    )

                if 'Region' in df.columns and 'Profit' in df.columns:
                    reg = (
                        df.groupby('Region')['Profit']
                        .sum().sort_values(ascending=False)
                    )
                    summary.append(
                        "Profit by Region: "
                        + ", ".join(f"{k}: ₹{v:,.0f}" for k, v in reg.items())
                    )

                dataset_summary = "\n".join(summary)

                # =================================================
                # QWEN PROMPT — short and direct
                # =================================================

                prompt = (
                    f"Dataset: {dataset.name}\n"
                    f"{dataset_summary}\n\n"
                    f"Question: {question}\n\n"
                    "Reply with ONLY the answer. "
                    "Do NOT write any reasoning, thinking, planning, or explanation. "
                    "Do NOT mention the user or the question. "
                    "Do NOT start with 'The dataset', 'The user', 'I should', 'Let me', or similar. "
                    "Use \u20b9 for money. "
                    "Start your reply directly with the answer."
                )

                # =================================================
                # CALL QWEN3
                # =================================================

                global _ollama_working
                if _ollama_working is not False:
                    # Quick check if Ollama server is actually listening
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.2)
                    is_listening = (sock.connect_ex(('127.0.0.1', 11434)) == 0)
                    sock.close()
                    if not is_listening:
                        _ollama_working = False
                        raise ConnectionRefusedError('Ollama daemon not running on port 11434')

                    ollama_client = ollama.Client(timeout=1.5)
                    response = ollama_client.chat(
                        model='qwen3:4b',
                        messages=[
                            {
                                'role': 'user',
                                'content': prompt
                            }
                        ],
                        think=False,
                        options={
                            'num_predict': 150,
                            'temperature': 0.1,
                        }
                    )

                    answer = response['message'].get('content', '').strip()

                    if '</think>' in answer:
                        answer = answer.split('</think>', 1)[1].strip()
                    elif '<think>' in answer:
                        before_think, after_think = answer.split('<think>', 1)
                        if '</think>' in after_think:
                            answer = after_think.split('</think>', 1)[1].strip()
                        else:
                            answer = before_think.strip()

                    answer = answer.replace('<think>', '').replace('</think>', '').strip()

                    _reasoning_openers = (
                        'hmm,', 'hmm.', 'okay,', 'okay.', 'ok,', 'ok.',
                        'alright,', 'alright.', 'wait,', 'wait.',
                        'let me ', "let's ", "i'll ", 'i will ',
                        'i need to', 'i should', 'i must',
                        'the user ', 'the user\'s',
                        'they want', "they've", 'they asked',
                        'the question asks', 'the question is',
                    )

                    lines = answer.split('\n')
                    clean_lines = []
                    found = False
                    for line in lines:
                        s = line.strip()
                        if not found:
                            if not s:
                                continue
                            if s.lower().startswith(_reasoning_openers):
                                continue
                            found = True
                        clean_lines.append(line)
                    answer = '\n'.join(clean_lines).strip()
                    _ollama_working = True

            except Exception as e:
                logger.info("Ollama unavailable or timed out (%s). Using verified deterministic engine.", e)
                _ollama_working = False
                answer = None

            # Robust fallback: If Ollama failed, timed out, or returned blank, use the deterministic data analytics engine
            if not answer or answer.startswith("⚠️"):
                answer = answer_dataset_question(df, dataset.name, question)

    elif initial_question:
        answer = answer_dataset_question(df, dataset.name, initial_question)
        question = initial_question

    # =========================================================
    # CONTEXT PREPARATION FOR ASK AI
    # =========================================================
    row_count = len(df)
    row_count_fmt = f"{row_count:,}"
    col_count = len(df.columns)
    columns_list = list(df.columns)

    col_lower = [c.lower() for c in df.columns]
    suggested_questions = []

    if any('sale' in c or 'revenue' in c for c in col_lower):
        sales_c = next((c for c in df.columns if 'sale' in c.lower() or 'revenue' in c.lower()), 'sales')
        suggested_questions.append({'icon': '💰', 'label': f'Total {sales_c}', 'query': f'What is the total {sales_c}?'})
        suggested_questions.append({'icon': '⚡', 'label': 'Peak single week/sale', 'query': f'What was the highest single {sales_c}?'})
    elif any('purchase' in c for c in col_lower):
        suggested_questions.append({'icon': '💰', 'label': 'Total purchase', 'query': 'What is the total purchase amount?'})
        suggested_questions.append({'icon': '⚡', 'label': 'Peak purchase', 'query': 'What is the highest purchase?'})

    if any('profit' in c for c in col_lower):
        suggested_questions.append({'icon': '📈', 'label': 'Profit margin', 'query': 'What is the overall profit margin?'})

    if any('store' in c for c in col_lower):
        suggested_questions.append({'icon': '🏆', 'label': 'Best store', 'query': 'Which store has the highest sales?'})
        suggested_questions.append({'icon': '🏪', 'label': 'Store 14 details', 'query': 'Tell me about Store 14'})
    elif any('category' in c for c in col_lower):
        suggested_questions.append({'icon': '🏆', 'label': 'Best category', 'query': 'Which category has the highest sales?'})

    if any('holiday' in c for c in col_lower):
        suggested_questions.append({'icon': '📅', 'label': 'Holiday sales impact', 'query': 'Do holidays increase sales?'})

    if any('date' in c or 'time' in c or 'day' in c for c in col_lower):
        suggested_questions.append({'icon': '📆', 'label': 'Peak sales date', 'query': 'Which date had the highest sales?'})

    if any('temp' in c for c in col_lower):
        suggested_questions.append({'icon': '🌡️', 'label': 'Temperature impact', 'query': 'How does temperature affect sales?'})
    if any('unemploy' in c for c in col_lower):
        suggested_questions.append({'icon': '📉', 'label': 'Unemployment effect', 'query': 'Does unemployment affect sales?'})
    if any('fuel' in c for c in col_lower):
        suggested_questions.append({'icon': '⛽', 'label': 'Fuel price analysis', 'query': 'What is the fuel price range?'})

    suggested_questions.append({'icon': '💡', 'label': 'Growth roadmap', 'query': 'How can I improve sales?'})
    suggested_questions.append({'icon': '📋', 'label': 'Dataset summary', 'query': 'Give me a summary of this dataset'})

    # =========================================================
    # RENDER ASK AI
    # =========================================================

    return render(
        request,
        'ask_ai.html',
        {
            'dataset': dataset,
            'answer': answer,
            'question': question,
            'initial_question': initial_question,
            'row_count': row_count,
            'row_count_fmt': row_count_fmt,
            'col_count': col_count,
            'columns': columns_list[:12],
            'suggested_questions': suggested_questions[:8]
        }
    )
# =========================================================
# AI BUSINESS REPORT
# =========================================================

@login_required
def ai_report(request, dataset_id):

    dataset = Dataset.objects.filter(
        id=dataset_id,
        user=request.user
    ).first()

    if not dataset:
        return redirect('dashboard')

    file_path = dataset.file.path

    try:

        # =====================================================
        # READ DATASET
        # =====================================================

        df = _read_file(file_path)

        analysis = analyze_dataset_with_ai(df, dataset.name)
        summary = analysis.get('dataset_summary', {})
        insights = analysis.get('insights', [])
        dq_list = analysis.get('data_quality', [])

        report_lines = [
            "EXECUTIVE SUMMARY",
            f"Domain: {summary.get('domain', 'Business Analytics')}",
            f"Dataset Overview: {summary.get('description', '')}",
            f"Primary Business Question: {summary.get('key_question', '')}",
            f"Dataset Dimensions: {len(df):,} rows and {len(df.columns)} columns analyzed.",
            "",
            "KEY FINDINGS & MEASURES"
        ]

        for ins in insights:
            if ins.get('type') in ['trend', 'comparison', 'correlation', 'outlier']:
                report_lines.append(f"• {ins.get('title')}: {ins.get('detail')}")

        report_lines.append("")
        report_lines.append("DATA QUALITY & INTEGRITY")
        for dq in dq_list:
            report_lines.append(f"• [{dq.get('severity', 'low').upper()}] {dq.get('column')}: {dq.get('issue')}. Recommendation: {dq.get('suggestion')}")

        rec_insights = [ins for ins in insights if ins.get('type') == 'recommendation']
        if rec_insights:
            report_lines.append("")
            report_lines.append("STRATEGIC RECOMMENDATIONS")
            for rec in rec_insights:
                report_lines.append(f"• {rec.get('title')}: {rec.get('detail')}")

        report = "\n".join(report_lines)

        return render(
            request,
            'ai_report.html',
            {
                'dataset': dataset,
                'report': report,
                'summary': summary,
                'insights': insights,
                'error': ''
            }
        )

    except Exception as e:

        return render(
            request,
            'ai_report.html',
            {
                'dataset': dataset,
                'report': '',
                'error': (
                    'Unable to generate AI report. '
                    f'Error: {str(e)}'
                )
            }
        )


@login_required
def rename_dataset(request, dataset_id):
    if request.method != 'POST':
        return redirect('dashboard')
    dataset = Dataset.objects.filter(id=dataset_id, user=request.user).first()
    if not dataset:
        return redirect('dashboard')
    new_name = request.POST.get('name', '').strip()
    if new_name:
        dataset.name = new_name
        dataset.save()
        messages.success(request, f'Dataset renamed to "{new_name}".')
    return redirect('dashboard')


@login_required
def delete_dataset(request, dataset_id):

    if request.method != 'POST':
        return redirect('dashboard')

    dataset = Dataset.objects.filter(
        id=dataset_id,
        user=request.user
    ).first()

    if not dataset:
        return redirect('dashboard')

    # Delete all chart files for this dataset
    charts_dir = os.path.join(
        settings.BASE_DIR, 'analyzer', 'static', 'charts'
    )
    if os.path.isdir(charts_dir):
        for fname in os.listdir(charts_dir):
            if fname.endswith(f'_{dataset_id}.png'):
                try:
                    os.remove(os.path.join(charts_dir, fname))
                except OSError as e:
                    logger.warning('Could not delete chart %s: %s', fname, e)

    _analysis_cache.pop(dataset_id, None)

    try:
        dataset.file.delete(save=False)
    except OSError as e:
        logger.warning('Could not delete dataset file: %s', e)

    dataset.delete()
    messages.success(request, 'Dataset deleted successfully.')
    return redirect('dashboard')
@login_required
def download_dataset(request, dataset_id):

    dataset = Dataset.objects.filter(
        id=dataset_id,
        user=request.user
    ).first()

    if not dataset:
        return redirect('dashboard')

    try:
        file = dataset.file.open('rb')
        ext = os.path.splitext(dataset.file.name)[1]
        # Sanitize filename — strip path separators and non-safe chars
        safe_name = re.sub(r'[^\w\s\-.]', '', dataset.name).strip()
        safe_filename = (safe_name or 'dataset') + ext
        return FileResponse(
            file,
            as_attachment=True,
            filename=safe_filename
        )

    except FileNotFoundError:
        return redirect('dashboard')

@login_required
def dataset_details(request, dataset_id):

    dataset = Dataset.objects.filter(
        id=dataset_id,
        user=request.user
    ).first()

    if not dataset:
        return redirect('dashboard')

    return render(
        request,
        'dataset_details.html',
        {
            'dataset': dataset
        }
    )


# =========================================================
# PWA (PROGRESSIVE WEB APP) ENDPOINTS
# =========================================================

def manifest(request):
    manifest_path = os.path.join(settings.BASE_DIR, 'analyzer', 'static', 'manifest.json')
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JsonResponse(data, content_type='application/manifest+json')
    except Exception:
        return JsonResponse({
            "name": "AI Data Analyzer",
            "short_name": "DataAnalyzer",
            "start_url": "/",
            "display": "standalone"
        }, content_type='application/manifest+json')


def service_worker(request):
    sw_path = os.path.join(settings.BASE_DIR, 'analyzer', 'static', 'sw.js')
    try:
        with open(sw_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        content = "// sw fallback"
    response = HttpResponse(content, content_type='application/javascript')
    response['Service-Worker-Allowed'] = '/'
    return response


def favicon(request):
    fav_path = os.path.join(settings.BASE_DIR, 'analyzer', 'static', 'icons', 'favicon.ico')
    try:
        with open(fav_path, 'rb') as f:
            content = f.read()
        return HttpResponse(content, content_type='image/x-icon')
    except Exception:
        return HttpResponse(status=404)
