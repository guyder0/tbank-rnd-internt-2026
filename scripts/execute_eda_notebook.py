"""Execute the EDA notebook sequentially without installing a Jupyter kernel.

Cells use ordinary Python. Capture explicit display(), stdout and pyplot.show()
into standard nbformat 4 outputs; also export a standalone HTML view.
Run from the repository root with baseline/.venv/bin/python.
"""
import argparse
import base64
import contextlib
import html
import io
import json
from pathlib import Path
import sys
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebook', nargs='?', type=Path, default=ROOT/'notebooks/user_quality_eda.ipynb')
args = parser.parse_args()
PATH = args.notebook.resolve()
notebook = json.loads(PATH.read_text())
namespace = {'__name__': '__main__'}
outputs = []


def display(value):
    data = {'text/plain': [value.to_string() if hasattr(value, 'to_string') else repr(value)]}
    if hasattr(value, 'to_html'):
        data['text/html'] = [value.to_html()]
    outputs.append({'output_type': 'display_data', 'data': data, 'metadata': {}})


def show(*args, **kwargs):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        stream = io.BytesIO()
        figure.savefig(stream, format='png', dpi=120, bbox_inches='tight')
        outputs.append({'output_type': 'display_data', 'data': {
            'image/png': base64.b64encode(stream.getvalue()).decode(),
            'text/plain': ['Matplotlib figure']}, 'metadata': {}})


namespace['display'] = display
plt.show = show
counter = 0
failure = None
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    counter += 1
    outputs = []
    print(f'Executing cell {counter}...', flush=True)
    stream = io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(''.join(cell['source']), f'{PATH.name}:cell{counter}', 'exec'), namespace)
    except BaseException as error:
        failure = error
        outputs.append({'output_type': 'error', 'ename': type(error).__name__,
                        'evalue': str(error), 'traceback': traceback.format_exc().splitlines()})
    if stream.getvalue():
        outputs.insert(0, {'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(keepends=True)})
    cell['execution_count'] = counter
    cell['outputs'] = outputs
    cell['metadata']['execution_seconds'] = round(time.monotonic()-started, 3)
    notebook['metadata']['execution_method'] = 'Sequential ordinary Python execution; explicit display and pyplot output captured without Jupyter kernel'
    PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    print(f'  {cell["metadata"]["execution_seconds"]} s; {len(outputs)} outputs', flush=True)
    if failure:
        break

parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>KION: качество пользователей</title>',
         '<style>body{font:15px system-ui;margin:30px auto;max-width:1250px;padding:0 24px;color:#202832}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:5px 9px;border:1px solid #dce2e7}img{max-width:100%}section{margin:24px 0}.markdown{white-space:pre-wrap;line-height:1.6}details{margin:15px 0}</style><body>']
for cell in notebook['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'markdown':
        parts.append('<section class="markdown">'+html.escape(source)+'</section>')
        continue
    parts.append('<details><summary>Python cell '+str(cell['execution_count'])+'</summary><pre>'+html.escape(source)+'</pre></details>')
    for output in cell.get('outputs', []):
        if output['output_type'] == 'stream':
            parts.append('<pre>'+html.escape(''.join(output['text']))+'</pre>')
        elif output['output_type'] == 'error':
            parts.append('<pre>'+html.escape('\n'.join(output['traceback']))+'</pre>')
        else:
            data = output.get('data', {})
            if 'image/png' in data:
                parts.append('<img src="data:image/png;base64,'+data['image/png']+'">')
            elif 'text/html' in data:
                parts.append(''.join(data['text/html']))
            else:
                parts.append('<pre>'+html.escape(''.join(data.get('text/plain', [])))+'</pre>')
parts.append('</body></html>')
PATH.with_suffix('.html').write_text('\n'.join(parts))
if failure:
    raise failure
print('Saved:', PATH, PATH.with_suffix('.html'), flush=True)
