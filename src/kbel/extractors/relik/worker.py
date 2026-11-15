# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""ReLiK worker: runs in the interpreter that has `relik` installed.

Standalone on purpose (standard library + relik only), since ReLiK pins old
versions of torch and transformers. It reads one JSON request per line on
stdin and writes one JSON response per line on the original stdout;
everything else the libraries print goes to stderr.

    request:  {"id": 1, "texts": ["..."], "options": {"top_k": 100}}
    response: {"id": 1, "outputs": [{"spans": [...], "triplets": [...]}]}
              {"id": 1, "error": "...", "type": "ValueError"}
"""

import argparse
import json
import os
import sys
import traceback


def span(s):
    return {'start': s.start, 'end': s.end, 'text': s.text, 'label': s.label}


def serialize(output):
    triplets = getattr(output, 'triplets', None) or getattr(output, 'triples', None) or []
    return {
        'spans': [span(s) for s in output.spans or []],
        'triplets': [{'subject': span(t.subject), 'label': t.label,
                      'object': span(t.object),
                      'confidence': getattr(t, 'confidence', None)} for t in triplets],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', default=None)
    parser.add_argument('--options', default='{}', help='JSON kwargs of from_pretrained')
    args = parser.parse_args()

    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1, encoding='utf-8')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def send(message):
        protocol.write(json.dumps(message, ensure_ascii=False) + '\n')
        protocol.flush()

    try:
        from relik import Relik
        kwargs = {'device': args.device, **json.loads(args.options)}
        if args.precision:
            kwargs['precision'] = args.precision
        relik = Relik.from_pretrained(args.model, **kwargs)
    except Exception as e:
        send({'ready': False, 'error': str(e), 'type': type(e).__name__,
              'traceback': traceback.format_exc()})
        return 1
    send({'ready': True, 'model': args.model})

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            texts = request['texts']
            outputs = relik(texts, **request.get('options', {})) if texts else []
            if not isinstance(outputs, list):
                outputs = [outputs]
            outputs = sorted(outputs, key=lambda o: o.id) if len(outputs) > 1 else outputs
            send({'id': request['id'], 'outputs': [serialize(o) for o in outputs]})
        except Exception as e:
            send({'id': request.get('id'), 'error': str(e), 'type': type(e).__name__,
                  'traceback': traceback.format_exc()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
