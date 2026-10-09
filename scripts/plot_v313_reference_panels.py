#!/usr/bin/env python3
"""Reference-style recorded ablation and checked cost panels; no inference."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from survot_rank.evidence.reference_panels import recorded_ablation, render_ablation, render_tradeoff


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('ablation');a.add_argument('--input',type=Path)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('tradeoff');a.add_argument('--input',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='ablation':
            data=recorded_ablation(ROOT) if args.input is None else json.loads(args.input.read_text(encoding='utf-8'))
            render_ablation(data,args.output)
        else:
            data=json.loads(args.input.read_text(encoding='utf-8'))
            render_tradeoff(data,args.input.parent,args.output)
    except (ValueError,OSError,KeyError,TypeError,StopIteration) as error:
        parser.exit(1,f'[reference panels] {error}\nNo model was run.\n')
    print(f'[reference panels] saved checked figures to {args.output}')


if __name__=='__main__':main()
