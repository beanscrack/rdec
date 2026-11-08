#!/usr/bin/env python3
"""Regenerate corpus/truth.tsv from corpus/src/*.c (M6 type gate truth).

Extracts every TOP-LEVEL definition (line starts at column 0, ends with `{`),
normalizes each declared type to a truth token, and prints
`name<TAB>ret<TAB>params;params`. Comments and string literals are stripped
first; array parameters decay to `px`; struct returns/params become `s2/s4`
by the corpus's own convention (the corpus uses only _Float32x2/_Float32x4).
The gate reads the generated TSV, never parses C — this script is the audit
recipe, run by hand when the corpus grows, and the diff is reviewed like code.
"""
import re, sys, os

def strip(t):
    t = re.sub(r'/\*.*?\*/', ' ', t, flags=re.S)
    t = re.sub(r'//[^\n]*', '', t)
    t = re.sub(r'"(\\.|[^"\\])*"', '"S"', t)
    t = re.sub(r"'(\\.|[^'\\])*'", "'c'", t)
    return t

M = {'void':'void','int':'i32','unsigned int':'u32','unsigned':'u32',
     'signed int':'i32','long long':'i64','long':'i64',
     'unsigned long long':'u64','unsigned long':'u64','short':'i16',
     'unsigned short':'u16','signed char':'i8','unsigned char':'u8','char':'i8',
     'uint64_t':'u64','int64_t':'i64','uint32_t':'u32','int32_t':'i32',
     'uint16_t':'u16','int16_t':'i16','uint8_t':'u8','int8_t':'i8',
     'float':'f32','double':'f64','size_t':'u64','long int':'i64',
     'signed long':'i64','bool':'i8','_Float32x2':'s2','_Float32x4':'s4',
     '__m64':'mmx'}

def norm(s):
    s = ' '.join(s.replace('const', '').split())
    if s.endswith('[]'):
        return 'px'
    if s.endswith('**'):
        return 'ppx'
    if s.endswith('*'):
        return 'px'
    if s in M:
        return M[s]
    raise SystemExit(f'truth_extract: unmapped type {s!r} — extend the table')

def main(root):
    print('# M6 type-recovery truth: the declared signature of every corpus source.')
    print('# Generated from corpus/src/*.c top-level definitions by scripts/truth_extract.py,')
    print('# audited by hand. Tokens: void i8 u8 i16 u16 i32 u32 i64 u64 px ppx f32 f64 s2 s4 mmx;')
    print('# array params decay to px.  name<TAB>ret<TAB>params (semicolon separated).')
    print('name\tret\tparams')
    srcdir = os.path.join(root, 'src')
    for path in sorted(os.listdir(srcdir)):
        if not path.endswith('.c'):
            continue
        raw = open(os.path.join(srcdir, path), encoding='utf-8',
                   errors='replace').read().replace('\r\n', '\n').replace('\r', '\n')
        t = strip(raw)
        stem = path[:-2]
        for m in re.finditer(r'(?m)^((?:[A-Za-z_]\w*\s|unsigned\s|signed\s|\*)+?)\**\s*([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{', t):
            rt, name, ps = m.group(1).strip(), m.group(2), m.group(3).strip()
            pars = []
            if ps and ps != 'void':
                depth, cur, parts = 0, '', []
                for ch in ps:
                    depth += ch in '(['
                    depth -= ch in ')]'
                    if ch == ',' and depth == 0:
                        parts.append(cur); cur = ''
                    else:
                        cur += ch
                parts.append(cur)
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    mm = re.search(r'([A-Za-z_]\w*)\s*$', p)
                    pars.append(norm(p[:mm.start()].strip() if mm else p))
            print(f'{stem}\t{norm(rt)}\t{";".join(pars)}')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'corpus')
