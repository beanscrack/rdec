#!/usr/bin/env python3
"""COFF relocation reader for objects objdump cannot read (MSVC `.obj`).

`objdump -r` prints NOTHING for an MSVC x64 object — binutils refuses the file.
This reads the COFF tables directly, which is the ground truth the roadmap's
"data symbol address space" decision has to be made against: in a COFF object
there are no image addresses at all. A relocation is (section, offset, symbol,
symbol-section, addend), and a symbol's "address" is its section plus an offset
inside it. CodeView puts its own relocations in the SAME array, so anything that
names an address has to say which target sections it accepts.

  python scripts/coff-relocs.py corpus/msvc/obj/array_walk.O2.obj --at 0x2f 0x45
  python scripts/coff-relocs.py corpus/obj/array_walk.O2.obj --summary

`--at LO HI` lists the relocations of every code section whose RVA is in range;
`--summary` prints one line per section and the relocation type histogram.
"""
import struct
import sys

RELOC_TYPES = {
    0: "ABS", 1: "REL32", 2: "SECTION", 3: "ADDR64", 4: "SECTIONREL",
    5: "DIR32", 6: "DIR32NB", 7: "REL32FRAME", 9: "SECREL", 20: "DIR64",
}


def parse(path):
    b = open(path, "rb").read()
    if len(b) < 20:
        sys.exit(f"{path}: too short to be a COFF object")
    magic, nsec, _tstamp, psym, nsym = struct.unpack_from("<HHIII", b, 0)
    if magic != 0x8664:
        sys.exit(f"{path}: not an AMD64 COFF object (magic {magic:#x})")
    opthdr = struct.unpack_from("<H", b, 16)[0]
    secs = []
    off = 20 + opthdr
    for _ in range(nsec):
        name = b[off:off + 8].rstrip(b"\0").decode("latin1")
        vsize, va, rawsize, praw, prp, _pln = struct.unpack_from("<IIIIII", b, off + 8)
        nreloc = struct.unpack_from("<H", b, off + 32)[0]
        secs.append(dict(name=name, va=va, vsize=vsize, raw=praw, rp=prp, nr=nreloc))
        off += 40
    strtop = psym + 20 * nsym
    strsz = struct.unpack_from("<I", b, strtop)[0]
    strtab = b[strtop:strtop + strsz]

    def sym(idx):
        so = psym + 20 * idx
        short = b[so:so + 8]
        if short[:4] == b"\0\0\0\0":
            o = struct.unpack_from("<I", b, so + 4)[0]
            name = strtab[o:strtab.find(b"\0", o)].decode("latin1")
        else:
            name = short.rstrip(b"\0").decode("latin1")
        value = struct.unpack_from("<I", b, so + 8)[0]
        secn = struct.unpack_from("<h", b, so + 12)[0]
        sec = secs[secn - 1]["name"] if 0 < secn <= len(secs) else {
            -1: "ABS", -2: "DEBUG", 0: "UNDEF"}.get(secn, f"sec#{secn}")
        return name, sec, value

    return secs, sym


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    path = args[0]
    at = summary = False
    lo = hi = 0
    i = 1
    while i < len(args):
        if args[i] == "--at":
            at = True
            lo, hi = int(args[i + 1], 0), int(args[i + 2], 0)
            i += 3
        elif args[i] == "--summary":
            summary = True
            i += 1
        else:
            i += 1
    secs, sym = parse(path)
    b = open(path, "rb").read()
    code = [s for s in secs if s["name"].startswith(".text")]
    if summary:
        print(f"{path}: {len(secs)} sections, {len(code)} code sections")
        hist = {}
        for s in secs:
            for k in range(s["nr"]):
                _v, _si, typ = struct.unpack_from("<IIH", b, s["rp"] + 10 * k)
                key = RELOC_TYPES.get(typ, str(typ))
                hist[key] = hist.get(key, 0) + 1
        for s in secs:
            print(f"  {s['name']:<14} vsize={s['vsize']:<7} relocs={s['nr']}")
        print("  relocation types:", ", ".join(f"{k}={v}" for k, v in sorted(hist.items())))
        return
    if at is None:
        sys.exit("give --at LO HI or --summary")
    rng = (lo, hi)
    b = open(path, "rb").read()
    for s in code:
        for k in range(s["nr"]):
            rva, symi, typ = struct.unpack_from("<IIH", b, s["rp"] + 10 * k)
            if not (rng[0] <= rva <= rng[1]):
                continue
            name, ssec, val = sym(symi)
            debug = ssec.startswith(".debug")
            print(f"  {s['name']} rva={rva:#06x} {RELOC_TYPES.get(typ, typ):<10} "
                  f"sym={name!r} -> section {ssec} value={val:#x}"
                  f"{'   [CodeView: not a data reference]' if debug else ''}")


main()
