//! Every RIP-relative memory operand in the corpus, asked of the relocation
//! table: does the relocation at the displacement's address name its target?
//!
//! A `movsd __xmm@...(%rip),%xmm0` or `leaq .rdata(%rip),%rcx` carries an
//! unrelocated `target - next_ip` in the bytes; the truth is the REL32
//! relocation at `next_ip - 4`. The lifter resolves those addresses ONLY
//! where the table speaks, so the count here is the count of addresses the
//! pipeline is allowed to name — a RIP site with no relocation is a site
//! whose data address can never come from this evidence source.

use rdec_arch_x86::loader;

#[test]
fn every_rip_relative_site_is_named_by_a_relocation() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let man = std::fs::read_to_string(root.join("corpus/manifest.tsv")).unwrap();
    let mut files: Vec<String> = man
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("name\t"))
        .filter_map(|l| l.split('\t').nth(2).map(|s| s.to_string()))
        .collect();
    let cpp = std::fs::read_dir(root.join("corpus/cpp/obj")).unwrap();
    for e in cpp {
        let p = e.unwrap().path();
        if p.extension().map(|x| x == "obj").unwrap_or(false) {
            files.push(
                p.strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    let mut named = 0usize;
    let mut unnamed = 0usize;
    for f in &files {
        let obj = loader::load(&root.join(f)).unwrap();
        for (si, sec) in obj.sections.iter().enumerate() {
            if !obj.is_code(si) {
                continue;
            }
            let mut dec = iced_x86::Decoder::new(64, &sec.bytes, iced_x86::DecoderOptions::NONE);
            while dec.can_decode() {
                let ins = dec.decode();
                for op in 0..ins.op_count() {
                    if ins.op_kind(op) != iced_x86::OpKind::Memory
                        || ins.memory_base() != iced_x86::Register::RIP
                    {
                        continue;
                    }
                    // A RIP disp32 is the instruction's last 4 bytes: that is
                    // the address a REL32 relocation carries.
                    let key = ins.next_ip().wrapping_sub(4);
                    if sec.relocations.iter().any(|r| {
                        r.address == key && r.kind() != rdec_arch_x86::loader::RelocKind::Other
                    }) {
                        named += 1;
                    } else {
                        unnamed += 1;
                        println!(
                            "unnamed rip site: {f} +{:#x} {:?}",
                            ins.ip(),
                            ins.mnemonic()
                        );
                    }
                }
            }
        }
    }
    println!("rip-relative sites: {named} named by a parsed relocation, {unnamed} not named");
    // Every data/code address the decompiler names through RIP comes from a
    // relocation; the corpus has no self-describing data reference.
    assert_eq!(unnamed, 0, "a RIP-relative site the table does not name");
    assert!(named >= 12, "the corpus's data references must be present");
}
