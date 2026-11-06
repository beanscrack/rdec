//! M1 step 2: recursive-descent discovery.
//!
//! Seeds are entry points (symbols) and the targets of `REL32` call
//! relocations. From a seed we decode until a terminator, enqueuing direct
//! branch targets as blocks and direct call targets as *function* seeds.
//! Anything we cannot follow - indirect jump, indirect call, undecodable byte,
//! target outside the code range - is recorded as a named gap, because the size
//! of that list *is* the coverage claim.
//!
//! Function starts and block starts are kept separate: a branch target is a
//! block, not a function. Conflating them is how a decompiler invents functions
//! inside a switch statement.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// Bytes claimed by a sweep: address -> instruction length. This is the
/// invariant that makes overlap impossible: a byte is claimed once, and a sweep
/// that reaches an already-claimed address stops there - that address is a
/// block boundary by construction. Claiming per-sweep, and stopping only at
/// starts known *at the time*, is what let a function body sweep past a branch
/// target that a later sweep then decoded again.
pub type Decoded = HashMap<u64, usize>;

use crate::decode::{decode, Syntax};
use crate::loader::Object;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gap {
    /// `jmp *%rax`, `jmp [tbl+rcx*8]` with no relocation evidence behind it: the
    /// target set is not in the bytes and not in the container. A site that
    /// *does* have a fully resolved table behind it is not a gap at all.
    IndirectBranch,
    /// An indirect jump whose table resolved only in part: the entries that
    /// resolved are discovered, the ones that did not are counted here, so a
    /// half-recovered table is reported as half-recovered and not as success.
    PartialTable,
    IndirectCall,
    /// The decoder refused here.
    Undecodable,
    /// A direct branch to an address outside every code range we were given.
    OutsideCode,
}

#[derive(Clone, Copy, Debug)]
pub struct Site {
    pub va: u64,
    pub gap: Gap,
}

/// One computed-jump table recovered from the container, not from the bytes.
///
/// The evidence, and all of it is the object's own:
///
/// * a `REL32` relocation in a code section names a symbol in a DATA section -
///   that is the instruction (`lea (%rip),%reg`) that materialises the table's
///   address;
/// * the object bytes at that symbol are 4-byte slots, and each slot is itself
///   covered by a `REL32` relocation whose symbol is in a code section;
/// * the dispatch is the first indirect `jmp` after the instruction that carries
///   the naming relocation, with no terminator in between - which is what makes
///   the data object a *jump table* and not a pointer that a call reads.
///
/// `entries` is derived from the consecutive relocation-covered slots, never
/// hardcoded, so an 8-entry table and a 4-entry table are the same rule.
#[derive(Clone, Debug)]
pub struct JumpTable {
    /// The `jmp *...` that dispatches through this table.
    pub site: u64,
    /// Section the table lives in (`.rdata` for clang's switch tables).
    pub base_section: String,
    /// Offset of the table's first entry within that section.
    pub base_offset: u64,
    /// Slots in the run: consecutive 4-byte entries carrying a REL32.
    pub entries: usize,
    /// Resolved targets, in table order, deduplicated.
    pub targets: Vec<u64>,
    /// (slot offset in the section, why it failed) for every slot that did not
    /// resolve. A table is only as good as this list being empty.
    pub unresolved: Vec<(u64, String)>,
}

impl JumpTable {
    /// Fully recovered: at least one target, nothing left unresolved.
    pub fn complete(&self) -> bool {
        self.unresolved.is_empty() && !self.targets.is_empty()
    }
}

/// Resolved tables, keyed by the address of the indirect jump they feed.
pub type Tables = BTreeMap<u64, JumpTable>;

/// The COFF `REL32` arithmetic for a jump-table slot.
///
/// In an unlinked object every section's own offset 0 is its origin, and the
/// bytes at a `REL32` site are the addend: the value the linker writes is
/// `symbol + addend - (site + 4)`. A clang switch table's dispatch is
/// `movslq (%base,%idx,4),%tmp; add %tmp,%base; jmp *%base`, i.e. the runtime
/// target is `table_base + slot_value`, so the target's offset *inside its own
/// section* is
///
/// ```text
/// sym_offset + addend - (slot_offset - table_base_offset) - 4
/// ```
///
/// Measured on `corpus/obj/switch_dense.O0.obj`: entry 0 is addend `0x2e` at
/// slot 0 of a table based at `.rdata+0`, naming `.text+0`, giving
/// `0 + 0x2e - 0 - 4 = 0x2a` - the first case body in `llvm-objdump -d`; entry 7
/// is `0x90 - 28 - 4 = 0x70` - the last. `tests/jump_tables.rs` pins all twelve
/// corpus entries against objdump's printed bodies.
fn table_target(sym_offset: u64, addend: i32, slot: u64, table_base: u64) -> i64 {
    sym_offset as i64 + addend as i64 - (slot - table_base) as i64 - 4
}

/// Find the indirect `jmp` that dispatches through a table, starting from the
/// instruction carrying the relocation that names the table's address.
///
/// The scan is bounded by the control flow: a `ret`, a direct `jmp`/`jcc` or a
/// decoder refusal before the indirect jump means the data object's address was
/// materialised for some other reason (a vtable pointer a `call` reads, a
/// static's address), and the site stays a counted gap.
fn dispatch_site(bytes: &[u8], code_base: u64, disp: u64) -> Option<u64> {
    let end = (disp as usize + 4).min(bytes.len());
    let prefix = decode(&bytes[..end], code_base, Syntax::Intel);
    let at = prefix.iter().rev().find(|i| {
        i.valid() && i.va <= code_base + disp && code_base + disp < i.va + i.len as u64
    })?;
    let start = (at.va - code_base) as usize;
    let fwd = decode(&bytes[start..], at.va, Syntax::Intel);
    for i in fwd.iter().take(16) {
        if !i.valid() {
            return None;
        }
        if i.is_indirect && !i.is_call {
            return Some(i.va);
        }
        if i.is_ret || ((i.is_jmp || i.is_jcc) && i.target.is_some()) {
            return None;
        }
    }
    None
}

/// Every jump table in an object, keyed by dispatch site, addressed in the
/// caller's flat per-object space (`base_of`: section index -> that section's
/// base, exactly as every gate builds it for its code ranges).
///
/// A table with no relocation naming it, a table whose slots are 8-byte
/// `ADDR64` entries (a C++ vtable), and a table whose run stops before two
/// entries all produce nothing here, which leaves the indirect jump in the
/// counted `Gap::IndirectBranch` bucket.
pub fn jump_tables(obj: &Object, base_of: &dyn Fn(usize) -> Option<u64>) -> Tables {
    let mut out: Tables = BTreeMap::new();
    for (csi, sec) in obj.sections.iter().enumerate() {
        if !obj.is_code(csi) {
            continue;
        }
        let Some(code_base) = base_of(csi) else {
            continue;
        };
        for rl in &sec.relocations {
            if !rl.kind().is_rel32() {
                continue;
            }
            let Some(base_sym) = obj.symbol_of(rl.symbol) else {
                continue;
            };
            let Some(dsi) = base_sym.section else {
                continue;
            };
            if dsi == csi || obj.is_code(dsi) {
                continue; // a call/lea of code, not a data object's address
            }
            let dsec = &obj.sections[dsi];
            // The run: consecutive 4-byte slots, each covered by a REL32.
            let base_off = base_sym.offset;
            let mut slots: Vec<(u64, usize, i32)> = Vec::new();
            loop {
                let off = base_off + 4 * slots.len() as u64;
                let Some(w) = dsec.bytes.get(off as usize..off as usize + 4) else {
                    break;
                };
                let Some(er) = obj.reloc_at(dsi, off) else {
                    break;
                };
                if !er.kind().is_rel32() {
                    break;
                }
                slots.push((off, er.symbol, i32::from_le_bytes([w[0], w[1], w[2], w[3]])));
                // A relocation run this long is a data object, not a switch:
                // clang's dense tables are tens of entries, and a table of
                // thousands means we are reading a struct of pointers.
                if slots.len() >= 4096 {
                    break;
                }
            }
            // One entry is a pointer, two is the smallest computed jump.
            if slots.len() < 2 {
                continue;
            }
            let Some(site) = dispatch_site(&sec.bytes, code_base, rl.address) else {
                continue;
            };
            let t = out.entry(site).or_insert_with(|| JumpTable {
                site,
                base_section: dsec.name.clone(),
                base_offset: base_off,
                entries: 0,
                targets: Vec::new(),
                unresolved: Vec::new(),
            });
            t.entries = t.entries.max(slots.len());
            for (off, sym_idx, addend) in &slots {
                let sym = obj.symbol_of(*sym_idx);
                let name = sym.map(|s| s.name.as_str()).unwrap_or("?");
                // `t` is already `&mut JumpTable` from `last_mut()`; passing it
                // by value moves the reborrow, and the closure writes through it.
                let fail = |t: &mut JumpTable, why: String| {
                    if !t.unresolved.iter().any(|(o, _)| o == off) {
                        t.unresolved.push((*off, why));
                    }
                };
                let Some(s) = sym else {
                    fail(t, format!("slot {off:#x} names a symbol this object lacks"));
                    continue;
                };
                let Some(tsi) = s.section else {
                    fail(t, format!("slot {off:#x} names `{name}`, undefined here"));
                    continue;
                };
                let Some(tb) = base_of(tsi) else {
                    fail(
                        t,
                        format!("slot {off:#x} names `{name}`, not in a code section"),
                    );
                    continue;
                };
                let rel = table_target(s.offset, *addend, *off, base_off);
                let lo = tb as i64;
                let hi = tb as i64 + obj.sections[tsi].bytes.len() as i64;
                if rel < lo || rel >= hi {
                    fail(
                        t,
                        format!("slot {off:#x} of `{name}` resolves to {rel:#x}, outside the code"),
                    );
                    continue;
                }
                let va = rel as u64;
                if !t.targets.contains(&va) {
                    t.targets.push(va);
                }
            }
            t.targets.sort_unstable();
            t.unresolved.sort_by_key(|(o, _)| *o);
        }
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Addresses proven to be function entry points (sorted, deduped).
    pub functions: Vec<u64>,
    /// Block starts reached by descent (sorted, deduped).
    pub blocks: Vec<u64>,
    /// Bytes proven to be instructions: (address, length), sorted.
    pub insns: Vec<(u64, usize)>,
    /// Every direct branch destination inside the code, sorted - including ones
    /// that land *inside* a block the sweep already built, which is why this is
    /// not the same set as `blocks`. A backward `jmp` to the middle of a decoded
    /// block claims no new bytes (the claim invariant forbids re-decoding them),
    /// so it never reaches `blocks`; yet a basic block must end there, or the loop
    /// header has no identity, no join, and no phi. Lifting splits on these.
    pub targets: Vec<u64>,
    pub gaps: Vec<Site>,
    /// The indirect-jump sites this descent met, with the table behind each one.
    /// A site with a complete table is not a gap: its targets are in `targets`
    /// and its case bodies are in `blocks`.
    pub tables: Vec<JumpTable>,
}

impl Plan {
    pub fn covered_bytes(&self) -> u64 {
        self.insns.iter().map(|(_, l)| *l as u64).sum()
    }

    pub fn gap_count(&self, g: Gap) -> usize {
        self.gaps.iter().filter(|s| s.gap == g).count()
    }
}

/// One decodable code range: bytes at a base address.
#[derive(Clone, Copy)]
pub struct Code<'a> {
    pub base: u64,
    pub bytes: &'a [u8],
}

impl<'a> Code<'a> {
    pub fn contains(&self, va: u64) -> bool {
        va >= self.base && va < self.base + self.bytes.len() as u64
    }
    fn tail(&self, va: u64) -> &'a [u8] {
        &self.bytes[(va - self.base) as usize..]
    }
}

/// Recursive descent over one code range.
pub fn discover(code: Code<'_>, entries: &[u64], call_targets: &[u64]) -> Plan {
    discover_with_tables(code, entries, call_targets, &Tables::new())
}

/// The same descent with the object's resolved jump tables in hand, so an
/// indirect jump over a relocation-covered table becomes N successors instead of
/// a gap. An empty `tables` is `discover` exactly.
pub fn discover_with_tables(
    code: Code<'_>,
    entries: &[u64],
    call_targets: &[u64],
    tables: &Tables,
) -> Plan {
    let mut decoded: Decoded = HashMap::new();
    discover_into(code, entries, call_targets, &mut decoded, tables)
}

/// The same descent, claiming into a caller-owned `decoded` map so several
/// ranges can share one claim set.
pub fn discover_into(
    code: Code<'_>,
    entries: &[u64],
    call_targets: &[u64],
    decoded: &mut Decoded,
    tables: &Tables,
) -> Plan {
    let mut plan = Plan::default();
    let mut functions: HashSet<u64> = entries
        .iter()
        .chain(call_targets.iter())
        .copied()
        .filter(|&va| code.contains(va))
        .collect();

    let mut seen: HashSet<u64> = HashSet::new();
    // Every address known to start a block, live: seeds, branch targets, call
    // targets, fallthroughs. A sweep must stop at these. A *snapshot* is not
    // enough - a call target discovered mid-sweep is a block start that the
    // sweep must stop for, and the first version of this function snapshotted,
    // swept straight through, and produced coverage over 100% with 35 objects
    // claiming the same byte twice.
    let mut known: HashSet<u64> = functions.clone();
    let mut queue: VecDeque<u64> = functions.iter().copied().collect();
    // Branch destinations, kept even when the bytes were already claimed. See
    // `Plan::targets`: the set that decides where a basic block must end.
    let mut targets: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();

    while let Some(start) = queue.pop_front() {
        if !seen.insert(start) || !code.contains(start) {
            continue;
        }
        plan.blocks.push(start);
        // `fallthrough` is a property of how the block ENDS, never of an
        // instruction in the middle of it: recording it at a `call` while the
        // sweep continued is how the block's own tail got enqueued as a second
        // block and swept twice.
        let mut fallthrough: Option<u64> = None;
        for i in decode(code.tail(start), start, Syntax::Intel) {
            if !i.valid() {
                plan.gaps.push(Site {
                    va: i.va,
                    gap: Gap::Undecodable,
                });
                break;
            }
            // Claim, or stop at the claim.
            if decoded.contains_key(&i.va) {
                fallthrough = Some(i.va);
                known.insert(i.va);
                break;
            }
            decoded.insert(i.va, i.len);
            known.insert(i.va);
            plan.insns.push((i.va, i.len));
            let next = i.va + i.len as u64;

            if i.is_ret {
                break; // a return transfers control to the caller
            }
            if i.is_indirect && !i.is_call {
                // The target set is not in the bytes. It may still be in the
                // CONTAINER: a resolved table turns this site into N block
                // successors, which is what claims the case bodies.
                match tables.get(&i.va) {
                    Some(t) => {
                        for &tg in &t.targets {
                            if !code.contains(tg) {
                                plan.gaps.push(Site {
                                    va: i.va,
                                    gap: Gap::OutsideCode,
                                });
                                continue;
                            }
                            targets.insert(tg);
                            if known.insert(tg) {
                                queue.push_back(tg);
                            }
                        }
                        if t.complete() {
                            plan.tables.push(t.clone());
                        } else {
                            plan.tables.push(t.clone());
                            plan.gaps.push(Site {
                                va: i.va,
                                gap: Gap::PartialTable,
                            });
                        }
                    }
                    None => plan.gaps.push(Site {
                        va: i.va,
                        gap: Gap::IndirectBranch,
                    }),
                }
                break; // the target set is not in the bytes
            }
            if i.is_call {
                match i.target {
                    Some(t) if code.contains(t) => {
                        functions.insert(t);
                        if known.insert(t) {
                            queue.push_back(t);
                        }
                    }
                    Some(_) => plan.gaps.push(Site {
                        va: i.va,
                        gap: Gap::OutsideCode,
                    }),
                    None => plan.gaps.push(Site {
                        va: i.va,
                        gap: Gap::IndirectCall,
                    }),
                }
                if known.contains(&next) {
                    fallthrough = Some(next);
                    break;
                }
                continue; // a call returns: keep sweeping this block
            }
            if i.is_jmp {
                match i.target {
                    Some(t) if code.contains(t) => {
                        targets.insert(t);
                        if known.insert(t) {
                            queue.push_back(t);
                        }
                    }
                    Some(_) => plan.gaps.push(Site {
                        va: i.va,
                        gap: Gap::OutsideCode,
                    }),
                    None => plan.gaps.push(Site {
                        va: i.va,
                        gap: Gap::IndirectBranch,
                    }),
                }
                break; // no fallthrough from an unconditional jump
            }
            if i.is_jcc {
                if let Some(t) = i.target {
                    if code.contains(t) {
                        targets.insert(t);
                        if known.insert(t) {
                            queue.push_back(t);
                        }
                    } else {
                        plan.gaps.push(Site {
                            va: i.va,
                            gap: Gap::OutsideCode,
                        });
                    }
                }
                fallthrough = Some(next);
                break;
            }
            // Not a terminator: the block continues, and stops at the next
            // known block start so no byte is claimed twice.
            if known.contains(&next) {
                fallthrough = Some(next);
                break;
            }
        }
        if let Some(s) = fallthrough {
            // The pop-side `seen` guard de-duplicates; inserting here would
            // mark the block visited before it is ever recorded.
            if code.contains(s) {
                queue.push_back(s);
            }
        }
    }

    plan.functions = functions.into_iter().collect();
    plan.functions.sort_unstable();
    plan.blocks.sort_unstable();
    plan.blocks.dedup();
    plan.insns.sort_unstable_by_key(|&(a, _)| a);
    plan.targets.extend(targets);
    plan.targets.sort_unstable();
    plan.targets.dedup();
    plan.gaps.sort_by_key(|g| g.va);
    plan
}

/// Discovery across several ranges, merged and sorted.
pub fn discover_all(ranges: &[Code<'_>], seeds: &HashMap<usize, (Vec<u64>, Vec<u64>)>) -> Plan {
    discover_all_with_tables(ranges, seeds, &Tables::new())
}

/// The same merge with the object's resolved jump tables, keyed by dispatch site.
pub fn discover_all_with_tables(
    ranges: &[Code<'_>],
    seeds: &HashMap<usize, (Vec<u64>, Vec<u64>)>,
    tables: &Tables,
) -> Plan {
    let mut acc = Plan::default();
    for (idx, r) in ranges.iter().enumerate() {
        let (entries, calls) = seeds.get(&idx).cloned().unwrap_or_default();
        let p = discover_with_tables(*r, &entries, &calls, tables);
        acc.functions.extend(p.functions);
        acc.targets.extend(p.targets);
        acc.blocks.extend(p.blocks);
        acc.insns.extend(p.insns);
        acc.gaps.extend(p.gaps);
        acc.tables.extend(p.tables);
    }
    acc.functions.sort_unstable();
    acc.functions.dedup();
    acc.blocks.sort_unstable();
    acc.blocks.dedup();
    acc.insns.sort_unstable_by_key(|&(a, _)| a);
    acc.gaps.sort_by_key(|g| g.va);
    acc
}

/// Bytes proven to be instructions must not overlap: an overlap means two
/// sweeps disagreed about a boundary and claimed the same byte twice.
pub fn overlap_free(insns: &[(u64, usize)]) -> bool {
    let mut owner: HashMap<u64, u64> = HashMap::new();
    for &(a, l) in insns {
        for k in 0..l as u64 {
            if owner.insert(a + k, a).is_some() {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// push rbx; sub rsp,0x20; mov eax,ecx; test eax,eax; jz +7; add eax,edx;
    /// mov rbx,[rsp+0x20]; add rsp,0x20; ret  (jz lands on the epilogue)
    const BODY: &[u8] = &[
        0x53, 0x48, 0x83, 0xEC, 0x20, 0x8B, 0xC1, 0x85, 0xC0, 0x74, 0x07, 0x01, 0xD0, 0x48, 0x8B,
        0x5C, 0x24, 0x20, 0x48, 0x83, 0xC4, 0x20, 0xC3,
    ];

    #[test]
    fn a_conditional_branch_reaches_both_targets_and_the_epilogue_is_one_block() {
        let code = Code {
            base: 0x1000,
            bytes: BODY,
        };
        let p = discover(code, &[0x1000], &[]);
        assert!(p.gaps.is_empty(), "no gaps expected: {:?}", p.gaps);
        // Blocks: entry, fallthrough after jz, jz target (the epilogue).
        // Blocks: entry, fallthrough after the jz, and the jz target - which is
        // the epilogue, reached first by fallthrough, and the sweep stops at a
        // known block start so no byte is claimed twice.
        assert_eq!(
            p.blocks,
            vec![0x1000, 0x100b, 0x1012],
            "blocks: {:?}",
            p.blocks
        );
        assert_eq!(
            p.functions,
            vec![0x1000],
            "a branch target is not a function"
        );
        assert!(overlap_free(&p.insns), "sweeps overlapped: {:?}", p.insns);
        // Every byte of the function is accounted for exactly once.
        assert_eq!(p.covered_bytes(), BODY.len() as u64);
    }

    #[test]
    fn a_direct_call_makes_the_target_a_function_seed() {
        // call rel32 -> +5, then ret. Target at 0x1000+11 lands in padding.
        let bytes: Vec<u8> = vec![
            0xE8, 0x05, 0x00, 0x00, 0x00, 0xC3, 0x90, 0x90, 0x90, 0x90, 0xC3,
        ];
        let code = Code {
            base: 0x2000,
            bytes: &bytes,
        };
        let p = discover(code, &[0x2000], &[]);
        assert_eq!(
            p.functions,
            vec![0x2000, 0x200a],
            "call target is a function"
        );
        assert!(overlap_free(&p.insns));
    }

    #[test]
    fn an_indirect_jump_is_a_recorded_gap_not_a_silent_stop() {
        // jmp qword ptr [rcx]  (FF 21) - the classic jump-table shape.
        let bytes: Vec<u8> = vec![0xFF, 0x21, 0x90];
        let code = Code {
            base: 0x3000,
            bytes: &bytes,
        };
        let p = discover(code, &[0x3000], &[]);
        assert_eq!(p.gap_count(Gap::IndirectBranch), 1, "{:?}", p.gaps);
        assert_eq!(p.gaps[0].va, 0x3000);
    }

    #[test]
    fn an_unreachable_tail_is_not_claimed_as_covered() {
        let mut bytes = BODY.to_vec();
        bytes.extend_from_slice(&[0x90, 0x90, 0x90]); // padding nobody jumps to
        let code = Code {
            base: 0x4000,
            bytes: &bytes,
        };
        let p = discover(code, &[0x4000], &[]);
        assert_eq!(
            p.covered_bytes(),
            BODY.len() as u64,
            "padding must stay uncovered"
        );
    }
}
