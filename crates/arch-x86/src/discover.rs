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

use std::collections::{HashMap, HashSet, VecDeque};

/// Bytes claimed by a sweep: address -> instruction length. This is the
/// invariant that makes overlap impossible: a byte is claimed once, and a sweep
/// that reaches an already-claimed address stops there - that address is a
/// block boundary by construction. Claiming per-sweep, and stopping only at
/// starts known *at the time*, is what let a function body sweep past a branch
/// target that a later sweep then decoded again.
pub type Decoded = HashMap<u64, usize>;

use crate::decode::{decode, Syntax};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gap {
    /// `jmp *%rax`, `call *%rax`, `jmp [tbl+rcx*8]`: the target set is not in
    /// the bytes. Jump-table recovery is M5; recording the site is M1.
    IndirectBranch,
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
    let mut decoded: Decoded = HashMap::new();
    discover_into(code, entries, call_targets, &mut decoded)
}

/// The same descent, claiming into a caller-owned `decoded` map so several
/// ranges can share one claim set.
pub fn discover_into(
    code: Code<'_>,
    entries: &[u64],
    call_targets: &[u64],
    decoded: &mut Decoded,
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
                plan.gaps.push(Site {
                    va: i.va,
                    gap: Gap::IndirectBranch,
                });
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
    let mut acc = Plan::default();
    for (idx, r) in ranges.iter().enumerate() {
        let (entries, calls) = seeds.get(&idx).cloned().unwrap_or_default();
        let p = discover(*r, &entries, &calls);
        acc.functions.extend(p.functions);
        acc.targets.extend(p.targets);
        acc.blocks.extend(p.blocks);
        acc.insns.extend(p.insns);
        acc.gaps.extend(p.gaps);
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
