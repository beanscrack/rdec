//! x86-64 decoding, with boundaries as the product.
//!
//! M1's gate is not "can we print mnemonics" — it is that the *instruction
//! boundaries* are right, because every later phase (basic blocks, SSA, CFG)
//! consumes addresses, and a single misaligned decode silently invents a
//! function. The checks that pin this are `check_objdump` (boundaries agree
//! with llvm-objdump on a real object) and the fuzz test (never panics, always
//! advances).

use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter, NasmFormatter};

/// One decoded instruction.
#[derive(Clone, Debug)]
pub struct Insn {
    /// Virtual (section-relative, in our model) address of the first byte.
    pub va: u64,
    /// Encoded length in bytes. 0 means the decoder refused these bytes.
    pub len: usize,
    /// Rendered text, `None` when the bytes did not decode.
    pub text: Option<String>,
    pub code: u16,
    pub is_branch: bool,
    /// `jmp` — transfers control and has no fallthrough.
    pub is_jmp: bool,
    /// `jcc` — transfers control conditionally, so it has a fallthrough.
    pub is_jcc: bool,
    /// An indirect branch/call: the target set is not in the bytes.
    pub is_indirect: bool,
    pub is_call: bool,
    pub is_ret: bool,
    /// A direct branch/call target, when the operand is an immediate.
    pub target: Option<u64>,
}

impl Insn {
    pub fn valid(&self) -> bool {
        self.len > 0 && self.text.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Syntax {
    Intel,
    Nasm,
}

pub(crate) fn formatter(syntax: Syntax) -> Box<dyn Formatter> {
    match syntax {
        Syntax::Intel => Box::new(IntelFormatter::new()),
        Syntax::Nasm => Box::new(NasmFormatter::new()),
    }
}

/// Linear sweep over `bytes` starting at `va`.
///
/// A linear sweep is not a disassembly strategy for decompilation — M1's block
/// discovery follows control flow — but it is the right instrument for
/// measuring decode coverage, because it visits every byte and stops the list
/// at the first refusal. Refusals are returned as `len == 0` entries, never
/// dropped: a gap must be visible in the output.
pub fn decode(bytes: &[u8], va: u64, syntax: Syntax) -> Vec<Insn> {
    let mut fmt = formatter(syntax);
    let mut out = Vec::new();
    let mut decoder = Decoder::with_ip(64, bytes, va, DecoderOptions::NONE);
    let mut instr = Instruction::default();
    while decoder.can_decode() {
        decoder.decode_out(&mut instr);
        if instr.code() as u16 == 0 {
            // Code::INVALID == 0: refuse, do not skip. Advancing here is how a
            // decoder invents instructions out of data.
            out.push(Insn {
                va: instr.ip(),
                len: 0,
                text: None,
                code: 0,
                is_branch: false,
                is_jmp: false,
                is_jcc: false,
                is_indirect: false,
                is_call: false,
                is_ret: false,
                target: None,
            });
            break;
        }
        let mut text = String::new();
        fmt.format(&instr, &mut text);
        let flow = flow_kind(&instr);
        out.push(Insn {
            va: instr.ip(),
            len: instr.len(),
            text: Some(text),
            code: instr.code() as u16,
            is_branch: flow.0,
            is_jmp: matches!(
                instr.flow_control(),
                iced_x86::FlowControl::UnconditionalBranch
            ),
            is_jcc: matches!(
                instr.flow_control(),
                iced_x86::FlowControl::ConditionalBranch
            ),
            is_indirect: matches!(
                instr.flow_control(),
                iced_x86::FlowControl::IndirectBranch | iced_x86::FlowControl::IndirectCall
            ),
            is_call: flow.1,
            is_ret: flow.2,
            target: direct_target(&instr),
        });
    }
    out
}

fn flow_kind(i: &Instruction) -> (bool, bool, bool) {
    use iced_x86::FlowControl;
    match i.flow_control() {
        FlowControl::UnconditionalBranch
        | FlowControl::ConditionalBranch
        | FlowControl::IndirectBranch => (true, false, false),
        FlowControl::Call | FlowControl::IndirectCall => (false, true, false),
        FlowControl::Return => (false, false, true),
        _ => (false, false, false),
    }
}

fn direct_target(i: &Instruction) -> Option<u64> {
    use iced_x86::{FlowControl, OpKind};
    if !matches!(
        i.flow_control(),
        FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch | FlowControl::Call
    ) {
        return None;
    }
    match i.op0_kind() {
        // A near branch target is its own operand kind in iced, NOT an
        // immediate: `immediate(0)` on it panics ("Invalid operand: 0"), which
        // is how the first version of this function silently returned no
        // targets and discovery lost every call edge.
        OpKind::NearBranch16 => Some(i.near_branch16() as u64),
        OpKind::NearBranch32 => Some(i.near_branch32() as u64),
        OpKind::NearBranch64 => Some(i.near_branch64()),
        OpKind::Immediate8 | OpKind::Immediate8_2nd | OpKind::Immediate16 | OpKind::Immediate32 => {
            Some((i.next_ip() as i64 + i.immediate(0) as i32 as i64) as u64)
        }
        _ => None,
    }
}

/// Decode, re-encode, and re-decode each instruction, requiring the boundary
/// sequence to be stable. This is the roadmap's decode -> encode -> decode
/// round-trip, and it catches operand-level damage a boundary check cannot.
pub fn round_trip(bytes: &[u8], va: u64) -> Result<usize, String> {
    use iced_x86::Encoder;
    let insns = decode(bytes, va, Syntax::Intel);
    for i in &insns {
        if !i.valid() {
            return Err(format!("undecodable at {:#x}", i.va));
        }
    }
    // Re-encode from the decoded instructions.
    let mut enc = Encoder::new(64);
    let mut decoder = Decoder::with_ip(64, bytes, va, DecoderOptions::NONE);
    let mut instr = Instruction::default();
    while decoder.can_decode() {
        decoder.decode_out(&mut instr);
        enc.encode(&instr, instr.ip())
            .map_err(|e| format!("encode failed at {:#x}: {:?}", instr.ip(), e))?;
    }
    let re_encoded = enc.take_buffer();
    let re = decode(&re_encoded, va, Syntax::Intel);
    if re.len() != insns.len() {
        return Err(format!(
            "instruction count changed after re-encode: {} -> {} ({} -> {} bytes)",
            insns.len(),
            re.len(),
            bytes.len(),
            re_encoded.len()
        ));
    }
    for (a, b) in insns.iter().zip(&re) {
        if a.va != b.va || a.len != b.len || a.code != b.code {
            return Err(format!(
                "boundary/opcode moved: {:#x} len {} code {} -> {:#x} len {} code {}",
                a.va, a.len, a.code, b.va, b.len, b.code
            ));
        }
    }
    Ok(insns.len())
}
