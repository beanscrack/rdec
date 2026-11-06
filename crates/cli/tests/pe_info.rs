//! `rdec pe-info` at the process boundary: exit codes and printed bytes.
//!
//! The header bytes are the same fixture as
//! `crates/arch-x86/tests/pe_fixtures.rs`, which is where every field is quoted
//! with its offset and audited. This file re-builds them because an
//! integration test cannot import another crate's test module; the loader
//! contract itself is asserted there, and what is asserted HERE is the CLI
//! surface the M7 gate will be run from:
//!
//!   * a PE32+ image exits **0** and prints image base, entry, subsystem and the
//!     section table (name / vaddr / vsize / rawptr / rawsize / flags);
//!   * a non-PE exits **2** - the refusal exit code this CLI already uses for
//!     `c` / `t2` / `lift` / `fuzz` - and the refusal on stderr names the field,
//!     the same message `PeImage::load` gives the library tests;
//!   * the real linked EXE (the M7 probe artifact) prints the sections the probe
//!     measured, when it is on the machine. Absent prints a named SKIP line:
//!     never a silent pass.

use std::path::{Path, PathBuf};
use std::process::Command;

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// The good PE32+ fixture: 2 sections, entry at .text+0x30, 0x600 bytes.
/// Field-by-field offsets and their meaning: `pe_fixtures.rs` in arch-x86.
fn pe32plus() -> Vec<u8> {
    let mut b = vec![0u8; 0x600];
    put16(&mut b, 0x00, 0x5a4d);
    put32(&mut b, 0x3c, 0x40);
    b[0x40..0x44].copy_from_slice(b"PE\0\0");
    put16(&mut b, 0x44, 0x8664);
    put16(&mut b, 0x46, 2);
    put16(&mut b, 0x54, 240);
    put16(&mut b, 0x56, 0x0022);
    put16(&mut b, 0x58, 0x20b);
    put32(&mut b, 0x68, 0x1030); // AddressOfEntryPoint
    put64(&mut b, 0x70, 0x0001_4000_0000); // ImageBase
    put32(&mut b, 0x78, 0x1000); // SectionAlignment
    put32(&mut b, 0x7c, 0x200); // FileAlignment
    put32(&mut b, 0x90, 0x3000); // SizeOfImage
    put32(&mut b, 0x94, 0x200); // SizeOfHeaders
    put16(&mut b, 0x9c, 3); // Subsystem: Windows CUI
    put16(&mut b, 0x9e, 0x0160); // DllCharacteristics
    put32(&mut b, 0xc4, 16); // NumberOfRvaAndSizes
    b[0x148..0x148 + 5].copy_from_slice(b".text");
    put32(&mut b, 0x150, 0x40); // VirtualSize
    put32(&mut b, 0x154, 0x1000); // VirtualAddress
    put32(&mut b, 0x158, 0x200); // SizeOfRawData
    put32(&mut b, 0x15c, 0x200); // PointerToRawData
    put32(&mut b, 0x16c, 0x6000_0020); // CODE | EXECUTE | READ
    b[0x170..0x170 + 6].copy_from_slice(b".rdata");
    put32(&mut b, 0x178, 0x200);
    put32(&mut b, 0x17c, 0x2000);
    put32(&mut b, 0x180, 0x200);
    put32(&mut b, 0x184, 0x400);
    put32(&mut b, 0x194, 0x4000_0040); // INIT_DATA | READ
    for i in 0..0x40usize {
        b[0x200 + i] = 0xcc;
    }
    b[0x230..0x236].copy_from_slice(&[0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]);
    b[0x400..0x404].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    b
}

fn write_tmp(name: &str, bytes: &[u8]) -> PathBuf {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&p, bytes).expect("the fixture writes");
    p
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rdec"))
        .args(args)
        .output()
        .expect("the rdec binary runs");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn pe_info_prints_the_image_header_and_exits_zero() {
    let p = write_tmp("pe_info_good.exe", &pe32plus());
    let (rc, out, err) = run(&["pe-info", &p.display().to_string()]);
    assert_eq!(rc, 0, "a PE32+ image is accepted; stderr was: {err}");
    assert!(out.contains("PE32+ image"), "{out}");
    assert!(out.contains("0x140000000"), "image base: {out}");
    assert!(out.contains("0x1030"), "entry RVA: {out}");
    assert!(out.contains("Windows CUI"), "subsystem by name: {out}");
    assert!(out.contains(".text"), "the section table: {out}");
    assert!(out.contains(".rdata"), "the section table: {out}");
    assert!(out.contains("CODE|EXECUTE|READ"), "flags summary: {out}");
    assert!(out.contains("INIT_DATA|READ"), "flags summary: {out}");
    assert!(out.contains("0x200"), "raw pointer/size columns: {out}");
    // The header line counts sections, so "it printed something" is checkable.
    assert!(out.contains("sections        2"), "{out}");
}

#[test]
fn pe_info_refuses_a_non_pe_with_exit_2_naming_the_field() {
    let mut bad = pe32plus();
    bad[0] = b'N'; // "NZ": not MZ
    let p = write_tmp("pe_info_badmz.exe", &bad);
    let (rc, out, err) = run(&["pe-info", &p.display().to_string()]);
    assert_eq!(
        rc, 2,
        "a refusal exits 2 (the c/t2/fuzz convention); stdout was: {out}"
    );
    assert!(err.contains("MZ"), "the refusal names the magic: {err}");
    assert!(
        err.contains("1536"),
        "and the length of the file it was measured against: {err}"
    );

    // The 32-bit image, refused by name at the CLI too.
    let mut pe32 = pe32plus();
    put16(&mut pe32, 0x58, 0x10b);
    let p = write_tmp("pe_info_pe32.exe", &pe32);
    let (rc, _, err) = run(&["pe-info", &p.display().to_string()]);
    assert_eq!(rc, 2, "a PE32 image is refused, exit 2");
    assert!(err.contains("0x10b"), "naming the magic found: {err}");
    assert!(err.contains("PE32"), "and what that magic is: {err}");
}

#[test]
fn pe_info_without_a_path_is_a_usage_refusal() {
    let (rc, _, err) = run(&["pe-info"]);
    assert_eq!(rc, 2, "no path is a refusal, exit 2");
    assert!(
        err.contains("usage"),
        "and says what the command wants: {err}"
    );
}

fn scratch_m7() -> Option<PathBuf> {
    let root = std::env::var("LOCALAPPDATA").map(PathBuf::from).ok()?;
    let p = root.join("hermes/cache/scratch/m7/m7.exe");
    p.is_file().then_some(p)
}

#[test]
fn pe_info_on_the_real_linked_exe_lists_the_sections_the_probe_measured() {
    let Some(p) = scratch_m7() else {
        println!("SKIP(pe-info real-exe): m7.exe is not present at %LOCALAPPDATA%/hermes/cache/scratch/m7/m7.exe - the real-image CLI leg did not run");
        return;
    };
    let (rc, out, err) = run(&["pe-info", &p.display().to_string()]);
    assert_eq!(rc, 0, "the probe artifact is a PE32+ image; stderr: {err}");
    for want in [
        "0x140000000",
        "entry RVA 0x1280",
        "Windows CUI",
        ".text",
        ".pdata",
        ".reloc",
    ] {
        assert!(out.contains(want), "pe-info must print {want}:\n{out}");
    }
    // The .text VMA from the probe, printed as an RVA of 0x1000.
    assert!(out.contains("0x1000"), ".text RVA: {out}");
    // The scope line: what the reader loaded and did NOT read.
    assert!(
        out.contains("loaded, not yet read"),
        "the CLI must say what it did not read: {out}"
    );
}
