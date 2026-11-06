//! `rdec fuzz` — the hostile-input gate.
//!
//! The pipeline in this command is the PRODUCTION pipeline, not a fuzzer's
//! sketch of it: the object goes through `rdec_arch_x86::loader::load_bytes`,
//! every symbol name through `rdec_demangle::demangle`, and the code through
//! `m4::lift_all_named_shapes` - the same function `rdec c` and `rdec cpp`
//! call - then `verify_ssa`, `emit_c_sig` with the object's real prototypes,
//! and the interpreter on the lifted IR. A fuzz gate that re-implemented the
//! front half would be measuring its own copy, and the copy is where the bugs
//! would hide.
//!
//! Exit code: 0 with zero crashes and zero deadline hits, 1 otherwise. The
//! census is machine-readable (one `key=value` line per family) so the numbers
//! can be grepped, diffed and put in a roadmap row.

use std::collections::BTreeMap;
use std::time::Instant;

use rdec_arch_x86::fuzz::{self, Config, Input, Seed, Stats};
use rdec_arch_x86::loader;
use rdec_core::emit_c::emit_c_sig;
use rdec_core::interp::{run_budget, Outcome};
use rdec_core::verify::verify_ssa;

/// Inputs per iteration get their own step budget: an input that runs 200 000
/// steps is not a crash, and a fuzzer that spends its wall clock on one
/// quadratic input measures one input.
const FUZZ_BUDGET: usize = 20_000;

pub fn cmd_fuzz(args: &[String]) -> Result<i32, String> {
    let mut cfg = Config::default();
    let mut out_dir: Option<String> = None;
    let mut dump: Option<u64> = None;
    let mut repro: Option<String> = None;
    let mut obj_dir = "corpus/obj".to_string();
    let cpp_dir = "corpus/cpp/obj".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                i += 1;
                cfg.iterations = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--iterations needs a number")?;
            }
            "--seed" => {
                i += 1;
                cfg.seed = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--seed needs a number")?;
            }
            "--max-seconds" => {
                i += 1;
                cfg.max_seconds = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--max-seconds needs a number")?;
            }
            "--max-input-ms" => {
                i += 1;
                cfg.max_input_ms = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--max-input-ms needs a number")?;
            }
            "--trace" => cfg.trace = true,
            // Repro extraction: name the seed and the iteration, get the bytes.
            // A hang found by a census is only fixable if the input that caused
            // it can be written out and loaded again - the census's provenance
            // line is a pointer, and this is what it points at.
            "--repro" => {
                i += 1;
                repro = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--repro needs a file written by --out/--dump-iter")?,
                );
            }
            "--dump-iter" => {
                i += 1;
                dump = Some(
                    args.get(i)
                        .and_then(|s| s.parse().ok())
                        .ok_or("--dump-iter needs a number")?,
                );
            }
            "--out" => {
                i += 1;
                out_dir = Some(args.get(i).cloned().ok_or("--out needs a directory")?);
            }
            "--corpus" => {
                i += 1;
                obj_dir = args.get(i).cloned().ok_or("--obj-dir needs a directory")?;
            }
            // Consumed here; `cpp` is read from `args` below so the seed-set
            // choice cannot drift from the flag that made it.
            "--cpp" => {}
            other => {
                println!(
                    "fuzz: note: ignoring unknown argument `{other}` (used: --iterations N --seed S [--out DIR] [--max-seconds N] [--max-input-ms N] [--trace] [--dump-iter N] [--corpus DIR])"
                );
            }
        }
        i += 1;
    }
    if cfg.iterations == 0 {
        return Err(
            "fuzz: --iterations must be at least 1: a run that generates nothing asserts nothing"
                .to_string(),
        );
    }

    // `--cpp` is the whole seed set, not a label: the C++ objects are where the
    // decorated names are, and a run that claims C++ coverage without loading
    // them would be a census line describing nothing. Declared before the seed
    // directories are chosen, because it chooses them.
    let cpp = args.iter().any(|a| a == "--cpp");
    if cpp {
        println!("fuzz: seed set: corpus/obj + corpus/cpp/obj (--cpp)");
    }

    // Seeds: the real corpus objects, mutated. A missing corpus is LOUD: the
    // byte-level generator would silently degrade to pure random bytes, and
    // "fuzzed the corpus" would be a claim about a run that did not.
    let mut raw: Vec<(String, Vec<u8>)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let dirs: Vec<&str> = if cpp {
        vec![obj_dir.as_str(), cpp_dir.as_str()]
    } else {
        vec![obj_dir.as_str()]
    };
    for dir in dirs {
        match std::fs::read_dir(dir) {
            Ok(rd) => {
                let mut paths: Vec<_> = rd
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().map(|x| x == "obj").unwrap_or(false))
                    .collect();
                paths.sort();
                for p in paths {
                    if let Ok(b) = std::fs::read(&p) {
                        raw.push((p.display().to_string(), b));
                    }
                }
            }
            Err(e) => missing.push(format!("{dir}: {e}")),
        }
    }
    let (mut seeds, skipped) = fuzz::seeds_from_raw(&raw);
    println!("fuzz: seed {} iterations {}", cfg.seed, cfg.iterations);
    println!(
        "fuzz: corpus seeds {} objects from {obj_dir} + {cpp_dir} ({} raw, {} skipped: not COFF)",
        seeds.len(),
        raw.len(),
        skipped.len()
    );
    for s in skipped.iter().take(5) {
        println!("fuzz:   skipped {s}");
    }
    for m in &missing {
        println!("fuzz: NOTE no corpus ({m}) - the byte generator runs on random bytes and on a synthetic object only");
    }
    if seeds.is_empty() {
        // Keep the mutation generator alive with at least one well-formed
        // object, so the field-editing path (the thin part of the COFF parser)
        // is exercised even in a tree where the corpus was never built.
        seeds_fill(&mut seeds);
    }

    // `--repro FILE`: one saved input through the pipeline, with the per-stage
    // wall clock printed. The census says "65 s, stage X"; this says it for one
    // file you can hand to a debugger, and it is the loop a minimisation runs
    // (shrink the bytes, re-run this, keep the shrink if the time survives).
    if let Some(path) = &repro {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let input = Input {
            seed: cfg.seed,
            iter: u64::MAX,
            gen: "bytes",
            detail: format!("repro of {path}"),
            bytes,
        };
        let mut sub = Stats::default();
        let t = Instant::now();
        let r = fuzz::catch_panic(|| pipeline(&input, &mut sub, true));
        let ms = t.elapsed().as_millis();
        println!(
            "fuzz: repro {path}: {} bytes, {ms} ms, stage-ms {}",
            input.bytes.len(),
            fmt_map(&sub.stage_ms, "none")
        );
        match r {
            Ok(()) => println!("fuzz: repro completed (no panic)"),
            Err(msg) => println!(
                "fuzz: repro PANICKED: {}",
                msg.lines().next().unwrap_or("").trim()
            ),
        }
        return Ok(0);
    }

    // `--dump-iter N`: write the exact bytes iteration N generates, and stop.
    // The generator is a pure function of (seed, iteration, seeds), so this is
    // the census line made loadable: no re-run of a 65-second input needed.
    if let Some(it) = dump {
        let cfg1 = Config {
            iterations: it + 1,
            ..cfg.clone()
        };
        let input = fuzz::generate(&cfg1, &seeds, it);
        let dir = out_dir.clone().unwrap_or_else(|| ".".to_string());
        std::fs::create_dir_all(&dir).map_err(|e| format!("{dir}: {e}"))?;
        let p = format!("{dir}/input_seed{}_iter{}.obj", cfg.seed, it);
        std::fs::write(&p, &input.bytes).map_err(|e| format!("{p}: {e}"))?;
        println!(
            "fuzz: wrote {} bytes to {p} :: {}",
            input.bytes.len(),
            input.provenance()
        );
        println!("fuzz:   {}", input.detail);
        return Ok(0);
    }

    let mut st = Stats::default();
    let t0 = Instant::now();
    fuzz::run(&cfg, &seeds, &mut st, &mut |input, sub| {
        pipeline(input, sub, cfg.trace)
    });
    // The exhaustive name sweep, after the iterations: every prefix of every
    // decorated name the corpus carries. Deterministic (no RNG), cheap, and the
    // shape that found c660ff3 - the fuzzer's name family samples this space,
    // the sweep covers it.
    let names = fuzz::name_corpus(&seeds);
    fuzz::sweep_names(
        &cfg,
        &names,
        &mut st,
        &mut |n| match rdec_demangle::demangle(n) {
            rdec_demangle::Demangled::Name { .. } => {}
            rdec_demangle::Demangled::Unsupported { .. } => {}
        },
    );
    let wall = t0.elapsed();

    // ---- the census -----------------------------------------------------
    println!(
        "fuzz: inputs total={} bytes={} max_input={} generators [{}]",
        st.iterations,
        st.bytes_total,
        st.max_input,
        st.inputs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "fuzz: build {} (debug_assert live: {}) - a release run does NOT check the decoder's boundary asserts",
        if st.debug_checks { "debug" } else { "release" },
        st.debug_checks
    );
    let mut stages: Vec<(&String, &u64)> = st.stages.iter().collect();
    stages.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!(
        "fuzz: stages {}",
        stages
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let sms: BTreeMap<String, u64> = st
        .stage_ms
        .iter()
        .filter(|(_, v)| **v > 0)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    println!(
        "fuzz: slowest-stage-ms {} (each stage's slowest single input)",
        fmt_map(&sms, "none")
    );
    if !st.slow_note.is_empty() {
        println!("fuzz: {note}", note = st.slow_note);
    }
    println!("fuzz: load-errors {}", fmt_map(&st.load_errors, "none"));
    let gaps: BTreeMap<String, u64> = st
        .gaps
        .iter()
        .filter(|(k, _)| !k.starts_with("crash:"))
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    println!("fuzz: gaps {}", fmt_map(&gaps, "none"));
    println!(
        "fuzz: crashes {} distinct ({} retained){}",
        st.get("crash"),
        st.crashes.len(),
        if st.get("deadline.hit") > 0 {
            " AND A WALL-CLOCK DEADLINE WAS HIT (a hang: the run stopped early)"
        } else {
            ""
        }
    );
    for (i, c) in st.crashes.iter().enumerate() {
        println!(
            "fuzz: CRASH {i}: stage={} [{}] :: {}",
            c.stage,
            c.input.provenance(),
            c.message.lines().next().unwrap_or("").trim()
        );
        println!(
            "fuzz:   input {} bytes, {} {}",
            c.input.bytes.len(),
            c.input.gen,
            c.input.detail
        );
    }
    if let Some(dir) = &out_dir {
        write_repros(dir, &st)?;
        println!(
            "fuzz: repros written to {dir}/ ({} files)",
            st.crashes.len()
        );
    }
    println!(
        "fuzz: wall {:.1}s, slowest single input {} ms, per-input ceiling {} ms, budget {FUZZ_BUDGET} steps/input",
        wall.as_secs_f64(),
        st.slowest_ms,
        cfg.max_input_ms
    );

    let deadline = st.get("deadline.hit") > 0;
    let over = st.get("slow.over_budget");
    if over > 0 {
        let ceiling = cfg.max_input_ms;
        println!(
            "fuzz: {over} input(s) over the {ceiling} ms per-input ceiling: a decompiler that takes \
             that long for one object is hung for its users, and the ceiling only MEASURES it - \
             the algorithm that took it must be bounded"
        );
    }
    if st.crashes.is_empty() && !deadline && st.get("crash") == 0 && over == 0 {
        println!(
            "fuzz OK: {} inputs, 0 panics, 0 hangs, {} load refusals and {} gaps counted loudly",
            st.iterations,
            st.load_errors.values().sum::<u64>(),
            gaps.values().sum::<u64>()
        );
        return Ok(0);
    }
    println!(
        "fuzz FAIL: {} panics, deadline hit: {}, over-budget inputs: {}",
        st.get("crash"),
        deadline,
        over
    );
    Ok(1)
}

fn fmt_map(m: &BTreeMap<String, u64>, empty: &str) -> String {
    if m.is_empty() {
        return empty.to_string();
    }
    let mut v: Vec<(&String, &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    v.iter()
        .map(|(k, n)| format!("{k}={n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn write_repros(dir: &str, st: &Stats) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    // The slowest input, always: the census line that names a stage is only
    // actionable if the bytes it describes are on disk.
    if let Some(sl) = &st.slowest {
        let p = format!("{dir}/slowest_seed{}_iter{}.obj", sl.seed, sl.iter);
        std::fs::write(&p, &sl.bytes).map_err(|e| format!("{p}: {e}"))?;
    }
    for (i, c) in st.crashes.iter().enumerate() {
        let p = format!(
            "{dir}/crash_{i}_seed{}_iter{}.obj",
            c.input.seed, c.input.iter
        );
        std::fs::write(&p, &c.input.bytes).map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}

/// One input through the whole product. Bumps `sub`, the run's scratch census:
/// a panic part-way through leaves the partial counts out of the census (the
/// driver merges only what a completed input produced), and the crash record
/// carries the bytes.
/// One input through the pipeline. `verbose` prints each stage's wall clock as
/// soon as that stage FINISHES, so a run that never returns has already printed
/// the stage it is stuck in - the census's `slowest-stage-ms` can only name the
/// stage after the input completes, and a 65-second input is exactly the case
/// where "after" is too late.
fn pipeline(input: &Input, sub: &mut Stats, verbose: bool) {
    // Name inputs are strings, not objects: they go straight to the demangler,
    // the only stage that can consume them. The truncation-at-every-length
    // sweep is a separate phase (`sweep_names`); an input that reaches this
    // function is one the generator chose to feed the pipeline.
    if input.gen == "name" {
        let name = input.detail.strip_prefix("name: ").unwrap_or(&input.detail);
        match rdec_demangle::demangle(name) {
            rdec_demangle::Demangled::Name { .. } => sub.bump("demangle.name", 1),
            rdec_demangle::Demangled::Unsupported { .. } => sub.bump("demangle.unsupported", 1),
        }
        sub.bump("name.inputs", 1);
        return;
    }
    // Every generated input is ALSO a byte stream for the decoder, straight
    // with no container: this is the call that fires `decode.rs`'s boundary
    // asserts on raw noise, in a debug build.
    let t = Instant::now();
    let ins =
        rdec_arch_x86::decode::decode(&input.bytes, 0x1000, rdec_arch_x86::decode::Syntax::Intel);
    stamp(sub, verbose, "stage.decode", t);
    sub.bump("decode.calls", 1);
    sub.bump("decode.insns", ins.len() as u64);
    sub.bump(
        "decode.refusals",
        ins.iter().filter(|i| !i.valid()).count() as u64,
    );

    // Stage 1: the container.
    let t = Instant::now();
    let obj = match loader::load_bytes(&input.bytes, "fuzz.obj") {
        Ok(o) => {
            sub.bump("load.ok", 1);
            o
        }
        Err(e) => {
            stamp(sub, verbose, "stage.load", t);
            sub.bump("load.err", 1);
            *sub.load_errors
                .entry(fuzz::classify_load_error(&e))
                .or_insert(0) += 1;
            return;
        }
    };
    stamp(sub, verbose, "stage.load", t);
    if obj.string_table_truncated {
        sub.bump("load.string_table_truncated", 1);
    }

    // Stage 2: the demangler, on every name the object carries. The pipeline
    // feeds it symbol names and relocation target names, so those are the
    // strings it sees in production.
    let t = Instant::now();
    for s in &obj.symbols {
        match rdec_demangle::demangle(&s.name) {
            rdec_demangle::Demangled::Name { .. } => sub.bump("demangle.name", 1),
            rdec_demangle::Demangled::Unsupported { .. } => sub.bump("demangle.unsupported", 1),
        }
    }
    stamp(sub, verbose, "stage.demangle", t);
    sub.bump("symbols", obj.symbols.len() as u64);
    sub.bump("sections", obj.sections.len() as u64);
    sub.bump("relocs", obj.total_relocs() as u64);
    // Absolute symbols (`SectionNumber == 0`, `Symbol::section == None`) are the
    // class of record production hands a decompiler and the class random bytes
    // never generate: `Object::code_symbols` used to `unwrap()` them in a sort
    // key. Counted, so the census says the path was walked.
    sub.bump(
        "symbols.absolute",
        obj.symbols.iter().filter(|s| s.section.is_none()).count() as u64,
    );

    // Stage 3: the production front half - layout, seeds, jump tables,
    // relocation evidence, recursive descent, lift, the pass pipeline.
    // Master's `m4::lift_all_named_shapes` takes FILE PATHS, so the generated
    // bytes go to a temp file and through the exact function `rdec c` and
    // `rdec cpp` call: the pipeline measured is the product's, not a fuzzer's
    // copy, and the temp file is the input's own bytes, not a re-encoding.
    let t = Instant::now();
    let path = std::env::temp_dir().join(format!(
        "rdec_fuzz_{}_{}.obj",
        std::process::id(),
        input.iter
    ));
    if let Err(e) = std::fs::write(&path, &input.bytes) {
        // Disk/env refusal, not product behaviour: counted LOUD in the census
        // (the gap line names it) and the input is skipped. No panic site.
        sub.gap(&format!("harness: temp write refused: {e}"), 1);
        return;
    }
    let subjects = match crate::m4::lift_all_named_shapes(
        &[path.display().to_string()],
        &|oi, raw| format!("o{oi}_{}", crate::m4::c_ident(raw)),
        &|_oi, _raw| rdec_arch_x86::lift::FnShape {
            abi: rdec_arch_x86::lift::FnAbi::integer(),
            typed: true,
        },
    ) {
        Ok(s) => s,
        Err(msg) => {
            // The loader accepted these bytes and the production front half
            // then failed to load the same file from disk: a harness-level
            // file-I/O refusal, counted loudly, never a panic site.
            let _ = std::fs::remove_file(&path);
            sub.gap(
                &format!(
                    "harness: m4 reload refused: {}",
                    msg.lines().next().unwrap_or("no message")
                ),
                1,
            );
            return;
        }
    };
    let _ = std::fs::remove_file(&path);
    stamp(sub, verbose, "stage.discover_lift", t);
    // The census, derived from the subjects. Master's API returns a flat
    // `Vec<Subject>` and no plan census, so the discover-side numbers here are
    // what the SURVIVING IR proves (post-`clean` blocks and instructions), not
    // the plan's pre-lift proof. Counters the old call site read off the
    // lane's per-object `LiftedObject` and that subjects cannot give -
    // `discover.bytes` (plan.covered_bytes), `discover.tables`,
    // `discover.table_slots`, `discover.table_unresolved`, and the per-`Gap`
    // `discover.<gap>` gap loop - are DROPPED, named here and in the run
    // report; inventing substitutes from other numbers would be census fraud.
    // The lifter's own per-function table report (jump_tables and friends) is
    // bumped below, so table coverage is not lost, only re-based.
    sub.bump("discover.functions", subjects.len() as u64);
    let blocks: usize = subjects.iter().map(|s| s.f.blocks.len()).sum();
    sub.bump("discover.blocks", blocks as u64);
    let insns: usize = subjects
        .iter()
        .map(|s| s.f.blocks.iter().map(|b| b.instrs.len()).sum::<usize>())
        .sum();
    sub.bump("lift.insns", insns as u64);
    if let Some(s0) = subjects.first() {
        // Object-level, carried on every subject of the object (the m4
        // contract); this call site lifts one object, so first == every.
        sub.bump("relocs.parsed", s0.relocs_parsed as u64);
    }
    sub.bump("lift.functions", subjects.len() as u64);
    for s in &subjects {
        sub.bump("lift.gaps", s.rep.gaps as u64);
        sub.bump("lift.exits", s.rep.exits as u64);
        sub.bump("lift.calls_indirect", s.rep.calls_indirect as u64);
        sub.bump("lift.jump_tables", s.rep.jump_tables as u64);
        sub.bump("lift.jump_table_edges", s.rep.jump_table_edges as u64);
        sub.bump("lift.jump_table_partial", s.rep.jump_table_partial as u64);
        sub.bump("lift.jump_table_missing", s.rep.jump_table_missing as u64);
    }

    // Stage 4: the emitter's prototypes, exactly as `rdec c` builds them, then
    // verify -> emit C -> interpreter on the lifted IR.
    let protos = crate::m4::protos_of(&subjects);
    let mut vms = 0u64;
    let mut ems = 0u64;
    let mut ims = 0u64;
    if verbose {
        eprintln!(
            "fuzz:   subjects={} blocks={} insns={}",
            subjects.len(),
            blocks,
            insns
        );
        for (i, s) in subjects.iter().enumerate() {
            eprintln!(
                "fuzz:   subject {i} {} start={:#x} blocks={} values={} instrs={}",
                s.cname,
                s.start,
                s.f.blocks.len(),
                s.f.next_value,
                s.f.blocks.iter().map(|b| b.instrs.len()).sum::<usize>()
            );
            eprintln!("{}", rdec_core::fmt_ir::format_function(&s.f));
        }
    }
    for s in &subjects {
        let tv = Instant::now();
        match verify_ssa(&s.f) {
            Ok(()) => sub.bump("verify.ok", 1),
            Err(errs) => {
                sub.bump("verify.reject", 1);
                // The IR the passes produced is not structurally valid: that is
                // a pass bug, and the census names it per first message.
                sub.gap(
                    &format!(
                        "verify: {}",
                        errs.first()
                            .map(|e| e.split(':').next().unwrap_or(e).to_string())
                            .unwrap_or_else(|| "no message".to_string())
                    ),
                    1,
                );
                vms += tv.elapsed().as_millis() as u64;
                continue;
            }
        }
        vms += tv.elapsed().as_millis() as u64;
        let te = Instant::now();
        match emit_c_sig(&s.f, &protos) {
            Ok(out) => {
                sub.bump("emit.c", 1);
                sub.bump("emit.c_bytes", out.text.len() as u64);
            }
            Err(rs) => {
                sub.bump("emit.refused", 1);
                for r in &rs {
                    sub.gap(&format!("emit: {}", crate::m4::key(r)), 1);
                }
            }
        }
        ems += te.elapsed().as_millis() as u64;
        // The interpreter, on the lifted IR, with a bounded step budget: the
        // same instrument `rdec t2` uses, at a fuzz-sized budget.
        let ti = Instant::now();
        let exec = run_budget(&s.f, &[1, 2, 3, 4], FUZZ_BUDGET);
        ims += ti.elapsed().as_millis() as u64;
        sub.bump("interp.runs", 1);
        match &exec.outcome {
            Outcome::Ret(_) => sub.bump("interp.ret", 1),
            Outcome::NoReturn => sub.bump("interp.noreturn", 1),
            Outcome::Budget => sub.bump("interp.budget", 1),
            Outcome::Fault(f) => {
                if matches!(f, rdec_core::interp::Fault::MissingFlag(_)) {
                    // The interpreter asked for a flag the node never carried:
                    // loud, counted, named apart from ordinary faults.
                    sub.bump("interp.missing_flag", 1);
                }
                sub.gap(&format!("interp.fault {f:?}"), 1)
            }
        }
        for e in &exec.effects {
            if matches!(e, rdec_core::interp::Effect::Opaque(_)) {
                sub.bump("interp.opaque", 1);
            }
        }
    }
    sub.time("stage.verify", vms);
    sub.time("stage.emit_c", ems);
    sub.time("stage.interp", ims);
}

/// Record a stage's wall clock, and - when tracing - print it the moment the
/// stage finishes. A census line printed at the end of a run describes inputs
/// that completed; the eprintln describes the one that did not.
fn stamp(sub: &mut Stats, verbose: bool, stage: &str, t: Instant) {
    let ms = t.elapsed().as_millis() as u64;
    sub.time(stage, ms);
    if verbose {
        eprintln!("fuzz:   {stage} {ms} ms");
    }
}

/// Fallback seeds when there is no corpus on disk: a synthetic, well-formed
/// object, so the field-editing mutator still has headers to edit.
fn seeds_fill(v: &mut Vec<Seed>) {
    v.push(Seed {
        name: "synthetic.obj".to_string(),
        obj: fuzz::text_object(&[0x48, 0x89, 0xC8, 0xC3]),
        text: vec![0x48, 0x89, 0xC8, 0xC3],
        names: vec![],
    });
}
