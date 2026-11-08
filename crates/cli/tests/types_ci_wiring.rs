//! The `types` gate's CI wiring — a test of the WORKFLOW FILE, not of the pass.
//!
//! This repo shipped a red pipeline twice for the same reason: a step that reads a
//! corpus object lived in a job whose corpus-generation step lived somewhere else
//! (`corpus/obj/*.obj` and `corpus/cpp/obj/*.obj` are generated, not committed), so
//! the step passed on the machine that built the artifacts and died in CI with
//! "The system cannot find the path specified (os error 3)" — runs 41415743892 and
//! 42148930772. `rdec types --corpus` reads `corpus/manifest.tsv`, the objects it
//! names, and `corpus/truth.tsv`, so it is exactly that class of step, and prose in
//! a comment is what failed the first two times.
//!
//! So this file pins the three things that make the step real:
//!   1. generation is on its LEFT, in the SAME job;
//!   2. the step checks `$LASTEXITCODE` after the gate (PowerShell does not fail on
//!      a native exit code, so an unchecked gate is a gate that runs red green);
//!   3. the step is not vacuous — the parser has to FIND the steps to assert about.

use std::path::PathBuf;

struct Step {
    name: String,
    run: String,
    shell: Option<String>,
}

struct Job {
    name: String,
    steps: Vec<Step>,
}

fn workflow_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let cand = dir.join(".github").join("workflows");
        if cand.is_dir() {
            return Some(cand);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn workflow_files() -> Vec<PathBuf> {
    let dir = match workflow_dir() {
        Some(d) => d,
        None => return Vec::new(),
    };
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    rd.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .map(|x| x == "yml" || x == "yaml")
                .unwrap_or(false)
        })
        .collect()
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// A deliberately small YAML reader: job keys at indent 2, `- name:` steps, and the
/// lines under a `run:` key. It parses the shape the repo's workflows are written
/// in, and the non-vacuity assertions below are what stop a parser change from
/// turning this into a test that checks nothing.
fn parse_jobs(text: &str) -> Vec<Job> {
    let lines: Vec<&str> = text.lines().collect();
    let mut jobs: Vec<Job> = Vec::new();
    let mut in_jobs = false;
    let mut in_steps = false;
    let mut run_indent: Option<usize> = None;
    for line in &lines {
        let t = line.trim_end();
        let trimmed = t.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let ind = indent(t);
        if ind == 0 {
            in_jobs = trimmed == "jobs:";
            in_steps = false;
            run_indent = None;
            continue;
        }
        if !in_jobs {
            continue;
        }
        if run_indent.map(|r| ind >= r).unwrap_or(false) {
            if let Some(job) = jobs.last_mut() {
                if let Some(step) = job.steps.last_mut() {
                    step.run.push('\n');
                    step.run.push_str(trimmed);
                }
            }
            continue;
        }
        run_indent = None;
        if ind == 2 && trimmed.ends_with(':') {
            jobs.push(Job {
                name: trimmed.trim_end_matches(':').to_string(),
                steps: Vec::new(),
            });
            in_steps = false;
            continue;
        }
        if trimmed.starts_with("steps:") {
            in_steps = true;
            continue;
        }
        if !in_steps {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- name:") {
            if let Some(job) = jobs.last_mut() {
                job.steps.push(Step {
                    name: rest.trim().to_string(),
                    run: String::new(),
                    shell: None,
                });
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("shell:") {
            if let Some(job) = jobs.last_mut() {
                if let Some(step) = job.steps.last_mut() {
                    step.shell = Some(rest.trim().to_string());
                }
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("run:") {
            let rest = rest.trim();
            if !rest.is_empty() && rest != "|" && rest != ">" {
                if let Some(job) = jobs.last_mut() {
                    if let Some(step) = job.steps.last_mut() {
                        step.run.push('\n');
                        step.run.push_str(rest);
                    }
                }
            } else {
                run_indent = Some(ind);
            }
        }
    }
    jobs
}

fn all_steps(text: &str) -> Vec<(String, Step)> {
    parse_jobs(text)
        .into_iter()
        .flat_map(|j| j.steps.into_iter().map(move |s| (j.name.clone(), s)))
        .collect()
}

#[test]
fn the_types_gate_has_its_corpus_generation_on_its_left_in_the_same_job() {
    let files = workflow_files();
    assert!(
        !files.is_empty(),
        "no workflow files found from {} - the test is checking nothing",
        std::env::current_dir().unwrap().display()
    );
    let mut types_steps = 0usize;
    let mut problems = Vec::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for job in parse_jobs(&text) {
            let mut corpus_at: Option<usize> = None;
            for (i, step) in job.steps.iter().enumerate() {
                let runs_corpus_gen =
                    step.run.contains("-p rdec-cli -- corpus") && !step.run.contains("-- baseline");
                let runs_types = step.run.contains("-p rdec-cli -- types");
                if runs_corpus_gen {
                    corpus_at = Some(i);
                }
                if runs_types {
                    types_steps += 1;
                    match corpus_at {
                        None => problems.push(format!(
                            "{}: job `{}` step `{}` reads the corpus with no generation step above it in the SAME job",
                            f.display(),
                            job.name,
                            step.name
                        )),
                        Some(c) => {
                            if job.steps[c]
                                .run
                                .lines()
                                .all(|l| !l.contains("-- corpus"))
                            {
                                problems.push(format!(
                                    "{}: job `{}` step `{}` - the step above it does not generate the C corpus",
                                    f.display(),
                                    job.name,
                                    step.name
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        types_steps > 0,
        "no `rdec types` step in any workflow file - the M6 gate is not wired into CI"
    );
    assert!(
        problems.is_empty(),
        "corpus-reading steps without generation on their left:\n{}",
        problems.join("\n")
    );
}

#[test]
fn the_types_step_checks_the_gate_exit_code() {
    // In PowerShell a native command's non-zero exit does not stop the script, so an
    // unchecked `types` step is a gate that runs red green. The raw
    // `types --corpus` run is EXPECTED to be red (the named carry_chain lies), which
    // is precisely why its code has to be asserted, not dropped.
    let files = workflow_files();
    assert!(!files.is_empty(), "no workflow files found");
    let mut checked = 0usize;
    let mut problems = Vec::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for (job, step) in all_steps(&text) {
            if !step.run.contains("-p rdec-cli -- types") {
                continue;
            }
            checked += 1;
            if !step.run.contains("$LASTEXITCODE") {
                problems.push(format!(
                    "{}: job `{}` step `{}` runs the types gate without checking $LASTEXITCODE",
                    f.display(),
                    job,
                    step.name
                ));
            }
            if step.shell.as_deref().map(|s| s == "bash").unwrap_or(false) {
                problems.push(format!(
                    "{}: job `{}` step `{}` uses $LASTEXITCODE in a bash shell",
                    f.display(),
                    job,
                    step.name
                ));
            }
        }
    }
    assert!(checked > 0, "no `rdec types` step in any workflow file");
    assert!(
        problems.is_empty(),
        "types steps that launder their exit code:\n{}",
        problems.join("\n")
    );
}

#[test]
fn the_types_baseline_file_exists_and_is_not_a_comment_block() {
    // The CI step's exit code comes from this file. Missing file = the step errors
    // (good); a file of only comments = a gate that checks nothing, so the ratchet
    // loader refuses it and this test says so from the repo root too.
    // `.github/workflows` -> `.github` -> the repo root: TWO pops. The version of
    // this test that got it wrong is the evidence for writing it this way: it looked
    // for `.github/.gate/types-baseline` and reported the baseline missing.
    let wf = workflow_dir().expect("workflows dir");
    let root = wf
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| panic!("cannot walk up from {}", wf.display()))
        .to_path_buf();
    let path = root.join(".gate").join("types-baseline");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} - the types CI step's baseline is missing",
            path.display()
        )
    });
    let rows: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    assert!(!rows.is_empty(), "{} has no data rows", path.display());
    for r in &rows {
        let mut cols = r.split_whitespace();
        let (Some(key), Some(cnt)) = (cols.next(), cols.next()) else {
            panic!("{}: baseline row too short: {r}", path.display());
        };
        assert!(
            cnt.parse::<usize>().is_ok(),
            "{}: baseline count is not a number: {r}",
            path.display()
        );
        let printable = ["signedness", "width", "pointer", "float", "void-return"]
            .iter()
            .any(|known| key.ends_with(&format!("|{known}")));
        if printable {
            continue;
        }
        panic!(
            "{}: baseline names a family the scorer cannot print: {key}",
            path.display()
        );
    }
}
