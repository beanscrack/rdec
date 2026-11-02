//! Workflow-file hygiene.
//!
//! A colon-space inside an unquoted `name:` makes a GitHub Actions file invalid,
//! and the failure mode is the worst kind: the run reports `failure` with zero
//! jobs and no step logs, so nothing tells you the file did not parse. This
//! happened with `name: objdump-check (M1: instruction boundaries ...)`.
//!
//! Parsing YAML is not worth a dependency; this checks the one trap that
//! produces a silent, log-less failure.

/// Tests run from the crate directory, so find the repo root by walking up.
/// Doing this by hand is the point: a test that silently finds no workflow
/// files passes without checking anything.
fn workflow_dir() -> Option<std::path::PathBuf> {
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

fn workflow_files() -> Vec<std::path::PathBuf> {
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

#[test]
fn step_names_do_not_contain_an_unquoted_colon_space() {
    let files = workflow_files();
    assert!(
        !files.is_empty(),
        "no workflow files found from {} - the test is checking nothing",
        std::env::current_dir().unwrap().display()
    );
    let mut bad = Vec::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let Some(rest) = line
                .trim_start()
                .strip_prefix("- name:")
                .or(line.trim_start().strip_prefix("name:"))
            else {
                continue;
            };
            let v = rest.trim();
            if v.is_empty() || v.starts_with('\'') || v.starts_with('"') {
                continue;
            }
            if v.contains(": ") {
                bad.push(format!("{}:{}: {line}", f.display(), i + 1));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "unquoted `name:` values containing ': ':\n{}",
        bad.join("\n")
    );
}

#[test]
fn every_step_that_needs_a_windows_shell_declares_it() {
    // The default shell on windows-latest is pwsh; steps here use PowerShell
    // idioms (fc.exe, Copy-Item, `(Get-ChildItem ...)`), and a `bash`-style
    // construct silently changes meaning.
    let files = workflow_files();
    assert!(
        !files.is_empty(),
        "no workflow files found from {} - the test is checking nothing",
        std::env::current_dir().unwrap().display()
    );
    let mut problems = Vec::new();
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let l = line.trim();
            if l.starts_with("fc.exe") || l.starts_with("Copy-Item") || l.contains("Get-ChildItem")
            {
                // Look back only to the START of this step: an earlier step's
                // shell declaration does not apply to this one.
                let lines: Vec<&str> = text.lines().collect();
                let mut shell = None;
                for j in (0..i).rev() {
                    let t = lines[j].trim_start();
                    if t.starts_with("- name:") {
                        break;
                    }
                    if let Some(v) = t.strip_prefix("shell:") {
                        shell = Some(v.trim().to_string());
                    }
                }
                match shell.as_deref() {
                    Some("powershell") | Some("pwsh") => {}
                    other => problems.push(format!(
                        "{}:{}: PowerShell syntax, shell = {:?} (run: {})",
                        f.display(),
                        i + 1,
                        other,
                        l
                    )),
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "PowerShell syntax without a declared shell:\n{}",
        problems.join("\n")
    );
}
