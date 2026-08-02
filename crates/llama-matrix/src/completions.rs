//! Shell completions and man page, generated from the same clap definition as the
//! CLI so they never drift from the actual command surface. `--man` and `--llm`
//! are handled as intercepted flags in `main` (before clap parses) so they work
//! from anywhere and never leak into the completion list.

use std::io;

use clap::CommandFactory;
use clap_complete::Shell;

use crate::Cli;

/// Print a completion script for `shell` to stdout.
pub fn print_completions(shell: Shell) {
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    clap_complete::generate(shell, &mut cmd, name, &mut io::stdout());
}

/// Print the roff man page to stdout.
pub fn print_man() -> anyhow::Result<()> {
    clap_mangen::Man::new(Cli::command()).render(&mut io::stdout())?;
    Ok(())
}
