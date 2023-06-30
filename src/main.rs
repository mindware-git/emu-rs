use clap::Parser;
use emu_rs::{decode, Core};
use std::{fs, path::Path};

fn str_inst_to_bin(hex_str: &str) -> u32 {
    let inst: u32 = u32::from_str_radix(hex_str, 16).expect("hex value should come here");
    inst
}

/// This program is for GPU reasearch (RV32IV + SIMT).
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Binary file to simulate ISA
    #[arg(short, long)]
    path: String,

    #[arg(short, long, default_value_t = false)]
    text_mode: bool,

    #[arg(short, long, default_value_t = 32)]
    mem_mega_size: u32,
}

fn main() {
    let args = Args::parse();
    let mut cpu = Core::default();
    let file_path = Path::new(&args.path);

    if args.text_mode {
        let text_file =
            fs::read_to_string(file_path).expect("Should have been able to read the file");
        for line in text_file.lines() {
            let inst = str_inst_to_bin(line);
            decode(&mut cpu, inst);
        }
    } else {
        let f = fs::read(file_path).expect("There was a problem opening the file");
        cpu.mem = f.to_owned();
        let min_mem_byte = (args.mem_mega_size * 1024 * 1024) as usize;
        if cpu.mem.len() < min_mem_byte {
            cpu.mem.resize(min_mem_byte, 0);
        }
        loop {
            let inst = cpu.fetch();
            println!(
                "pc :{:#08x}, inst :{:#08x}, {:#032b}",
                cpu.int_reg.pc, inst, inst
            );
            if inst == 0x0ff0000f {
                break;
            }
            // cpu.int_reg.pc += 4;
            decode(&mut cpu, inst);
        }
    }
}
