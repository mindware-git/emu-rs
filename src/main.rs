use emu_rs::{decode, Cpu};
use std::{env, fs, path::Path};

fn str_inst_to_bin(hex_str: &str) -> u32 {
    let inst: u32 = u32::from_str_radix(hex_str, 16).expect("hex value should come here");
    inst
}
fn main() {
    let mut cpu = Cpu::default();

    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2);
    let file_path = Path::new(&args[1]);
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");

    for line in contents.lines() {
        let inst = str_inst_to_bin(line);
        decode(&mut cpu, inst);
    }
}
