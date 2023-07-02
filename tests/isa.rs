// https://github.com/riscv-software-src/riscv-tests/tree/master/isa

use emu_rs::{decode, Core};
use std::{fs, path::Path};

#[test]
fn test_32i_isa() {
    assert_eq!(0, 0);
    let tc_names: [&str; 39] = [
        "add", "addi", "and", "andi", "auipc", "beq", "bge", "bgeu", "blt", "bltu", "bne",
        "fence_i", "jal", "jalr", "lb", "lbu", "lh", "lhu", "lw", "lui", "ma_data", "or", "ori",
        "sb", "sh", "sw", "sll", "slli", "slt", "slti", "sltiu", "sltu", "sra", "srai", "srl",
        "srli", "sub", "xor", "xori",
    ];
    for tc_name in tc_names {
        let file_path = Path::new(tc_name);
        let f = fs::read(file_path).expect("There was a problem opening the file");
        let mut cpu = Core::default();
        cpu.mem = f.to_owned();

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
        // pass condition
    }
}
