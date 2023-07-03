// https://github.com/riscv-software-src/riscv-tests/tree/master/isa

use emu_rs::{decode, Core};
use std::{fs, path::Path};

#[test]
fn test_32i_isa() {
    assert_eq!(0, 0);
    let tc_names: [&str; 39] = [
        "add", "addi", "and", "andi", "auipc", "beq", "bge", "bgeu", "blt", "bltu", "bne", "lb",
        "lbu", "lh", "lhu", "lw", "lui", "or", "ori", "sb", "sh", "sw", "sll", "slli", "slt",
        "slti", "sltiu", "sltu", "sra", "srai", "srl", "srli", "sub", "xor", "xori", "jal", "jalr",
        "fence_i", "ma_data",
    ];
    for tc_name in tc_names {
        let mut tc_bin: String = "tests/res/rv32ui-p-".to_owned();
        tc_bin.push_str(tc_name);
        tc_bin.push_str(".bin");
        println!("{}", tc_bin);
        let file_path = Path::new(&tc_bin);
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
                // fence
                cpu.int_reg.pc += 4;
                break;
            }
            decode(&mut cpu, inst);
        }
        // pass condition
        // li	gp,1
        // li	a7,93
        // li	a0,0
        for _ in 0..3 {
            let inst = cpu.fetch();
            decode(&mut cpu, inst);
        }
        assert_eq!(cpu.int_reg.x[17], 93);
        assert_eq!(cpu.int_reg.x[10], 0);
    }
}
