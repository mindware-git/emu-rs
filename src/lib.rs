const REG_SIZE: usize = 32;

#[derive(Default)]
pub struct Cpu {
    pub reg: Register,
}

#[derive(Default)]
pub struct Register {
    pub pc: u32,
    pub x: [u32; REG_SIZE],
}

fn parse_itype(inst: u32) -> (u32, usize, usize) {
    let imm = inst >> 20;
    let rd = usize::try_from(floor_mask_value(inst, 7, 11)).unwrap();
    let rs1 = usize::try_from(floor_mask_value(inst, 15, 19)).unwrap();
    (imm, rd, rs1)
}

fn parse_utype(inst: u32) -> (u32, usize) {
    let imm = inst >> 12;
    let rd = usize::try_from(floor_mask_value(inst, 7, 11)).unwrap();
    (imm, rd)
}

fn execute_lui(reg: &mut Register, inst: u32) {
    let (imm, rd) = parse_utype(inst);
    reg.x[rd] = imm << 12;
}

fn execute_auipc(reg: &mut Register, inst: u32) {
    let (imm, rd) = parse_utype(inst);
    reg.x[rd] = imm << 12 + (reg.pc | bit_gen(0, 12));
}
fn execute_jal(reg: &mut Register, inst: u32) {
    let (imm, rd) = parse_utype(inst);
    reg.x[rd] = reg.pc + 4;
    reg.pc = imm;
}

fn execute_jalr(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    reg.x[rd] = reg.pc + 4;
    reg.pc = imm + reg.x[rs1];
}

fn execute_addi(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    reg.x[rd] = reg.x[rs1] + imm;
}

fn execute_slti(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let iimm: i32 = imm as i32;
    let ix: i32 = reg.x[rs1] as i32;
    if ix < iimm {
        reg.x[rd] = 1;
    } else {
        reg.x[rd] = 0;
    }
}
fn execute_sltiu(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    if reg.x[rs1] < imm {
        reg.x[rd] = 1;
    } else {
        reg.x[rd] = 0;
    }
}
fn execute_xori(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let iimm: i32 = imm as i32;
    let ix: i32 = reg.x[rs1] as i32;
    let ret: u32 = (ix ^ iimm) as u32;
    reg.x[rd] = ret;
}
fn execute_ori(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let iimm: i32 = imm as i32;
    let ix: i32 = reg.x[rs1] as i32;
    let ret: u32 = (ix | iimm) as u32;
    reg.x[rd] = ret;
}
fn execute_andi(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let iimm: i32 = imm as i32;
    let ix: i32 = reg.x[rs1] as i32;
    let ret: u32 = (ix & iimm) as u32;
    reg.x[rd] = ret;
}

fn bit_gen(start: u32, end: u32) -> u32 {
    assert!(start <= end);
    if end == 31 && start == 0 {
        return u32::MAX;
    }
    let rv = ((1 << (end - start + 1)) - 1) << start;
    rv
}

fn mask_value(val: u32, start: u32, end: u32) -> u32 {
    val & bit_gen(start, end)
}

fn floor_mask_value(val: u32, start: u32, end: u32) -> u32 {
    mask_value(val, start, end) >> start
}

pub fn decode(cpu: &mut Cpu, inst: u32) {
    let reg = &mut cpu.reg;
    reg.pc += 1;
    let opcode = bit_gen(0, 6);

    match inst & opcode {
        0b0110111 => {
            execute_lui(reg, inst);
        }
        0b0010111 => {
            execute_auipc(reg, inst);
        }
        0b1101111 => {
            execute_jal(reg, inst);
        }
        0b1100111 => {
            execute_jalr(reg, inst);
        }
        // 0b1100011 =>
        // // branch
        // {
        //     match floor_mask_value(inst, 12, 14) {
        //         0b000 => execute_beq(reg, inst),
        //         0b001 => execute_bne(reg, inst),
        //         0b100 => execute_blt(reg, inst),
        //         0b101 => execute_bge(reg, inst),
        //         0b110 => execute_bltu(reg, inst),
        //         0b111 => execute_bgeu(reg, inst),
        //         _ => panic!("unsupported branch type"),
        //     }
        // }
        // 0b0000011 =>
        // // load
        // {
        //     match floor_mask_value(inst, 12, 14) {
        //         0b000 => execute_lb(reg, inst),
        //         0b001 => execute_lh(reg, inst),
        //         0b010 => execute_lw(reg, inst),
        //         0b100 => execute_lbu(reg, inst),
        //         0b101 => execute_lhu(reg, inst),
        //         _ => panic!("unsupported load type"),
        //     }
        // }
        // 0b0100011 => {
        //     // store
        //     {
        //         match floor_mask_value(inst, 12, 14) {
        //             0b000 => execute_sb(reg, inst),
        //             0b001 => execute_sh(reg, inst),
        //             0b010 => execute_sw(reg, inst),
        //             _ => panic!("unsupported store type"),
        //         }
        //     }
        // }
        0b0010011 =>
        // I-type
        {
            match floor_mask_value(inst, 12, 14) {
                0b000 => execute_addi(reg, inst),
                0b010 => execute_slti(reg, inst),
                0b011 => execute_sltiu(reg, inst),
                0b100 => execute_xori(reg, inst),
                0b110 => execute_ori(reg, inst),
                0b111 => execute_andi(reg, inst),

                // 0b001 => execute_slli(reg, inst),
                // 0b101 => execute_sri(reg, inst),
                _ => panic!("unsupported I-type"),
            }
        }
        // 0b0110011 =>
        // // S-type
        // {
        //     match floor_mask_value(inst, 12, 14) {
        //         0b000 => execute_add_sub(reg, inst),
        //         _ => panic!("unsupported S-type"),
        //     }
        // }
        // 0b0001111 => {}
        // 0b1110011 => {}
        _ => panic!("invalid opcode"),
    }
}

#[test]
fn test_bit_gen() {
    assert_eq!(bit_gen(0, 0), 1);
    assert_eq!(bit_gen(0, 31), std::u32::MAX);
    assert_eq!(bit_gen(1, 2), 6);
}

#[test]
fn decode_addi() {
    let mut cpu = Cpu::default();

    //addi x1, x0, 1000
    let inst: u32 = 0b111110100000000000000010010011;
    decode(&mut cpu, inst);
    assert_eq!(cpu.reg.pc, 1);
    assert_eq!(cpu.reg.x[1], 1000);
}

#[test]
fn reg_init() {
    let reg = Register::default();

    assert_eq!(reg.pc, 0);
    for iter in reg.x {
        assert_eq!(iter, 0);
    }
}
