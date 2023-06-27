use std::num::Wrapping;

const REG_SIZE: usize = 32;

#[derive(Default)]
pub struct Cpu {
    pub reg: Register,
    pub mem: Vec<u8>,
}
impl Cpu {
    pub fn fetch(&self) -> u32 {
        let pc = self.reg.pc as usize;
        let code: [u8; 4] = self.mem[pc..pc + 4]
            .try_into()
            .expect("slice with incorrect length");
        u32::from_le_bytes(code)
    }
}
#[derive(Default)]
pub struct Register {
    pub pc: u32,
    pub x: [u32; REG_SIZE],
}

fn parse_itype(inst: u32) -> (u32, usize, usize) {
    let imm = inst >> 20;
    let rd = floor_mask_value(inst, 7, 11) as usize;
    let rs1 = floor_mask_value(inst, 15, 19) as usize;
    (imm, rd, rs1)
}

fn parse_rtype(inst: u32) -> (u32, usize, usize, usize) {
    let funct7 = inst >> 25;
    let rd = floor_mask_value(inst, 7, 11) as usize;
    let rs1 = floor_mask_value(inst, 15, 19) as usize;
    let rs2 = floor_mask_value(inst, 20, 24) as usize;
    (funct7, rd, rs1, rs2)
}

fn parse_utype(inst: u32) -> (u32, usize) {
    let imm = inst >> 12;
    let rd = floor_mask_value(inst, 7, 11) as usize;
    (imm, rd)
}

fn parse_btype(inst: u32) -> (u32, usize, usize) {
    let imm = (floor_mask_value(inst, 31, 31) << 12)
        + (floor_mask_value(inst, 25, 30) << 5)
        + (floor_mask_value(inst, 8, 11) << 1)
        + (floor_mask_value(inst, 7, 7) << 11);
    let rs1 = floor_mask_value(inst, 7, 11) as usize;
    let rs2 = floor_mask_value(inst, 15, 19) as usize;
    (imm, rs1, rs2)
}

fn execute_lui(reg: &mut Register, inst: u32) {
    let (imm, rd) = parse_utype(inst);
    reg.x[rd] = imm << 12;
}

fn execute_auipc(reg: &mut Register, inst: u32) {
    let (imm, rd) = parse_utype(inst);
    reg.x[rd] = imm << 12 + reg.pc;
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

fn cal_addi(val: u32, imm: u32) -> u32 {
    let mut sext: u32 = imm;
    if (imm & bit_gen(11, 11)) > 0 {
        sext = imm | bit_gen(12, 31);
    }
    let may_overflow = Wrapping(sext) + Wrapping(val);
    may_overflow.0
}
fn cal_add(val1: u32, val2: u32) -> u32 {
    let may_overflow = Wrapping(val1) + Wrapping(val2);
    may_overflow.0
}
fn cal_and(val1: u32, val2: u32) -> u32 {
    val1 & val2
}

fn execute_addi(reg: &mut Register, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    reg.x[rd] = cal_addi(reg.x[rs1], imm);
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
    reg.x[rd] = cal_and(reg.x[rs1], imm);
}

fn execute_add_sub(reg: &mut Register, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    if (funct7 | bit_gen(30, 30)) > 0 {
        reg.x[rd] = cal_add(reg.x[rs1], reg.x[rs2]);
    } else {
        reg.x[rd] = cal_add(reg.x[rs1], reg.x[rs2]);
    }
}
fn execute_beq(reg: &mut Register, inst: u32) {
    let (imm, rs1, rs2) = parse_btype(inst);
    if rs1 == rs2 {
        reg.pc += imm;
    }
}

fn execute_and(reg: &mut Register, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    assert!(funct7 == 0);
    reg.x[rd] = cal_and(reg.x[rs1], reg.x[rs2]);
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
        0b1100011 =>
        // branch
        {
            match floor_mask_value(inst, 12, 14) {
                0b000 => execute_beq(reg, inst),
                //         0b001 => execute_bne(reg, inst),
                //         0b100 => execute_blt(reg, inst),
                //         0b101 => execute_bge(reg, inst),
                //         0b110 => execute_bltu(reg, inst),
                //         0b111 => execute_bgeu(reg, inst),
                _ => panic!("unsupported branch type"),
            }
        }
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
        0b0110011 =>
        // S-type
        {
            match floor_mask_value(inst, 12, 14) {
                0b000 => execute_add_sub(reg, inst),
                0b111 => execute_and(reg, inst),
                _ => panic!("unsupported S-type"),
            }
        }
        // 0b0001111 => {}
        // 0b1110011 => {}
        _ => panic!("invalid opcode"),
    }

    reg.pc += 4;
}

#[test]
fn test_bit_gen() {
    assert_eq!(bit_gen(0, 0), 1);
    assert_eq!(bit_gen(0, 31), std::u32::MAX);
    assert_eq!(bit_gen(1, 2), 6);
}

#[test]
fn reg_init() {
    let reg = Register::default();

    assert_eq!(reg.pc, 0);
    for iter in reg.x {
        assert_eq!(iter, 0);
    }
}

#[test]
fn mem_init() {
    let mut cpu = Cpu::default();
    cpu.mem.resize(128, 0);
    assert_eq!(cpu.mem.len(), 128);
}

// https://github.com/riscv-software-src/riscv-tests/tree/master/isa
#[test]
fn test_cal_addi() {
    assert_eq!(0x00000000, cal_addi(0x00000000, 0x000));
    assert_eq!(0x00000002, cal_addi(0x00000001, 0x001));
    assert_eq!(0x0000000a, cal_addi(0x00000003, 0x007));
    assert_eq!(0xfffff800, cal_addi(0x00000000, 0x800));
    assert_eq!(0x80000000, cal_addi(0x80000000, 0x000));
    assert_eq!(0x7ffff800, cal_addi(0x80000000, 0x800));
    assert_eq!(0x000007ff, cal_addi(0x00000000, 0x7ff));
    assert_eq!(0x7fffffff, cal_addi(0x7fffffff, 0x000));
    assert_eq!(0x800007fe, cal_addi(0x7fffffff, 0x7ff));
    assert_eq!(0x800007ff, cal_addi(0x80000000, 0x7ff));
    assert_eq!(0x7ffff7ff, cal_addi(0x7fffffff, 0x800));
    assert_eq!(0xffffffff, cal_addi(0x00000000, 0xfff));
    assert_eq!(0x00000000, cal_addi(0xffffffff, 0x001));
    assert_eq!(0xfffffffe, cal_addi(0xffffffff, 0xfff));
    assert_eq!(0x80000000, cal_addi(0x7fffffff, 0x001));
}

#[test]
fn test_cal_add() {
    assert_eq!(0x00000000, cal_add(0x00000000, 0x00000000));
    assert_eq!(0x00000002, cal_add(0x00000001, 0x00000001));
    assert_eq!(0x0000000a, cal_add(0x00000003, 0x00000007));
    assert_eq!(0xffff8000, cal_add(0x00000000, 0xffff8000));
    assert_eq!(0x80000000, cal_add(0x80000000, 0x00000000));
    assert_eq!(0x7fff8000, cal_add(0x80000000, 0xffff8000));
    assert_eq!(0x00007fff, cal_add(0x00000000, 0x00007fff));
    assert_eq!(0x7fffffff, cal_add(0x7fffffff, 0x00000000));
    assert_eq!(0x80007ffe, cal_add(0x7fffffff, 0x00007fff));
    assert_eq!(0x80007fff, cal_add(0x80000000, 0x00007fff));
    assert_eq!(0x7fff7fff, cal_add(0x7fffffff, 0xffff8000));
    assert_eq!(0xffffffff, cal_add(0x00000000, 0xffffffff));
    assert_eq!(0x00000000, cal_add(0xffffffff, 0x00000001));
    assert_eq!(0xfffffffe, cal_add(0xffffffff, 0xffffffff));
    assert_eq!(0x80000000, cal_add(0x00000001, 0x7fffffff));
}
#[test]
fn test_cal_and() {
    assert_eq!(0x0f000f00, cal_and(0xff00ff00, 0x0f0f0f0f));
    assert_eq!(0x00f000f0, cal_and(0x0ff00ff0, 0xf0f0f0f0));
    assert_eq!(0x000f000f, cal_and(0x00ff00ff, 0x0f0f0f0f));
    assert_eq!(0xf000f000, cal_and(0xf00ff00f, 0xf0f0f0f0));
}

#[test]
fn decode_addi() {
    let mut cpu = Cpu::default();

    //addi x1, x0, 1000
    let inst: u32 = 0b111110100000000000000010010011;
    decode(&mut cpu, inst);
    assert_eq!(cpu.reg.pc, 4);
    assert_eq!(cpu.reg.x[1], 1000);
}
