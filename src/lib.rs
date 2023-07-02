use std::num::Wrapping;

const REG_SIZE: usize = 32;

#[derive(Default)]
pub struct Core {
    pub int_reg: IntRegister,
    pub vec_reg: VecRegister,
    pub mem: Vec<u8>,
}
impl Core {
    pub fn fetch(&self) -> u32 {
        let pc = self.int_reg.pc as usize;
        assert!(pc % 4 == 0);
        assert!(pc < self.mem.len());
        let code: [u8; 4] = self.mem[pc..pc + 4]
            .try_into()
            .expect("slice with incorrect length");
        u32::from_le_bytes(code)
    }
}
#[derive(Default)]
pub struct IntRegister {
    pub pc: u32,
    pub x: [u32; REG_SIZE],
}
#[derive(Default)]
pub struct VecRegister {
    pub x: [u128; REG_SIZE],
}

/// this can not handle u32::MAX because const fn not accept if
const fn bit_gen(start: u32, end: u32) -> u32 {
    assert!(start <= end);
    ((1 << (end - start + 1)) - 1) << start
}

const fn mask_value(val: u32, start: u32, end: u32) -> u32 {
    val & bit_gen(start, end)
}

const fn floor_mask_value(val: u32, start: u32, end: u32) -> u32 {
    mask_value(val, start, end) >> start
}

// basically rust integer is arithmetic shift
fn sigend_ext(val: u32, leftmost: u32) -> u32 {
    let mut sext = val;
    if (val & bit_gen(leftmost, leftmost)) > 0 {
        sext = val | bit_gen(leftmost + 1, 31);
    }
    sext
}

fn add_ignore_overflow(val1: u32, val2: u32) -> u32 {
    (Wrapping(val1) + Wrapping(val2)).0
}

fn parse_rtype(inst: u32) -> (u32, usize, usize, usize) {
    let funct7 = inst >> 25;
    let rd = floor_mask_value(inst, 7, 11) as usize;
    let rs1 = floor_mask_value(inst, 15, 19) as usize;
    let rs2 = floor_mask_value(inst, 20, 24) as usize;
    (funct7, rd, rs1, rs2)
}

fn parse_itype(inst: u32) -> (i32, usize, usize) {
    let imm = (inst as i32) >> 20;
    let rd = floor_mask_value(inst, 7, 11) as usize;
    let rs1 = floor_mask_value(inst, 15, 19) as usize;
    (imm, rd, rs1)
}

fn parse_stype(inst: u32) -> (u32, usize, usize) {
    let imm = floor_mask_value(inst, 7, 11) + (floor_mask_value(inst, 25, 31) << 5);
    let rs1 = floor_mask_value(inst, 15, 19) as usize;
    let rs2 = floor_mask_value(inst, 20, 24) as usize;
    (imm, rs1, rs2)
}

fn parse_utype(inst: u32) -> (u32, usize) {
    let imm = inst >> 12;
    let rd = floor_mask_value(inst, 7, 11) as usize;
    (imm, rd)
}

fn parse_jtype(inst: u32) -> i32 {
    let imm = (inst as i32 >> 11) & (bit_gen(20, 31) as i32);
    let mid = (floor_mask_value(inst, 21, 30) << 1)
        + (floor_mask_value(inst, 20, 20) << 11)
        + (floor_mask_value(inst, 12, 19) << 12);
    imm | mid as i32
}

fn parse_btype(inst: u32) -> (i32, usize, usize) {
    let mid = (floor_mask_value(inst, 7, 7) << 11) + (floor_mask_value(inst, 8, 11) << 1);
    let mut imm = (inst as i32) >> 20;
    imm = imm & (bit_gen(5, 31) as i32);
    imm |= mid as i32;
    let rs1 = floor_mask_value(inst, 15, 19) as usize;
    let rs2 = floor_mask_value(inst, 20, 24) as usize;
    (imm, rs1, rs2)
}

fn execute_lui(reg: &mut IntRegister, inst: u32) {
    let (imm, rd) = parse_utype(inst);
    reg.x[rd] = imm << 12;
}

fn execute_auipc(reg: &mut IntRegister, inst: u32) {
    let (imm, rd) = parse_utype(inst);
    reg.x[rd] = add_ignore_overflow(imm << 12, reg.pc - 4)
}

fn execute_jal(reg: &mut IntRegister, inst: u32) {
    let imm = parse_jtype(inst);
    reg.pc -= 4;
    reg.pc = (reg.pc as i32 + imm) as u32;
}

fn execute_jalr(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    if rd != 0 {
        reg.x[rd] = reg.pc;
    }
    reg.pc -= 4;
    reg.pc = add_ignore_overflow(imm as u32, reg.x[rs1]);
}

fn cal_and(val1: u32, val2: u32) -> u32 {
    val1 & val2
}

fn execute_addi(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    reg.x[rd] = add_ignore_overflow(reg.x[rs1], imm as u32);
}

fn execute_slti(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let ix: i32 = reg.x[rs1] as i32;
    if ix < imm {
        reg.x[rd] = 1;
    } else {
        reg.x[rd] = 0;
    }
}
fn execute_sltiu(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    if reg.x[rs1] < imm as u32 {
        reg.x[rd] = 1;
    } else {
        reg.x[rd] = 0;
    }
}
fn execute_xori(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let iimm: i32 = imm as i32;
    let ix: i32 = reg.x[rs1] as i32;
    let ret: u32 = (ix ^ iimm) as u32;
    reg.x[rd] = ret;
}
fn execute_ori(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let iimm: i32 = imm as i32;
    let ix: i32 = reg.x[rs1] as i32;
    let ret: u32 = (ix | iimm) as u32;
    reg.x[rd] = ret;
}
fn execute_andi(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    reg.x[rd] = reg.x[rs1] & imm as u32;
}
fn execute_slli(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    reg.x[rd] = reg.x[rs1] << imm;
}

fn execute_srli_a(reg: &mut IntRegister, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let uimm = imm as u32;
    let shamt = uimm & bit_gen(0, 4);
    if uimm & bit_gen(10, 10) > 0 {
        reg.x[rd] = (reg.x[rs1] as i32 >> shamt) as u32;
    } else {
        reg.x[rd] = reg.x[rs1] >> shamt;
    }
}

fn execute_add_sub(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    if (funct7 & bit_gen(5, 5)) > 0 {
        reg.x[rd] = reg.x[rs1] - reg.x[rs2];
    } else {
        reg.x[rd] = add_ignore_overflow(reg.x[rs1], reg.x[rs2]);
    }
}
fn execute_sll(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    assert_eq!(funct7, 0);
    reg.x[rd] = reg.x[rs1] << reg.x[rs2];
}

fn execute_slt(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    assert_eq!(funct7, 0);
    if (reg.x[rs1] as i32) < (reg.x[rs2] as i32) {
        reg.x[rd] = 1;
    } else {
        reg.x[rd] = 0;
    }
}
fn execute_sltu(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    assert_eq!(funct7, 0);
    if reg.x[rs1] < reg.x[rs2] {
        reg.x[rd] = 1;
    } else {
        reg.x[rd] = 0;
    }
}

fn execute_xor(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    assert_eq!(funct7, 0);
    reg.x[rd] = reg.x[rs1] ^ reg.x[rs2];
}

fn execute_srl_a(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    if (funct7 & bit_gen(5, 5)) > 0 {
        reg.x[rd] = ((reg.x[rs1] as i32) >> (reg.x[rs2] as i32)) as u32;
    } else {
        reg.x[rd] = reg.x[rs1] >> reg.x[rs2];
    }
}

fn execute_or(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    assert_eq!(funct7, 0);
    reg.x[rd] = reg.x[rs1] | reg.x[rs2];
}

fn execute_and(reg: &mut IntRegister, inst: u32) {
    let (funct7, rd, rs1, rs2) = parse_rtype(inst);
    assert!(funct7 == 0);
    reg.x[rd] = cal_and(reg.x[rs1], reg.x[rs2]);
}

fn execute_beq(reg: &mut IntRegister, inst: u32) {
    let (imm, rs1, rs2) = parse_btype(inst);
    if reg.x[rs1] == reg.x[rs2] {
        reg.pc -= 4;
        reg.pc = (reg.pc as i32 + imm) as u32;
    }
}
fn execute_bne(reg: &mut IntRegister, inst: u32) {
    let (imm, rs1, rs2) = parse_btype(inst);
    if reg.x[rs1] != reg.x[rs2] {
        reg.pc -= 4;
        reg.pc = (reg.pc as i32 + imm) as u32;
    }
}

fn execute_blt(reg: &mut IntRegister, inst: u32) {
    let (imm, rs1, rs2) = parse_btype(inst);
    if (reg.x[rs1] as i32) < (reg.x[rs2] as i32) {
        reg.pc -= 4;
        reg.pc = (reg.pc as i32 + imm) as u32;
    }
}
fn execute_bge(reg: &mut IntRegister, inst: u32) {
    let (imm, rs1, rs2) = parse_btype(inst);
    if (reg.x[rs1] as i32) >= (reg.x[rs2] as i32) {
        reg.pc -= 4;
        reg.pc = (reg.pc as i32 + imm) as u32;
    }
}
fn execute_bltu(reg: &mut IntRegister, inst: u32) {
    let (imm, rs1, rs2) = parse_btype(inst);
    if reg.x[rs1] < reg.x[rs2] {
        reg.pc -= 4;
        reg.pc = (reg.pc as i32 + imm) as u32;
    }
}
fn execute_bgeu(reg: &mut IntRegister, inst: u32) {
    let (imm, rs1, rs2) = parse_btype(inst);
    if reg.x[rs1] >= reg.x[rs2] {
        reg.pc -= 4;
        reg.pc = (reg.pc as i32 + imm) as u32;
    }
}

fn execute_lb(cpu: &mut Core, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] as i32 + imm) as usize;
    let data = cpu.mem[mem_addr] as u32;
    cpu.int_reg.x[rd] = sigend_ext(data, 7);
}

fn execute_lh(cpu: &mut Core, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] as i32 + imm) as usize;
    let data: [u8; 2] = cpu.mem[mem_addr..mem_addr + 2]
        .try_into()
        .expect("slice with incorrect length");
    let valid_data = u16::from_le_bytes(data) as u32;
    cpu.int_reg.x[rd] = sigend_ext(valid_data, 15);
}

fn execute_lw(cpu: &mut Core, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] as i32 + imm) as usize;
    let data: [u8; 4] = cpu.mem[mem_addr..mem_addr + 4]
        .try_into()
        .expect("slice with incorrect length");
    let valid_data = u32::from_le_bytes(data);
    cpu.int_reg.x[rd] = sigend_ext(valid_data, 15);
}

fn execute_lbu(cpu: &mut Core, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] as i32 + imm) as usize;
    let data = cpu.mem[mem_addr] as u32;
    cpu.int_reg.x[rd] = data;
}

fn execute_lhu(cpu: &mut Core, inst: u32) {
    let (imm, rd, rs1) = parse_itype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] as i32 + imm) as usize;
    let data = cpu.mem[mem_addr] as u32;
    cpu.int_reg.x[rd] = data;
}

fn execute_sb(cpu: &mut Core, inst: u32) {
    let (imm, rs1, rs2) = parse_stype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] + sigend_ext(imm, 11)) as usize;
    let data = cpu.int_reg.x[rs2] as u8;
    cpu.mem[mem_addr] = data;
}

fn execute_sh(cpu: &mut Core, inst: u32) {
    let (imm, rs1, rs2) = parse_stype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] + sigend_ext(imm, 11)) as usize;
    let mut data = cpu.int_reg.x[rs2] as u16;
    cpu.mem[mem_addr] = data as u8;
    data = data >> 8;
    cpu.mem[mem_addr + 1] = data as u8;
}

fn execute_sw(cpu: &mut Core, inst: u32) {
    let (imm, rs1, rs2) = parse_stype(inst);
    let mem_addr = (cpu.int_reg.x[rs1] + sigend_ext(imm, 11)) as usize;
    let mut data = cpu.int_reg.x[rs2];
    for idx in 0..4 {
        cpu.mem[mem_addr + idx] = data as u8;
        data = data >> 8;
    }
}

fn execute_i_inst(cpu: &mut Core, inst: u32) {
    let reg = &mut cpu.int_reg;

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
        // B-type
        {
            match floor_mask_value(inst, 12, 14) {
                0b000 => execute_beq(reg, inst),
                0b001 => execute_bne(reg, inst),
                0b100 => execute_blt(reg, inst),
                0b101 => execute_bge(reg, inst),
                0b110 => execute_bltu(reg, inst),
                0b111 => execute_bgeu(reg, inst),
                _ => panic!("unsupported branch type"),
            }
        }
        0b0000011 =>
        // I-type
        {
            match floor_mask_value(inst, 12, 14) {
                0b000 => execute_lb(cpu, inst),
                0b001 => execute_lh(cpu, inst),
                0b010 => execute_lw(cpu, inst),
                0b100 => execute_lbu(cpu, inst),
                0b101 => execute_lhu(cpu, inst),
                _ => panic!("unsupported load type"),
            }
        }
        0b0100011 =>
        // S-type
        {
            match floor_mask_value(inst, 12, 14) {
                0b000 => execute_sb(cpu, inst),
                0b001 => execute_sh(cpu, inst),
                0b010 => execute_sw(cpu, inst),
                _ => panic!("unsupported store type"),
            }
        }
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
                0b001 => execute_slli(reg, inst),
                0b101 => execute_srli_a(reg, inst),
                _ => panic!("unsupported I-type"),
            }
        }
        0b0110011 =>
        // R-type
        {
            match floor_mask_value(inst, 12, 14) {
                0b000 => execute_add_sub(reg, inst),
                0b001 => execute_sll(reg, inst),
                0b010 => execute_slt(reg, inst),
                0b011 => execute_sltu(reg, inst),
                0b100 => execute_xor(reg, inst),
                0b101 => execute_srl_a(reg, inst),
                0b110 => execute_or(reg, inst),
                0b111 => execute_and(reg, inst),
                _ => panic!("unsupported S-type"),
            }
        }
        0b0001111 => {
            panic!("E_CALL,BREA not implemented!");
        }
        0b1110011 => {
            // ECALL, EBREAK;
            panic!("E_CALL,BREA not implemented!");
        }
        _ => panic!("invalid opcode {:b}", inst),
    }
}

pub fn decode(cpu: &mut Core, inst: u32) {
    cpu.int_reg.pc += 4;
    execute_i_inst(cpu, inst);
}

#[test]
fn test_bit_gen() {
    assert_eq!(bit_gen(0, 0), 1);
    assert_eq!(bit_gen(1, 2), 6);
}

#[test]
fn reg_init() {
    let reg = IntRegister::default();

    assert_eq!(reg.pc, 0);
    for iter in reg.x {
        assert_eq!(iter, 0);
    }
}

#[test]
fn mem_init() {
    let mut cpu = Core::default();
    cpu.mem.resize(128, 0);
    assert_eq!(cpu.mem.len(), 128);
}

// https://github.com/riscv-software-src/riscv-tests/tree/master/isa
#[test]
fn addi() {
    let mut int_reg = IntRegister::default();
    int_reg.x[1] = 0x00000000;
    /*
        addi x2 , x1,   0x000
    */
    execute_addi(&mut int_reg, 0x00008113);
    assert_eq!(int_reg.x[2], 0x00000000);
    execute_addi(&mut int_reg, 0x80008113);
    assert_eq!(int_reg.x[2], 0xfffff800);

    int_reg.x[1] = 0x00000001;
    execute_addi(&mut int_reg, 0x00108113);
    assert_eq!(int_reg.x[2], 0x00000002);

    int_reg.x[1] = 0x80000000;
    execute_addi(&mut int_reg, 0x80008113);
    assert_eq!(int_reg.x[2], 0x7ffff800);
    execute_addi(&mut int_reg, 0x7ff08113);
    assert_eq!(int_reg.x[2], 0x800007ff);

    int_reg.x[1] = 0xffffffff;
    execute_addi(&mut int_reg, 0xfff08113);
    assert_eq!(int_reg.x[2], 0xfffffffe);
}

#[test]
fn test_add_ignore_overflow() {
    assert_eq!(0x00000000, add_ignore_overflow(0x00000000, 0x00000000));
    assert_eq!(0x00000002, add_ignore_overflow(0x00000001, 0x00000001));
    assert_eq!(0x0000000a, add_ignore_overflow(0x00000003, 0x00000007));
    assert_eq!(0xffff8000, add_ignore_overflow(0x00000000, 0xffff8000));
    assert_eq!(0x80000000, add_ignore_overflow(0x80000000, 0x00000000));
    assert_eq!(0x7fff8000, add_ignore_overflow(0x80000000, 0xffff8000));
    assert_eq!(0x00007fff, add_ignore_overflow(0x00000000, 0x00007fff));
    assert_eq!(0x7fffffff, add_ignore_overflow(0x7fffffff, 0x00000000));
    assert_eq!(0x80007ffe, add_ignore_overflow(0x7fffffff, 0x00007fff));
    assert_eq!(0x80007fff, add_ignore_overflow(0x80000000, 0x00007fff));
    assert_eq!(0x7fff7fff, add_ignore_overflow(0x7fffffff, 0xffff8000));
    assert_eq!(0xffffffff, add_ignore_overflow(0x00000000, 0xffffffff));
    assert_eq!(0x00000000, add_ignore_overflow(0xffffffff, 0x00000001));
    assert_eq!(0xfffffffe, add_ignore_overflow(0xffffffff, 0xffffffff));
    assert_eq!(0x80000000, add_ignore_overflow(0x00000001, 0x7fffffff));
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
    let mut cpu = Core::default();

    //addi x1, x0, 1000
    let inst: u32 = 0b111110100000000000000010010011;
    decode(&mut cpu, inst);
    assert_eq!(cpu.int_reg.pc, 4);
    assert_eq!(cpu.int_reg.x[1], 1000);
}

#[test]
fn decode_beq() {
    /*
        li  x1, 1;
        beq x0, x0, foo;
        addi x1, x1, 1;
        addi x1, x1, 1;
        addi x1, x1, 1;
        addi x1, x1, 1;
    foo:addi x1, x1, 1;
        addi x1, x1, 1;
    */
    let mut cpu = Core::default();
    let insts: [u32; 8] = [
        0x00100093, 0x00000a63, 0x00108093, 0x00108093, 0x00108093, 0x00108093, 0x00108093,
        0x00108093,
    ];
    for inst in insts {
        cpu.mem.push(inst as u8);
        cpu.mem.push((inst >> 8) as u8);
        cpu.mem.push((inst >> 16) as u8);
        cpu.mem.push((inst >> 24) as u8);
    }

    for _ in 0..4 {
        let inst = cpu.fetch();
        decode(&mut cpu, inst);
    }
    assert_eq!(cpu.int_reg.x[1], 3);
}

#[test]
fn decode_bge() {
    /*
        li  x1, 1;
        bge x1, x0, foo;
        addi x1, x1, 1;
        addi x1, x1, 1;
        addi x1, x1, 1;
        addi x1, x1, 1;
    foo:addi x1, x1, 1;
        addi x1, x1, 1;
    */
    let mut cpu = Core::default();
    let insts: [u32; 8] = [
        0x00100093, 0x0000da63, 0x00108093, 0x00108093, 0x00108093, 0x00108093, 0x00108093,
        0x00108093,
    ];
    for inst in insts {
        cpu.mem.push(inst as u8);
        cpu.mem.push((inst >> 8) as u8);
        cpu.mem.push((inst >> 16) as u8);
        cpu.mem.push((inst >> 24) as u8);
    }

    for _ in 0..4 {
        let inst = cpu.fetch();
        decode(&mut cpu, inst);
    }
    assert_eq!(cpu.int_reg.x[1], 3);
}
#[test]
fn test_jal1() {
    /*
    test_2:
      li  x1, 2
      li  ra, 0

      jal x4, target_2
    linkaddr_2:
      nop
      nop

      j fail

    target_2:
      la  x2, linkaddr_2
      bne x2, x4, fail
    */
    let mut cpu = Core::default();
    let insts: [u32; 10] = [
        0x00200093, 0x00000093, 0x0100026f, 0x00000013, 0x00000013, 0xfedff06f, 0x00000117,
        0x00010113, 0x00410463, 0xfddff06f,
    ];
    for inst in insts {
        cpu.mem.push(inst as u8);
        cpu.mem.push((inst >> 8) as u8);
        cpu.mem.push((inst >> 16) as u8);
        cpu.mem.push((inst >> 24) as u8);
    }

    for _ in 0..4 {
        let inst = cpu.fetch();
        decode(&mut cpu, inst);
    }
}
#[test]
fn test_jal2() {
    /*
        li  ra, 1;
        jal x0, foo;
        addi ra, ra, 1;
        addi ra, ra, 1;
        addi ra, ra, 1;
        addi ra, ra, 1;
    foo:  addi ra, ra, 1;
        addi ra, ra, 1;
    */
    let mut cpu = Core::default();
    let insts: [u32; 8] = [
        0x00100093, 0x0140006f, 0x00108093, 0x00108093, 0x00108093, 0x00108093, 0x00108093,
        0x00108093,
    ];
    for inst in insts {
        cpu.mem.push(inst as u8);
        cpu.mem.push((inst >> 8) as u8);
        cpu.mem.push((inst >> 16) as u8);
        cpu.mem.push((inst >> 24) as u8);
    }

    for _ in 0..4 {
        let inst = cpu.fetch();
        decode(&mut cpu, inst);
    }

    assert_eq!(cpu.int_reg.x[1], 3);
}

#[test]
fn test_jalr() {
    /*
        addi t2, t2, 2
        addi t0, t0, 2
        jalr t0, t2, -2
    */
    let mut cpu = Core::default();
    let insts: [u32; 3] = [0x00238393, 0x00228293, 0xffe382e7];
    for inst in insts {
        cpu.mem.push(inst as u8);
        cpu.mem.push((inst >> 8) as u8);
        cpu.mem.push((inst >> 16) as u8);
        cpu.mem.push((inst >> 24) as u8);
    }

    for _ in 0..5 {
        let inst = cpu.fetch();
        decode(&mut cpu, inst);
    }

    assert_eq!(cpu.int_reg.x[5], 14);
}
