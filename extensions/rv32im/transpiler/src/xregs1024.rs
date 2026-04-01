//! Transpiler extension for XRegs1024 64-bit instruction encoding.
//!
//! Each instruction is 2 x u32 (8 bytes):
//!
//! Low u32: Standard RISC-V bit layout with [6:0] = 0x3F (64-bit marker).
//!   Immediate bits at standard positions. Register fields hold LOW 5 bits.
//!
//! High u32:
//!   [16:10] = original 7-bit opcode
//!   [21:17] = rd high bits (bits 9:5)
//!   [26:22] = rs1 high bits (bits 9:5)
//!   [31:27] = rs2 high bits (bits 9:5)
//!
//! Full register: reg = high[4:0] << 5 | low[4:0]

use openvm_instructions::{
    instruction::Instruction,
    riscv::RV32_REGISTER_NUM_LIMBS,
    utils::isize_to_field,
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{TranspilerExtension, TranspilerOutput};

use crate::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, Rv32AuipcOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode,
    Rv32LoadStoreOpcode, ShiftOpcode,
};

const XREGS1024_MARKER: u32 = 0x3F; // bits[6:0] = 0111111

/// Transpiler extension for XRegs1024 64-bit encoded RISC-V instructions.
#[derive(Default)]
pub struct XRegs1024TranspilerExtension;

/// Decoded 64-bit instruction fields.
pub struct Decoded64 {
    pub opcode: u8,
    pub funct3: u8,
    pub funct7: u8,
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    /// Raw low u32 (for immediate extraction)
    lo: u32,
}

pub fn decode_64bit(lo: u32, hi: u32) -> Decoded64 {
    let opcode = ((hi >> 10) & 0x7F) as u8;
    let funct3 = ((lo >> 12) & 0x7) as u8;
    let funct7 = ((lo >> 25) & 0x7F) as u8;

    let rd_lo = ((lo >> 7) & 0x1F) as usize;
    let rs1_lo = ((lo >> 15) & 0x1F) as usize;
    let rs2_lo = ((lo >> 20) & 0x1F) as usize;

    let rd_hi = ((hi >> 17) & 0x1F) as usize;
    let rs1_hi = ((hi >> 22) & 0x1F) as usize;
    let rs2_hi = ((hi >> 27) & 0x1F) as usize;

    Decoded64 {
        opcode,
        funct3,
        funct7,
        rd: (rd_hi << 5) | rd_lo,
        rs1: (rs1_hi << 5) | rs1_lo,
        rs2: (rs2_hi << 5) | rs2_lo,
        lo,
    }
}

/// Extract sign-extended 12-bit I-type immediate from low u32
fn i_imm(lo: u32) -> i32 {
    (lo as i32) >> 20
}

/// Extract 12-bit S-type immediate from low u32
fn s_imm(lo: u32) -> i32 {
    let lo5 = ((lo >> 7) & 0x1F) as i32;
    let hi7 = ((lo as i32) >> 25) & 0x7F;
    (hi7 << 5) | lo5
}

/// Extract 13-bit B-type immediate from low u32
fn b_imm(lo: u32) -> i32 {
    let imm11 = ((lo >> 7) & 1) as i32;
    let imm4_1 = ((lo >> 8) & 0xF) as i32;
    let imm10_5 = ((lo >> 25) & 0x3F) as i32;
    let imm12 = ((lo >> 31) & 1) as i32;
    let imm = (imm4_1 << 1) | (imm10_5 << 5) | (imm11 << 11) | (imm12 << 12);
    if imm12 != 0 { imm - (1 << 13) } else { imm }
}

/// Extract 21-bit J-type immediate from low u32
fn j_imm(lo: u32) -> i32 {
    let imm19_12 = ((lo >> 12) & 0xFF) as i32;
    let imm11 = ((lo >> 20) & 1) as i32;
    let imm10_1 = ((lo >> 21) & 0x3FF) as i32;
    let imm20 = ((lo >> 31) & 1) as i32;
    let imm = (imm10_1 << 1) | (imm11 << 11) | (imm19_12 << 12) | (imm20 << 20);
    if imm20 != 0 { imm - (1 << 21) } else { imm }
}

/// Extract U-type immediate (upper 20 bits) from low u32
fn u_imm(lo: u32) -> u32 {
    lo & 0xFFFFF000
}

fn i12_to_u24(imm: i32) -> u32 {
    (imm as u32) & 0xffffff
}

fn nop<F: PrimeField32>() -> Instruction<F> {
    Instruction {
        opcode: SystemOpcode::PHANTOM.global_opcode(),
        ..Default::default()
    }
}

fn unimp<F: PrimeField32>() -> Instruction<F> {
    Instruction {
        opcode: SystemOpcode::TERMINATE.global_opcode(),
        c: F::TWO,
        ..Default::default()
    }
}

// RISC-V opcodes
const OP: u8 = 0b0110011;
const OP_IMM: u8 = 0b0010011;
const LOAD: u8 = 0b0000011;
const STORE: u8 = 0b0100011;
const BRANCH: u8 = 0b1100011;
const JAL: u8 = 0b1101111;
const JALR: u8 = 0b1100111;
const LUI: u8 = 0b0110111;
const AUIPC: u8 = 0b0010111;
const SYSTEM: u8 = 0b1110011;
const MISC_MEM: u8 = 0b0001111;

impl<F: PrimeField32> TranspilerExtension<F> for XRegs1024TranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.len() < 2 {
            return None;
        }

        let lo = instruction_stream[0];
        // Check for 64-bit marker
        if (lo & 0x7F) != XREGS1024_MARKER {
            return None;
        }

        let hi = instruction_stream[1];
        let d = decode_64bit(lo, hi);

        let instruction = match d.opcode {
            OP => {
                match d.funct7 {
                    0x00 | 0x20 => {
                        // Standard ALU R-type
                        let opcode = match (d.funct7, d.funct3) {
                            (0x00, 0) => BaseAluOpcode::ADD,
                            (0x20, 0) => BaseAluOpcode::SUB,
                            (0x00, 4) => BaseAluOpcode::XOR,
                            (0x00, 6) => BaseAluOpcode::OR,
                            (0x00, 7) => BaseAluOpcode::AND,
                            _ => return Some(TranspilerOutput {
                                instructions: vec![Some(unimp()), None],
                                used_u32s: 2,
                            }),
                        };
                        if d.funct3 == 1 || d.funct3 == 5 {
                            // SLL, SRL, SRA
                            let opcode = match (d.funct7, d.funct3) {
                                (0x00, 1) => ShiftOpcode::SLL,
                                (0x00, 5) => ShiftOpcode::SRL,
                                (0x20, 5) => ShiftOpcode::SRA,
                                _ => unreachable!(),
                            };
                            make_r_type(opcode.global_opcode().as_usize(), &d)
                        } else if d.funct3 == 2 || d.funct3 == 3 {
                            // SLT, SLTU
                            let opcode = match d.funct3 {
                                2 => LessThanOpcode::SLT,
                                3 => LessThanOpcode::SLTU,
                                _ => unreachable!(),
                            };
                            make_r_type(opcode.global_opcode().as_usize(), &d)
                        } else {
                            make_r_type(opcode.global_opcode().as_usize(), &d)
                        }
                    }
                    0x01 => {
                        // M-extension
                        match d.funct3 {
                            0 => make_r_type(MulOpcode::MUL.global_opcode().as_usize(), &d),
                            1 => make_r_type(MulHOpcode::MULH.global_opcode().as_usize(), &d),
                            2 => make_r_type(MulHOpcode::MULHSU.global_opcode().as_usize(), &d),
                            3 => make_r_type(MulHOpcode::MULHU.global_opcode().as_usize(), &d),
                            4 => make_r_type(DivRemOpcode::DIV.global_opcode().as_usize(), &d),
                            5 => make_r_type(DivRemOpcode::DIVU.global_opcode().as_usize(), &d),
                            6 => make_r_type(DivRemOpcode::REM.global_opcode().as_usize(), &d),
                            7 => make_r_type(DivRemOpcode::REMU.global_opcode().as_usize(), &d),
                            _ => return None,
                        }
                    }
                    _ => return None,
                }
            }
            OP_IMM => {
                match d.funct3 {
                    0 => make_i_type(BaseAluOpcode::ADD.global_opcode().as_usize(), &d),
                    2 => make_i_type(LessThanOpcode::SLT.global_opcode().as_usize(), &d),
                    3 => make_i_type(LessThanOpcode::SLTU.global_opcode().as_usize(), &d),
                    4 => make_i_type(BaseAluOpcode::XOR.global_opcode().as_usize(), &d),
                    6 => make_i_type(BaseAluOpcode::OR.global_opcode().as_usize(), &d),
                    7 => make_i_type(BaseAluOpcode::AND.global_opcode().as_usize(), &d),
                    1 => {
                        // SLLI
                        let shamt = (d.lo >> 20) & 0x1F;
                        make_shift_imm(ShiftOpcode::SLL.global_opcode().as_usize(), &d, shamt)
                    }
                    5 => {
                        // SRLI / SRAI
                        let shamt = (d.lo >> 20) & 0x1F;
                        let is_arith = (d.lo >> 30) & 1;
                        let opcode = if is_arith != 0 { ShiftOpcode::SRA } else { ShiftOpcode::SRL };
                        make_shift_imm(opcode.global_opcode().as_usize(), &d, shamt)
                    }
                    _ => return None,
                }
            }
            LOAD => {
                let opcode = match d.funct3 {
                    0 => Rv32LoadStoreOpcode::LOADB,
                    1 => Rv32LoadStoreOpcode::LOADH,
                    2 => Rv32LoadStoreOpcode::LOADW,
                    4 => Rv32LoadStoreOpcode::LOADBU,
                    5 => Rv32LoadStoreOpcode::LOADHU,
                    _ => return None,
                };
                make_load(opcode.global_opcode().as_usize(), &d)
            }
            STORE => {
                let opcode = match d.funct3 {
                    0 => Rv32LoadStoreOpcode::STOREB,
                    1 => Rv32LoadStoreOpcode::STOREH,
                    2 => Rv32LoadStoreOpcode::STOREW,
                    _ => return None,
                };
                make_store(opcode.global_opcode().as_usize(), &d)
            }
            BRANCH => {
                let opcode = match d.funct3 {
                    0 => BranchEqualOpcode::BEQ.global_opcode().as_usize(),
                    1 => BranchEqualOpcode::BNE.global_opcode().as_usize(),
                    4 => BranchLessThanOpcode::BLT.global_opcode().as_usize(),
                    5 => BranchLessThanOpcode::BGE.global_opcode().as_usize(),
                    6 => BranchLessThanOpcode::BLTU.global_opcode().as_usize(),
                    7 => BranchLessThanOpcode::BGEU.global_opcode().as_usize(),
                    _ => return None,
                };
                make_branch(opcode, &d)
            }
            LUI => {
                if d.rd == 0 { Some(nop()) }
                else { make_lui(&d) }
            }
            AUIPC => {
                make_auipc(&d)
            }
            JAL => {
                make_jal(&d)
            }
            JALR => {
                make_jalr(&d)
            }
            SYSTEM => {
                let imm = i_imm(d.lo) as u32;
                if d.rd == 0 && d.rs1 == 0 {
                    if imm == 0 {
                        // ECALL → terminate with code 0
                        Some(Instruction {
                            opcode: SystemOpcode::TERMINATE.global_opcode(),
                            ..Default::default()
                        })
                    } else if imm == 1 {
                        // EBREAK
                        Some(unimp())
                    } else {
                        Some(nop())
                    }
                } else {
                    Some(nop())
                }
            }
            MISC_MEM => {
                // FENCE → nop in zkVM
                Some(nop())
            }
            // OpenVM CUSTOM_0 opcode (0x0b) for terminate/phantom/hint/reveal
            0x0b => {
                match d.funct3 {
                    0 => {
                        // TERMINATE
                        let imm = i_imm(d.lo);
                        Some(Instruction {
                            opcode: SystemOpcode::TERMINATE.global_opcode(),
                            c: F::from_u32(imm as u32 & 0xff),
                            ..Default::default()
                        })
                    }
                    _ => Some(nop()),
                }
            }
            _ => return None,
        };

        // Emit instruction without gap. Each 64-bit instruction consumes 2 u32s
        // but occupies only 1 PC slot. Branch offsets from LLVM are for 8-byte
        // instructions, so we halve them in make_branch/make_jal/make_jalr.
        instruction.map(|inst| TranspilerOutput {
            instructions: vec![Some(inst)],
            used_u32s: 2,
        })
    }
}

// Helper functions that create OpenVM instructions from decoded fields

fn make_r_type<F: PrimeField32>(opcode: usize, d: &Decoded64) -> Option<Instruction<F>> {
    if d.rd == 0 {
        return Some(nop());
    }
    Some(Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs1),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs2),
        F::ONE,
        F::ONE,
        F::ZERO,
        F::ZERO,
    ))
}

fn make_i_type<F: PrimeField32>(opcode: usize, d: &Decoded64) -> Option<Instruction<F>> {
    if d.rd == 0 {
        return Some(nop());
    }
    Some(Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs1),
        F::from_u32(i12_to_u24(i_imm(d.lo))),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ))
}

fn make_shift_imm<F: PrimeField32>(opcode: usize, d: &Decoded64, shamt: u32) -> Option<Instruction<F>> {
    if d.rd == 0 {
        return Some(nop());
    }
    Some(Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs1),
        F::from_u32(shamt),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ))
}

fn make_load<F: PrimeField32>(opcode: usize, d: &Decoded64) -> Option<Instruction<F>> {
    let imm = i_imm(d.lo);
    Some(Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs1),
        F::from_u32((imm as u32) & 0xffff),
        F::ONE,
        F::TWO,
        F::from_bool(d.rd != 0),
        F::from_bool(imm < 0),
    ))
}

fn make_store<F: PrimeField32>(opcode: usize, d: &Decoded64) -> Option<Instruction<F>> {
    let imm = s_imm(d.lo);
    Some(Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs2),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs1),
        F::from_u32((imm as u32) & 0xffff),
        F::ONE,
        F::TWO,
        F::ONE,
        F::from_bool(imm < 0),
    ))
}

fn make_branch<F: PrimeField32>(opcode: usize, d: &Decoded64) -> Option<Instruction<F>> {
    // Halve the branch offset: LLVM calculates for 8-byte instructions,
    // but OpenVM uses DEFAULT_PC_STEP=4 per instruction slot.
    let imm = b_imm(d.lo) / 2;
    Some(Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs1),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs2),
        isize_to_field(imm as isize),
        F::ONE,
        F::ONE,
        F::ZERO,
        F::ZERO,
    ))
}

fn make_lui<F: PrimeField32>(d: &Decoded64) -> Option<Instruction<F>> {
    let imm = u_imm(d.lo);
    Some(Instruction::new(
        VmOpcode::from_usize(Rv32JalLuiOpcode::LUI.global_opcode().as_usize()),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::ZERO,
        F::from_u32((imm >> 12) & 0xfffff),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ))
}

fn make_auipc<F: PrimeField32>(d: &Decoded64) -> Option<Instruction<F>> {
    let imm = u_imm(d.lo);
    Some(Instruction::new(
        VmOpcode::from_usize(Rv32AuipcOpcode::AUIPC.global_opcode().as_usize()),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::ZERO,
        F::from_u32((imm >> 12) & 0xfffff),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ))
}

fn make_jal<F: PrimeField32>(d: &Decoded64) -> Option<Instruction<F>> {
    // Halve the jump offset (8-byte instructions → 4-byte PC slots)
    let imm = j_imm(d.lo) / 2;
    Some(Instruction::new(
        VmOpcode::from_usize(Rv32JalLuiOpcode::JAL.global_opcode().as_usize()),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::ZERO,
        isize_to_field(imm as isize),
        F::ONE,
        F::ZERO,
        F::from_bool(d.rd != 0),
        F::ZERO,
    ))
}

fn make_jalr<F: PrimeField32>(d: &Decoded64) -> Option<Instruction<F>> {
    let imm = i_imm(d.lo);
    Some(Instruction::new(
        VmOpcode::from_usize(Rv32JalrOpcode::JALR.global_opcode().as_usize()),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rd),
        F::from_usize(RV32_REGISTER_NUM_LIMBS * d.rs1),
        F::from_u32(i12_to_u24(imm)),
        F::ONE,
        F::ZERO,
        F::from_bool(d.rd != 0),
        F::ZERO,
    ))
}

#[cfg(test)]
#[path = "xregs1024_test.rs"]
mod tests;
