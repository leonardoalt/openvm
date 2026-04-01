#[cfg(test)]
mod tests {
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use crate::xregs1024::{XRegs1024TranspilerExtension, decode_64bit, Decoded64};
    use openvm_transpiler::TranspilerExtension;

    type F = BabyBear;

    /// Encode a 64-bit instruction in our fixup-compatible format.
    /// Returns (lo_u32, hi_u32).
    fn encode_r_type(opcode: u32, funct3: u32, funct7: u32, rd: u32, rs1: u32, rs2: u32) -> (u32, u32) {
        let lo = 0x3F  // marker
            | ((rd & 0x1F) << 7)
            | (funct3 << 12)
            | ((rs1 & 0x1F) << 15)
            | ((rs2 & 0x1F) << 20)
            | (funct7 << 25);
        let hi = (opcode << 10)
            | (((rd >> 5) & 0x1F) << 17)
            | (((rs1 >> 5) & 0x1F) << 22)
            | (((rs2 >> 5) & 0x1F) << 27);
        (lo, hi)
    }

    fn encode_i_type(opcode: u32, funct3: u32, rd: u32, rs1: u32, imm12: i32) -> (u32, u32) {
        let imm = (imm12 as u32) & 0xFFF;
        let lo = 0x3F
            | ((rd & 0x1F) << 7)
            | (funct3 << 12)
            | ((rs1 & 0x1F) << 15)
            | (imm << 20);
        let hi = (opcode << 10)
            | (((rd >> 5) & 0x1F) << 17)
            | (((rs1 >> 5) & 0x1F) << 22);
        (lo, hi)
    }

    #[test]
    fn test_decode_r_type_standard_regs() {
        // ADD x10, x11, x12  (standard registers, should work like normal)
        let (lo, hi) = encode_r_type(0b0110011, 0b000, 0b0000000, 10, 11, 12);
        let d = decode_64bit(lo, hi);
        assert_eq!(d.opcode, 0b0110011);
        assert_eq!(d.funct3, 0);
        assert_eq!(d.funct7, 0);
        assert_eq!(d.rd, 10);
        assert_eq!(d.rs1, 11);
        assert_eq!(d.rs2, 12);
    }

    #[test]
    fn test_decode_r_type_extended_regs() {
        // ADD x100, x200, x300  (extended registers)
        let (lo, hi) = encode_r_type(0b0110011, 0b000, 0b0000000, 100, 200, 300);
        let d = decode_64bit(lo, hi);
        assert_eq!(d.rd, 100);
        assert_eq!(d.rs1, 200);
        assert_eq!(d.rs2, 300);
    }

    #[test]
    fn test_transpiler_extension_detects_marker() {
        let ext = XRegs1024TranspilerExtension;

        // 64-bit instruction: ADD x10, x11, x12
        let (lo, hi) = encode_r_type(0b0110011, 0b000, 0b0000000, 10, 11, 12);
        let stream = vec![lo, hi];
        let result = TranspilerExtension::<F>::process_custom(&ext, &stream);
        assert!(result.is_some(), "Should detect 64-bit instruction");
        let output = result.unwrap();
        assert_eq!(output.used_u32s, 2, "Should consume 2 u32s");
        assert_eq!(output.instructions.len(), 2, "Should emit instruction + gap");
        assert!(output.instructions[0].is_some(), "First should be instruction");
        assert!(output.instructions[1].is_none(), "Second should be gap");
    }

    #[test]
    fn test_transpiler_extension_ignores_standard() {
        let ext = XRegs1024TranspilerExtension;

        // Standard 32-bit instruction (opcode != 0x3F in bits[6:0])
        let standard_add: u32 = 0x00b50533; // add a0, a0, a1
        let stream = vec![standard_add];
        let result = TranspilerExtension::<F>::process_custom(&ext, &stream);
        assert!(result.is_none(), "Should NOT detect standard 32-bit instruction");
    }

    #[test]
    fn test_transpiler_all_formats() {
        let ext = XRegs1024TranspilerExtension;

        // Test I-type: ADDI x50, x60, 42
        let (lo, hi) = encode_i_type(0b0010011, 0b000, 50, 60, 42);
        let stream = vec![lo, hi];
        let result = TranspilerExtension::<F>::process_custom(&ext, &stream);
        assert!(result.is_some(), "ADDI should be decoded");

        // Test load: LW x50, 8(x60)
        let (lo, hi) = encode_i_type(0b0000011, 0b010, 50, 60, 8);
        let stream = vec![lo, hi];
        let result = TranspilerExtension::<F>::process_custom(&ext, &stream);
        assert!(result.is_some(), "LW should be decoded");
    }
}
