use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::Result;
use num_bigint::BigUint;
use openvm_algebra_circuit::*;
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_bigint_circuit::*;
use openvm_circuit::{
    arch::{InitFileGenerator, SystemConfig, VmExecutor},
    derive::VmConfig,
    system::SystemExecutor,
    utils::air_test,
};
use openvm_ecc_circuit::{SECP256K1_MODULUS, SECP256K1_ORDER};
use openvm_instructions::exe::VmExe;
use openvm_platform::memory::MEM_SIZE;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImBuilder, Rv32ImConfig, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    XRegs1024TranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_stark_backend::p3_field::PrimeField32 as _;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};
use test_case::test_case;

type F = BabyBear;

fn get_elf(elf_path: impl AsRef<Path>) -> Result<Elf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    Ok(elf)
}

// An "eyeball test" only: prints the decoded ELF for eyeball inspection
#[test]
fn test_decode_elf() -> Result<()> {
    let elf = get_elf("tests/data/rv32im-empty-program-elf")?;
    dbg!(elf);
    Ok(())
}

// To create ELF directly from .S file, `brew install riscv-gnu-toolchain` and run
// `riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostartfiles -e _start -Ttext 0 fib.S -o
// rv32im-fib-from-as` riscv64-unknown-elf-gcc supports rv32im if you set -march target
#[test_case("tests/data/rv32im-fib-from-as")]
#[test_case("tests/data/rv32im-intrin-from-as")]
fn test_generate_program(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let program = Transpiler::<F>::default()
        .with_extension(Rv32ITranspilerExtension)
        .with_extension(Rv32MTranspilerExtension)
        .with_extension(Rv32IoTranspilerExtension)
        .with_extension(ModularTranspilerExtension)
        .transpile(&elf.instructions)?;
    for instruction in program {
        println!("{instruction:?}");
    }
    Ok(())
}

#[cfg(feature = "aot")]
#[test_case("tests/data/rv32im-exp-from-as")]
fn test_rv32im_aot_pure_runtime(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config.clone())?;

    let interpreter = executor.instance(&exe)?;
    let _interp_state = interpreter.execute(vec![], None)?;

    Ok(())
}
/*
#[cfg(feature = "aot")]
#[test_case("tests/data/rv32im-exp-from-as")]
fn test_rv32im_aot_pure_runtime_with_path(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config.clone())?;

    let interpreter = executor.instance(&exe)?;
    let interp_state = interpreter.execute(vec![], None)?;

    let asm_name = String::from("asm_test_name");
    let mut aot_instance = executor.aot_instance_with_asm_name(&exe, &asm_name)?;
    let aot_state = aot_instance.execute(vec![], None)?;

    assert_eq!(interp_state.instret(), aot_state.instret());
    assert_eq!(interp_state.pc(), aot_state.pc());

    Ok(())
}
    */

#[test_case("tests/data/rv32im-exp-from-as")]
#[test_case("tests/data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config)?;
    let interpreter = executor.instance(&exe)?;
    interpreter.execute(vec![], None)?;
    Ok(())
}

#[test]
fn test_xregs1024_runtime() -> Result<()> {
    let elf = get_elf("tests/data/rv32im-xregs1024-fib")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(XRegs1024TranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config)?;
    let interpreter = executor.instance(&exe)?;
    interpreter.execute(vec![], None)?;
    Ok(())
}

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct Rv32ModularFp2Int256Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub mul: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub modular: ModularExtension,
    #[extension]
    pub fp2: Fp2Extension,
    #[extension]
    pub int256: Int256,
}

impl Rv32ModularFp2Int256Config {
    pub fn new(modular_moduli: Vec<BigUint>, fp2_moduli: Vec<(String, BigUint)>) -> Self {
        Self {
            system: SystemConfig::default(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(modular_moduli),
            fp2: Fp2Extension::new(fp2_moduli),
            int256: Default::default(),
        }
    }
}

impl InitFileGenerator for Rv32ModularFp2Int256Config {
    fn generate_init_file_contents(&self) -> Option<String> {
        Some(format!(
            "{}\n{}\n",
            self.modular.generate_moduli_init(),
            self.fp2.generate_complex_init(&self.modular)
        ))
    }
}

#[test_case("tests/data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let config = Rv32ModularFp2Int256Config::new(
        vec![SECP256K1_MODULUS.clone(), SECP256K1_ORDER.clone()],
        vec![("Secp256k1Coord".to_string(), SECP256K1_MODULUS.clone())],
    );
    let elf = get_elf(elf_path)?;
    let openvm_exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(Fp2TranspilerExtension),
    )?;
    let executor = VmExecutor::new(config)?;
    let interpreter = executor.instance(&openvm_exe)?;
    interpreter.execute(vec![], None)?;
    Ok(())
}

#[test]
fn test_terminate_prove() -> Result<()> {
    let config = Rv32ImConfig::default();
    let elf = get_elf("tests/data/rv32im-terminate-from-as")?;
    let openvm_exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    )?;
    air_test(Rv32ImBuilder, config, openvm_exe);
    Ok(())
}

#[test]
fn test_xregs1024_keccak_comparison() -> Result<()> {
    use std::time::Instant;

    // BASELINE: Standard 32-register RISC-V
    let baseline_elf = get_elf("tests/data/keccak-baseline")?;
    let mut baseline_exe = VmExe::from_elf(
        baseline_elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    // Initialize SP (register x2, byte offset 8 in address space 1) to 0x200400
    let sp_val: u32 = 0x200400;
    for (i, byte) in sp_val.to_le_bytes().iter().enumerate() {
        baseline_exe.init_memory.insert((1, 8 + i as u32), *byte);
    }

    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config.clone())?;

    let t0 = Instant::now();
    let baseline_interpreter = executor.instance(&baseline_exe)?;
    baseline_interpreter.execute(vec![], None)?;
    let baseline_time = t0.elapsed();

    // Count instructions in baseline
    let baseline_insn_count = baseline_exe.program.defined_instructions().len();
    eprintln!("=== BASELINE (32 registers) ===");
    eprintln!("Instructions in program: {}", baseline_insn_count);
    eprintln!("Execution time: {:?}", baseline_time);

    // EXTENDED: 1024-register RISC-V (64-bit encoding)
    // Register BOTH XRegs1024 (for 64-bit compiled code) and standard
    // extensions (for 32-bit inline asm / linker stubs)
    let extended_elf = get_elf("tests/data/keccak-xregs1024")?;
    let mut extended_exe = VmExe::from_elf(
        extended_elf,
        Transpiler::<F>::default()
            .with_extension(XRegs1024TranspilerExtension)
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    // Initialize SP for extended version too
    for (i, byte) in sp_val.to_le_bytes().iter().enumerate() {
        extended_exe.init_memory.insert((1, 8 + i as u32), *byte);
    }

    let t1 = Instant::now();
    let extended_interpreter = executor.instance(&extended_exe)?;
    extended_interpreter.execute(vec![], None)?;
    let extended_time = t1.elapsed();

    let extended_insn_count = extended_exe.program.defined_instructions().len();
    eprintln!("=== EXTENDED (1024 registers) ===");
    eprintln!("Instructions in program: {}", extended_insn_count);
    eprintln!("Execution time: {:?}", extended_time);

    eprintln!("=== COMPARISON ===");
    eprintln!("Instruction reduction: {} -> {} ({:.1}% fewer)",
        baseline_insn_count, extended_insn_count,
        (1.0 - extended_insn_count as f64 / baseline_insn_count as f64) * 100.0);

    Ok(())
}

#[test]
fn test_xregs1024_trace_instructions() -> Result<()> {
    let elf = get_elf("tests/data/keccak-xregs1024")?;
    
    eprintln!("u32 stream ({} words):", elf.instructions.len());
    for (i, w) in elf.instructions.iter().enumerate().take(20) {
        let is64 = w & 0x7f == 0x3f;
        eprintln!("  [{:3}] 0x{:08x} {}", i, w, if is64 { "LO" } else { "HI" });
    }
    
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(XRegs1024TranspilerExtension)
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    
    eprintln!("\nTranspiled ({} instructions):", exe.program.defined_instructions().len());
    for (i, inst) in exe.program.defined_instructions().iter().enumerate().take(15) {
        let inst = inst;
        eprintln!("  [{:3}] op={:4} a={:6} b={:6} c={:10} d={} e={}",
            i, inst.opcode.as_usize(),
            inst.a.as_canonical_u32(), inst.b.as_canonical_u32(),
            inst.c.as_canonical_u32(), inst.d.as_canonical_u32(), inst.e.as_canonical_u32());
    }
    
    Ok(())
}

#[test]
fn test_xregs1024_extended_only() -> Result<()> {
    let elf = get_elf("tests/data/keccak-xregs1024")?;
    
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(XRegs1024TranspilerExtension)
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    
    eprintln!("pc_start: 0x{:08x}", exe.pc_start);
    eprintln!("Program size: {}", exe.program.defined_instructions().len());
    eprintln!("Init memory size: {}", exe.init_memory.len());
    
    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config)?;
    let interpreter = executor.instance(&exe)?;
    interpreter.execute(vec![], None)?;
    
    Ok(())
}
